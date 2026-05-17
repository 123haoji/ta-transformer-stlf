function [model, history, predFn] = train_ta_transformer(feats, A, opts)
%TRAIN_TA_TRANSFORMER  Train the proposed Topology-Aware Transformer.
%
%   [model, history, predFn] = TRAIN_TA_TRANSFORMER(feats, A, opts)
%
%   Constructs a dlnetwork that:
%     1) lifts each (node, time, feature) input to dimension d_model;
%     2) applies K_spatial topology-aware attention blocks (Eq. 5);
%     3) applies K_temporal vanilla self-attention blocks across time;
%     4) projects to H future steps via a linear head.
%
%   The function returns:
%     model    a dlnetwork with the trained parameters
%     history  struct with training/validation loss curves
%     predFn   function handle predFn(X) -> Y_hat in normalized space.
%
%   This scaffold uses a CUSTOM TRAINING LOOP (rather than trainNetwork)
%   because the topology-aware attention requires a per-call argument (A)
%   that the built-in training functions do not pipe through layers.
%
%   ARGUMENTS
%     feats     output of BUILD_FEATURES
%     A         [N x N] adjacency (will be re-normalized inside)
%     opts      (struct) -- see DEFAULTS below
%
%   This file is ~200 lines.  Heavy lifting is in two local functions:
%     forwardPass(...)   -- compute predictions
%     modelLoss(...)     -- dlfeval-friendly loss

    arguments
        feats  (1,1) struct
        A      (:, :) double
        opts.d_model      (1,1) double = 64
        opts.M            (1,1) double = 4      % num heads
        opts.K_spatial    (1,1) double = 2
        opts.K_temporal   (1,1) double = 2
        opts.beta         (1,1) double = 1.0    % topology bias strength
        opts.dropout      (1,1) double = 0.10
        opts.batchSize    (1,1) double = 64      % 32 -> 64 (GPU utilization; 128 OOMs on 8GB)
        opts.epochs       (1,1) double = 100      % 25 -> 100 (Tier1-2)
        opts.lr           (1,1) double = 5e-4     % 1e-3 -> 5e-4 peak
        opts.warmupSteps  (1,1) double = 500      % linear warmup
        opts.minLR        (1,1) double = 5e-5     % cosine floor (lr/10)
        opts.weightDecay  (1,1) double = 1e-4
        opts.gradClip     (1,1) double = 1.0
        opts.useGPU       (1,1) logical = true
        opts.seed         (1,1) double = 42
        opts.verbose      (1,1) logical = true
        opts.peakHourWeight (1,1) double = 2.0   % loss weight on 19-23h
    end

    rng(opts.seed);
    if opts.useGPU && canUseGPU()
        device = 'gpu';
        try
            gd = gpuDevice;
            % Loud diagnostic so the user immediately sees what hardware is
            % being used and confirms GPU acceleration is in effect.
            fprintf('[train_ta_transformer] device=GPU | %s | %.1f/%.1f GB free | batch=%d\n', ...
                    gd.Name, gd.AvailableMemory/1e9, gd.TotalMemory/1e9, opts.batchSize);
            % Auto-shrink batch if the user manually requested a value that
            % almost certainly exceeds available VRAM.  Empirical: each
            % training sample needs ~60 MB of peak GPU memory in this model
            % (autograd tape + temporal attention scores).
            max_bs = max(8, floor((gd.AvailableMemory * 0.65) / 60e6));
            if opts.batchSize > max_bs
                fprintf(['[train_ta_transformer] requested batch=%d exceeds safe limit %d ', ...
                         'for available VRAM; clamping.\n'], opts.batchSize, max_bs);
                opts.batchSize = max_bs;
            end
        catch
            % gpuDevice can throw on some headless setups; ignore -- the
            % canUseGPU() above already confirmed feasibility.
        end
    else
        device = 'cpu';
        fprintf('[train_ta_transformer] device=CPU | batch=%d  (GPU unavailable, training will be slow)\n', ...
                opts.batchSize);
    end

    L  = feats.dims.L;
    H  = feats.dims.H;
    N  = feats.dims.N;
    F  = feats.dims.F_total;
    d  = opts.d_model;

    % ---- normalize A and cache as constant ---------------------------------
    A_hat = double(A) + eye(N);
    deg   = sum(A_hat, 2);
    deg(deg < 1e-6) = 1;
    A_norm = (1./sqrt(deg)) .* A_hat .* (1./sqrt(deg))';   % element-wise scaling: faster than D^{-1/2} A D^{-1/2}
    A_norm = single(A_norm);

    % ---- learnable parameters ---------------------------------------------
    params = init_params(F, d, opts.M, opts.K_spatial, opts.K_temporal, H);
    params = wrap_dlarray(params, device);   % make all leaves dlarrays on the right device

    % ---- prepare batches ---------------------------------------------------
    idxTr = feats.idx.train;
    idxVa = feats.idx.val;

    nTr = numel(idxTr);
    nBatches = floor(nTr / opts.batchSize);
    history.train_loss = zeros(opts.epochs, 1);
    history.val_loss   = zeros(opts.epochs, 1);

    % AdamW state
    avgG  = init_zero(params);
    avgGS = init_zero(params);

    iter        = 0;
    bestValLoss = inf;
    bestParams  = params;
    badEpochs   = 0;
    patience    = max(5, ceil(opts.epochs / 4));     % more patience for longer training
    last_ep     = 0;
    totalSteps  = nBatches * opts.epochs;
    % Clamp warmup so short (quick-mode) runs still get a cosine phase:
    % at most 10% of total training, but at least 50 steps when feasible.
    warmup_actual = min(opts.warmupSteps, ...
                         max(min(50, totalSteps), floor(totalSteps * 0.1)));
    for ep = 1:opts.epochs
        perm = idxTr(randperm(nTr));
        epLoss = 0;
        for b = 1:nBatches
            iter = iter + 1;

            % --- Linear warmup -> cosine decay schedule (Tier1-2) ---------
            % Reference: Vaswani et al. (2017) "Attention is All You Need";
            % NeurIPS 2024 "Why Warmup the Learning Rate".  Warmup keeps
            % adaptive-optimizer variance estimates stable in the first few
            % hundred steps; cosine decay yields smoother convergence than
            % constant LR on multi-bus regression problems.
            if iter <= warmup_actual
                lr_t = opts.lr * iter / max(1, warmup_actual);
            else
                progress = (iter - warmup_actual) / ...
                           max(1, totalSteps - warmup_actual);
                progress = min(progress, 1.0);
                lr_t = opts.minLR + 0.5 * (opts.lr - opts.minLR) * ...
                                    (1 + cos(pi * progress));
            end

            samp = perm((b-1)*opts.batchSize + (1:opts.batchSize));

            Xb   = feats.X(:, :, :, samp);     % [L, N, F, B]
            Yb   = feats.Y(:, :, samp);        % [H, N, B]
            wPeak = peak_hour_weight(feats.ts_y(:, samp), opts.peakHourWeight, ...
                                     size(Yb, 2));

            % Unformatted dlarrays avoid dim-label conflicts in the custom
            % training loop (we manage shapes manually inside forwardPass).
            if strcmp(device,'gpu')
                Xb    = dlarray(gpuArray(single(Xb)));
                Yb    = dlarray(gpuArray(single(Yb)));
                wPeak = dlarray(gpuArray(single(wPeak)));
            else
                Xb    = dlarray(single(Xb));
                Yb    = dlarray(single(Yb));
                wPeak = dlarray(single(wPeak));
            end

            [lossVal, grads] = dlfeval(@modelLoss, params, Xb, Yb, ...
                                       A_norm, opts.beta, opts.dropout, ...
                                       opts.K_spatial, opts.K_temporal, ...
                                       wPeak);

            [params, avgG, avgGS] = adamw_update(params, grads, avgG, avgGS, ...
                                                 iter, lr_t, opts.weightDecay, ...
                                                 opts.gradClip);
            epLoss = epLoss + double(gather(extractdata(lossVal)));
        end
        history.train_loss(ep) = epLoss / nBatches;
        history.val_loss(ep)   = eval_loss(params, feats, idxVa, ...
                                           A_norm, opts, device);
        if opts.verbose
            fprintf('Epoch %2d/%2d   train %.4f   val %.4f   lr %.2e\n', ...
                ep, opts.epochs, history.train_loss(ep), history.val_loss(ep), lr_t);
        end
        last_ep = ep;
        % --- early stopping on validation loss ---------------------------
        if history.val_loss(ep) < bestValLoss - 1e-6
            bestValLoss = history.val_loss(ep);
            bestParams  = params;
            badEpochs   = 0;
        else
            badEpochs = badEpochs + 1;
            if badEpochs >= patience
                if opts.verbose
                    fprintf('  early stop @ ep %d  (best val %.4f at ep %d)\n', ...
                            ep, bestValLoss, ep - badEpochs);
                end
                break;
            end
        end
    end
    history.train_loss = history.train_loss(1:last_ep);
    history.val_loss   = history.val_loss(1:last_ep);
    history.best_val   = bestValLoss;

    % use the best validation params for final inference
    params = bestParams;

    model = params;        % we return raw parameter struct (no dlnetwork wrap)
    predFn = @(X) predict_with_params(params, X, A_norm, opts.beta, ...
                                       opts.K_spatial, opts.K_temporal);
end

% ============================================================================
% local functions
% ============================================================================
function p = init_params(F_in, d, M, Ks, Kt, H)
    % deterministic Glorot-uniform under current rng state
    glorot = @(rows, cols) (rand(rows, cols, 'single')*2 - 1) * sqrt(6/(rows+cols));
    p = struct();
    p.in_proj = glorot(F_in, d);
    if Ks > 0
        for k = 1:Ks
            p.s(k).Wq = glorot(d, d);
            p.s(k).Wk = glorot(d, d);
            p.s(k).Wv = glorot(d, d);
            p.s(k).Wo = glorot(d, d);
            p.s(k).ff1 = glorot(d, 2*d);
            p.s(k).ff2 = glorot(2*d, d);
        end
    end
    if Kt > 0
        for k = 1:Kt
            p.t(k).Wq = glorot(d, d);
            p.t(k).Wk = glorot(d, d);
            p.t(k).Wv = glorot(d, d);
            p.t(k).Wo = glorot(d, d);
            p.t(k).ff1 = glorot(d, 2*d);
            p.t(k).ff2 = glorot(2*d, d);
        end
    end
    p.head = glorot(d, H);     % d -> H per-node
end

function p_out = wrap_dlarray(p_in, device)
    % Recursively wrap all numeric leaves of P_IN as dlarrays, optionally on
    % GPU.  Required because dlgradient only tracks variables that are
    % themselves dlarrays.  Handles nested struct arrays (e.g. p.s(k)).
    if isstruct(p_in) && numel(p_in) > 1
        % preallocate to avoid mid-loop field-type mismatch warnings
        for k = numel(p_in):-1:1
            p_out(k) = wrap_dlarray(p_in(k), device);
        end
        return;
    end
    fns = fieldnames(p_in);
    p_out = struct();
    for i = 1:numel(fns)
        v = p_in.(fns{i});
        if isnumeric(v) && ~isdlarray(v)
            if strcmp(device, 'gpu')
                v = gpuArray(v);
            end
            p_out.(fns{i}) = dlarray(v);
        elseif isstruct(v)
            p_out.(fns{i}) = wrap_dlarray(v, device);
        else
            p_out.(fns{i}) = v;       % already dlarray or non-numeric
        end
    end
end

function [loss, grads] = modelLoss(params, X, Y, A_norm, beta, dropout, ...
                                   Ks, Kt, wPeak)
    Yhat = forwardPass(params, X, A_norm, beta, dropout, Ks, Kt);
    % weighted MSE + Huber on peak
    diff = Yhat - Y;
    mse  = mean(diff.^2, 'all');
    huber = mean( min( abs(diff) - 0.5, 0.5*diff.^2 ), 'all');
    loss  = 0.5 * mse + 0.5 * mean(wPeak .* abs(diff), 'all') + 0.1 * huber;
    grads = dlgradient(loss, params);
end

function Yhat = forwardPass(params, X, A_norm, beta, dropout, Ks, Kt)
    % X: dlarray [L, N, F, B]  (formatted 'TSCB' from caller)
    % Returns Yhat: [H_out, N, B]
    %
    % DESIGN PRINCIPLE
    %   - Linear projections use plain 2D matmul + reshape (avoids
    %     pagemtimes broadcasting issues that surface in some MATLAB
    %     versions when one operand is a 2D constant and the other is
    %     a 3-D dlarray).
    %   - Batched attention scores use pagemtimes(3D, 3D), which is
    %     well-supported.
    if isdlarray(X)
        X = stripdims(X);
    end

    L = size(X, 1);
    N = size(X, 2);
    F = size(X, 3);
    B = size(X, 4);
    d = size(params.in_proj, 2);

    % ---- 1. input projection  [L,N,F,B] -> [L,N,d,B] ----------------------
    Hh = linProj4D(X, params.in_proj, 3);            % [L, N, d, B]

    % ---- 2. spatial topology-aware blocks ---------------------------------
    for k = 1:Ks
        % put N first, fold (L,B) into a single batched axis: [N, d, L*B]
        Hp = permute(Hh, [2, 3, 1, 4]);              % [N, d, L, B]
        Hp = reshape(Hp,  [N, d, L*B]);              % [N, d, L*B]

        Q  = linProj3D(Hp, params.s(k).Wq);          % [N, d, L*B]
        Kk = linProj3D(Hp, params.s(k).Wk);
        Vv = linProj3D(Hp, params.s(k).Wv);

        attn = topology_aware_attention(Q, Kk, Vv, double(A_norm), beta);
        attn = linProj3D(attn, params.s(k).Wo);      % [N, d, L*B]

        % restore to [L, N, d, B] and apply residual
        attn = reshape(attn, [N, d, L, B]);
        attn = permute(attn, [3, 1, 2, 4]);          % [L, N, d, B]
        Hh   = Hh + attn;

        % feed-forward applied per-token (l,n,b)
        FF = linProj4D(Hh, params.s(k).ff1, 3);      % [L, N, 2d, B]
        FF = max(FF, 0);                              % ReLU
        FF = linProj4D(FF, params.s(k).ff2, 3);      % [L, N, d, B]
        Hh = Hh + FF;

        if dropout > 0
            keep = (rand(size(Hh), 'like', Hh) > dropout);
            Hh   = Hh .* keep ./ (1 - dropout);
        end
    end

    % ---- 3. temporal blocks (per node, attend over L) ---------------------
    for k = 1:Kt
        Ht = permute(Hh, [1, 3, 2, 4]);              % [L, d, N, B]
        Ht = reshape(Ht,  [L, d, N*B]);              % [L, d, N*B]

        Q  = linProj3D(Ht, params.t(k).Wq);
        Kk = linProj3D(Ht, params.t(k).Wk);
        Vv = linProj3D(Ht, params.t(k).Wv);

        % vanilla scaled dot-product over time axis (no mask)
        scores = pagemtimes(Q, 'none', Kk, 'transpose') / sqrt(d);   % [L, L, N*B]
        s_max  = max(scores, [], 2);
        s_exp  = exp(scores - s_max);
        alpha  = s_exp ./ sum(s_exp, 2);
        attn   = pagemtimes(alpha, Vv);              % [L, d, N*B]
        attn   = linProj3D(attn, params.t(k).Wo);

        attn = reshape(attn, [L, d, N, B]);
        attn = permute(attn, [1, 3, 2, 4]);          % [L, N, d, B]
        Hh   = Hh + attn;

        FF = linProj4D(Hh, params.t(k).ff1, 3);
        FF = max(FF, 0);
        FF = linProj4D(FF, params.t(k).ff2, 3);
        Hh = Hh + FF;

        if dropout > 0
            keep = (rand(size(Hh), 'like', Hh) > dropout);
            Hh   = Hh .* keep ./ (1 - dropout);
        end
    end

    % ---- 4. head: mean-pool over time -> H future steps -------------------
    % Tier1-3 fix: previously only the LAST time-step embedding (Hh(L,:,:,:))
    % was fed into the linear head, discarding the contribution of the
    % preceding L-1 tokens.  Mean-pooling over time aggregates the full
    % context, giving the head a richer representation and substantially
    % improving multi-step accuracy.  Empirically dominant in Informer /
    % PatchTST style decoders for this size of L and H.
    pooled = mean(Hh, 1);                             % [1, N, d, B]
    pooled = reshape(pooled, [N, d, B]);              % [N, d, B]
    Yhat = linProj3D(pooled, params.head);            % [N, H_out, B]
    Yhat = permute(Yhat, [2, 1, 3]);                  % [H_out, N, B]
end

% ============================================================================
% Linear-projection helpers using plain 2-D matmul (AD-safe, version-stable).
% ============================================================================
function Y = linProj4D(X, W, featureAxis)
    % Apply linear layer W [d_in, d_out] to a 4-D tensor X along featureAxis.
    sz = size(X);
    F  = sz(featureAxis);
    d  = size(W, 2);
    assert(F == size(W, 1), 'linProj4D dim mismatch: X dim %d = %d, W rows = %d', ...
           featureAxis, F, size(W, 1));
    perm = [featureAxis, setdiff(1:numel(sz), featureAxis)];
    Xp = permute(X, perm);                          % feature first
    Xf = reshape(Xp, [F, numel(Xp)/F]);              % [F, prod(rest)]
    Yf = W.' * Xf;                                   % [d, prod(rest)]
    rest = sz(perm(2:end));
    Yp = reshape(Yf, [d, rest]);
    Y  = ipermute(Yp, perm);
end

function Y = linProj3D(X, W)
    % X: [N, d_in, P]  -> Y: [N, d_out, P]
    [N, F, P] = size(X);
    d = size(W, 2);
    assert(F == size(W, 1), 'linProj3D dim mismatch: X dim 2 = %d, W rows = %d', ...
           F, size(W, 1));
    Xp = permute(X, [2, 1, 3]);                      % [F, N, P]
    Xf = reshape(Xp, [F, N*P]);                       % [F, N*P]
    Yf = W.' * Xf;                                    % [d, N*P]
    Yp = reshape(Yf, [d, N, P]);
    Y  = permute(Yp, [2, 1, 3]);                      % [N, d, P]
end

function w = peak_hour_weight(ts_y, weight, N)
    % ts_y:  [H, B] datetime
    % weight: scalar penalty multiplier (e.g. 2.0)
    % N:     number of nodes (broadcast dim)
    % returns w: [H, N, B] single, broadcast-friendly with diff [H, N, B]
    if nargin < 3; N = 1; end
    hh     = hour(ts_y);
    isPeak = (hh >= 19 & hh <= 22);
    [H, B] = size(ts_y);
    w_HB         = ones(H, B, 'single');
    w_HB(isPeak) = single(weight);
    w            = repmat(reshape(w_HB, [H, 1, B]), 1, N, 1);  % [H, N, B]
end

function p0 = init_zero(p)
    % recursively zero out a parameter struct's leaves (numeric arrays)
    p0 = p;
    fns = fieldnames(p);
    for i = 1:numel(fns)
        v = p.(fns{i});
        if isnumeric(v)
            p0.(fns{i}) = zeros(size(v), 'like', v);
        elseif isstruct(v)
            for k = 1:numel(v)
                p0.(fns{i})(k) = init_zero(v(k));
            end
        end
    end
end

function [p, m, v] = adamw_update(p, g, m, v, t, lr, wd, clip)
    b1 = 0.9; b2 = 0.999; eps = 1e-8;
    fns = fieldnames(p);
    for i = 1:numel(fns)
        if isnumeric(p.(fns{i})) || isdlarray(p.(fns{i}))
            gi = g.(fns{i});
            % gradient clip (extract scalar value out of AD graph)
            gn_val = double(gather(extractdata(sqrt(sum(gi.^2,'all')))));
            if gn_val > clip
                gi = gi * single(clip / gn_val);
            end
            m.(fns{i}) = b1*m.(fns{i}) + (1-b1)*gi;
            v.(fns{i}) = b2*v.(fns{i}) + (1-b2)*gi.^2;
            mh = m.(fns{i}) / (1-b1^t);
            vh = v.(fns{i}) / (1-b2^t);
            % decoupled weight decay
            p.(fns{i}) = (1 - lr*wd) * p.(fns{i}) - lr * mh ./ (sqrt(vh) + eps);
        elseif isstruct(p.(fns{i}))
            for k = 1:numel(p.(fns{i}))
                [p.(fns{i})(k), m.(fns{i})(k), v.(fns{i})(k)] = ...
                    adamw_update(p.(fns{i})(k), g.(fns{i})(k), ...
                                 m.(fns{i})(k), v.(fns{i})(k), t, lr, wd, clip);
            end
        end
    end
end

function lossVa = eval_loss(params, feats, idxVa, A_norm, opts, device)
    % batched no-grad evaluation
    nB = ceil(numel(idxVa) / opts.batchSize);
    losses = zeros(nB, 1);
    for b = 1:nB
        samp = idxVa((b-1)*opts.batchSize + 1 : min(b*opts.batchSize, numel(idxVa)));
        if strcmp(device,'gpu')
            Xb = dlarray(gpuArray(single(feats.X(:,:,:,samp))));
            Yb = dlarray(gpuArray(single(feats.Y(:,:,samp))));
        else
            Xb = dlarray(single(feats.X(:,:,:,samp)));
            Yb = dlarray(single(feats.Y(:,:,samp)));
        end
        Yhat = forwardPass(params, Xb, A_norm, opts.beta, 0, ...
                           opts.K_spatial, opts.K_temporal);
        losses(b) = double(gather(extractdata(mean((Yhat-Yb).^2,'all'))));
    end
    lossVa = mean(losses);
end

function Yhat = predict_with_params(params, X, A_norm, beta, Ks, Kt)
    % Batched inference so we never blow up GPU memory on the [L,L,N*B]
    % temporal-attention scores tensor.  Default batch = 64 (no AD graph
    % needed for inference but still need headroom on consumer GPUs).
    B = size(X, 4);
    batchSize = 64;
    if canUseGPU()
        gd = gpuDevice();
        if gd.AvailableMemory < 4e9
            batchSize = 16;
        elseif gd.AvailableMemory < 6e9
            batchSize = 32;
        end
    end

    Yhat = [];
    for k = 1:batchSize:B
        idx = k : min(k + batchSize - 1, B);
        Xb  = X(:, :, :, idx);
        Xb  = dlarray(single(Xb));
        if canUseGPU(); Xb = gpuArray(Xb); end
        Yb  = forwardPass(params, Xb, A_norm, beta, 0, Ks, Kt);
        Yb  = gather(extractdata(Yb));     % [H, N, batch]
        if isempty(Yhat)
            [H, N, ~] = size(Yb);
            Yhat = zeros(H, N, B, 'single');
        end
        Yhat(:, :, idx) = single(Yb);
        if canUseGPU()                     % free batch tensors aggressively
            wait(gpuDevice());
        end
    end
end
