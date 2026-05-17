function Yhat = predict_external(params, X, A_norm, beta, Ks, Kt)
%PREDICT_EXTERNAL  Stateless forward pass for the TA-Transformer.
%
%   Yhat = PREDICT_EXTERNAL(PARAMS, X, A_NORM, BETA, KS, KT)
%
%   Batched inference: feeds the test tensor in chunks so that the
%   [L,L,N*B] temporal attention scores never exceed GPU memory.

    if isnumeric(X)
        % keep X on CPU until each batch; saves memory
        Bsize = size(X, 4);
    else
        % already dlarray
        Bsize = size(X, 4);
        X = gather(extractdata(X));   % keep on CPU master copy
    end

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
    for k = 1:batchSize:Bsize
        idx = k : min(k + batchSize - 1, Bsize);
        Xb  = dlarray(single(X(:, :, :, idx)));
        if canUseGPU(); Xb = gpuArray(Xb); end
        Yb  = forwardPass(params, Xb, A_norm, beta, 0, Ks, Kt);
        Yb  = gather(extractdata(Yb));
        if isempty(Yhat)
            [H, N, ~] = size(Yb);
            Yhat = zeros(H, N, Bsize, 'single');
        end
        Yhat(:, :, idx) = single(Yb);
        if canUseGPU(); wait(gpuDevice()); end
    end
end

% ============================================================================
% Local mirror of TRAIN_TA_TRANSFORMER's forwardPass.  Kept in sync manually.
% ============================================================================
function Yhat = forwardPass(params, X, A_norm, beta, dropout, Ks, Kt)
    if isdlarray(X); X = stripdims(X); end

    L = size(X, 1); N = size(X, 2);
    F = size(X, 3); B = size(X, 4);
    d = size(params.in_proj, 2);

    Hh = linProj4D(X, params.in_proj, 3);            % [L, N, d, B]

    for k = 1:Ks
        Hp = permute(Hh, [2, 3, 1, 4]);
        Hp = reshape(Hp,  [N, d, L*B]);
        Q  = linProj3D(Hp, params.s(k).Wq);
        Kk = linProj3D(Hp, params.s(k).Wk);
        Vv = linProj3D(Hp, params.s(k).Wv);
        attn = topology_aware_attention(Q, Kk, Vv, double(A_norm), beta);
        attn = linProj3D(attn, params.s(k).Wo);
        attn = reshape(attn, [N, d, L, B]);
        attn = permute(attn, [3, 1, 2, 4]);
        Hh = Hh + attn;
        FF = linProj4D(Hh, params.s(k).ff1, 3);
        FF = max(FF, 0);
        FF = linProj4D(FF, params.s(k).ff2, 3);
        Hh = Hh + FF;
        if dropout > 0
            keep = (rand(size(Hh), 'like', Hh) > dropout);
            Hh   = Hh .* keep ./ (1 - dropout);
        end
    end

    for k = 1:Kt
        Ht = permute(Hh, [1, 3, 2, 4]);
        Ht = reshape(Ht,  [L, d, N*B]);
        Q  = linProj3D(Ht, params.t(k).Wq);
        Kk = linProj3D(Ht, params.t(k).Wk);
        Vv = linProj3D(Ht, params.t(k).Wv);
        scores = pagemtimes(Q, 'none', Kk, 'transpose') / sqrt(d);
        s_max = max(scores, [], 2);
        s_exp = exp(scores - s_max);
        alpha = s_exp ./ sum(s_exp, 2);
        attn   = pagemtimes(alpha, Vv);
        attn   = linProj3D(attn, params.t(k).Wo);
        attn = reshape(attn, [L, d, N, B]);
        attn = permute(attn, [1, 3, 2, 4]);
        Hh = Hh + attn;
        FF = linProj4D(Hh, params.t(k).ff1, 3);
        FF = max(FF, 0);
        FF = linProj4D(FF, params.t(k).ff2, 3);
        Hh = Hh + FF;
        if dropout > 0
            keep = (rand(size(Hh), 'like', Hh) > dropout);
            Hh   = Hh .* keep ./ (1 - dropout);
        end
    end

    % Mean-pool over time -> H future steps (mirrors train_ta_transformer
    % forwardPass after Tier1-3 fix).
    pooled = mean(Hh, 1);                             % [1, N, d, B]
    pooled = reshape(pooled, [N, d, B]);              % [N, d, B]
    Yhat = linProj3D(pooled, params.head);            % [N, H_out, B]
    Yhat = permute(Yhat, [2, 1, 3]);                  % [H_out, N, B]
end

function Y = linProj4D(X, W, featureAxis)
    sz = size(X);
    F  = sz(featureAxis);
    d  = size(W, 2);
    perm = [featureAxis, setdiff(1:numel(sz), featureAxis)];
    Xp = permute(X, perm);
    Xf = reshape(Xp, [F, numel(Xp)/F]);
    Yf = W.' * Xf;
    rest = sz(perm(2:end));
    Yp = reshape(Yf, [d, rest]);
    Y  = ipermute(Yp, perm);
end

function Y = linProj3D(X, W)
    [N, F, P] = size(X);
    d = size(W, 2);
    Xp = permute(X, [2, 1, 3]);
    Xf = reshape(Xp, [F, N*P]);
    Yf = W.' * Xf;
    Yp = reshape(Yf, [d, N, P]);
    Y  = permute(Yp, [2, 1, 3]);
end
