function [model, yhat, fitTime] = train_arima(feats, opts)
%TRAIN_ARIMA  Per-bus ARIMA baseline with AICc auto-order selection.
%
%   [model, yhat, fitTime] = TRAIN_ARIMA(feats, opts) fits one ARIMA(p,d,q)
%   model per bus on the training portion of feats.X(:,:,1,:), selecting
%   (p,d,q) by minimum AICc over a small grid.  Recursive H-step forecast.
%
%   The function deliberately keeps the per-bus setup independent so the
%   baseline reflects what a practitioner would do without spatial coupling.

    arguments
        feats (1,1) struct
        opts.pRange  (1, :) double = 0:3
        opts.qRange  (1, :) double = 0:3
        opts.diffD   (1, 1) double = 1
        opts.verbose (1, 1) logical = true
    end

    L = feats.dims.L;   H = feats.dims.H;
    N = feats.dims.N;   nTest = numel(feats.idx.test);

    % --- per-bus norm stats (compatible with both scalar legacy & vector
    %     per-bus forms produced by build_features.m).
    mu_v = feats.norm.mu_load(:);                       % [N x 1] (or [1 x 1])
    sd_v = feats.norm.sd_load(:);
    if isscalar(mu_v); mu_v = repmat(mu_v, N, 1); end
    if isscalar(sd_v); sd_v = repmat(sd_v, N, 1); end

    % De-normalize back to kW so AICc magnitudes are meaningful.
    % feats.X(:,:,1,idx.train) is [L,N,1,nTr]; squeeze -> [L,N,nTr].
    % Broadcast mu_v/sd_v on the N dimension.
    mu_X = reshape(mu_v, [1, N, 1]);
    sd_X = reshape(sd_v, [1, N, 1]);
    Xtr  = squeeze(feats.X(:, :, 1, feats.idx.train)) .* sd_X + mu_X;
    % For ARIMA we need contiguous training history: rebuild a single long
    % series per bus from training-sample contexts (concatenate the last
    % element of each context window to maintain stride).
    tic;
    model = cell(N, 1);
    bestOrd = zeros(N, 3);
    yhat   = zeros(H, N, nTest, 'single');
    for i = 1:N
        % aggregate training context windows into a series (last point per sample)
        ySeries = squeeze(Xtr(end, i, :));     % [nTr, 1]
        % de-NaN
        ySeries = fillmissing(ySeries, 'linear');
        % select order
        bestAICc = inf;  bestMdl = [];  bestP = 0; bestQ = 0;
        for p = opts.pRange
            for q = opts.qRange
                try
                    Mdl = arima(p, opts.diffD, q);
                    EstMdl = estimate(Mdl, ySeries, 'Display', 'off');
                    summary = summarize(EstMdl);
                    aic = summary.AIC;
                    n   = numel(ySeries);
                    k   = p + q + 1;
                    aicc = aic + 2*k*(k+1)/(n-k-1);
                    if aicc < bestAICc
                        bestAICc = aicc;  bestMdl = EstMdl;
                        bestP = p; bestQ = q;
                    end
                catch
                    continue;
                end
            end
        end
        model{i}      = bestMdl;
        bestOrd(i, :) = [bestP, opts.diffD, bestQ];

        % forecast for every test sample (in physical kW units)
        Xte_i = squeeze(feats.X(:, i, 1, feats.idx.test)) * sd_v(i) + mu_v(i);
        for s = 1:nTest
            ctx = Xte_i(:, s);
            try
                yhat_i = forecast(bestMdl, H, 'Y0', ctx);
            catch
                yhat_i = repmat(ctx(end), H, 1);     % fallback: persistence
            end
            yhat(:, i, s) = single(yhat_i);
        end

        if opts.verbose && mod(i,5) == 0
            fprintf('[ARIMA] bus %d/%d  order (%d,%d,%d)  AICc %.1f\n', ...
                i, N, bestP, opts.diffD, bestQ, bestAICc);
        end
    end
    fitTime = toc;
    % re-normalize yhat to match other models' (normalized) output convention
    mu_Y = reshape(single(mu_v), [1, N, 1]);
    sd_Y = reshape(single(sd_v), [1, N, 1]);
    yhat = (yhat - mu_Y) ./ sd_Y;
end
