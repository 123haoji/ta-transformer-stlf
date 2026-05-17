function metrics = evaluate_metrics(yhat, ytrue, feats, opts)
%EVALUATE_METRICS  Compute MAPE, RMSE, R^2, and peak-hour MAPE.
%
%   metrics = EVALUATE_METRICS(YHAT, YTRUE, FEATS, OPTS)
%
%   INPUTS
%     yhat, ytrue   [H, N, S] normalized predictions and targets
%     feats         output of BUILD_FEATURES (for norm stats and ts_y)
%     opts.idx      indices into feats.ts_y to evaluate (default: test)
%     opts.peakHours default [19 20 21 22]
%
%   OUTPUTS
%     metrics.MAPE       scalar percent
%     metrics.RMSE       scalar (kW)
%     metrics.R2         scalar
%     metrics.PH_MAPE    scalar percent restricted to peak hours
%     metrics.perHorizon struct of arrays length H
%     metrics.perBus     struct of arrays length N

    arguments
        yhat   single
        ytrue
        feats  (1,1) struct
        opts.idx       (1,:) double = feats.idx.test
        opts.peakHours (1,:) double = [19 20 21 22]
    end

    % bring back to physical scale (kW).  Support both legacy (scalar) and
    % per-bus (vector) normalization stats.  Per-bus stats are stored as
    % row vectors of length N by build_features.m.
    mu_load = feats.norm.mu_load;
    sd_load = feats.norm.sd_load;
    if isscalar(mu_load)
        yhat_kw  = double(yhat)  * sd_load + mu_load;
        ytrue_kw = double(ytrue) * sd_load + mu_load;
    else
        % broadcast vector [1 x N] -> [1 x N x 1] over [H x N x S]
        N_norm = numel(mu_load);
        mu3 = reshape(double(mu_load), [1, N_norm, 1]);
        sd3 = reshape(double(sd_load), [1, N_norm, 1]);
        yhat_kw  = double(yhat)  .* sd3 + mu3;
        ytrue_kw = double(ytrue) .* sd3 + mu3;
    end

    [H, N, S] = size(yhat_kw);

    % --- exclude zero-load buses from metrics --------------------------------
    % Buses that carry no load (e.g., when an aggregation protocol assigns
    % fewer clusters than the number of available load buses) produce
    % meaningless 100%+ MAPE because the true value is essentially zero
    % everywhere on that bus.  We detect and drop them BEFORE computing
    % metrics, and report the count separately for transparency.
    bus_mean_load = squeeze(mean(abs(ytrue_kw), [1 3], 'omitnan'));   % [N x 1]
    activeBus = bus_mean_load > max(1e-3, 0.01 * max(bus_mean_load)); % logical [N x 1]
    nDropped  = sum(~activeBus);
    if nDropped > 0
        yhat_kw  = yhat_kw(:,  activeBus, :);
        ytrue_kw = ytrue_kw(:, activeBus, :);
        [H, N, S] = size(yhat_kw);
        fprintf('  (evaluate_metrics: excluded %d/%d zero-load buses)\n', ...
                nDropped, numel(activeBus));
    end

    diff  = yhat_kw - ytrue_kw;
    absDf = abs(diff);

    % Scale-aware epsilon: at least 1 kW or 1% of dataset peak, whichever larger.
    % This keeps MAPE meaningful on small loads but does not artificially deflate
    % errors on normal-range data.
    peak  = max(abs(ytrue_kw(:)), [], 'omitnan');
    eps0  = max(1, 0.01 * double(peak));

    % Clip per-sample APE at 500% to keep aggregate metric stable on
    % near-zero targets (e.g., the zero-shot microgrid case).
    ape   = min( absDf ./ max(abs(ytrue_kw), eps0), 5.0 );

    % Symmetric MAPE (always bounded in [0, 2]) as a robust alternative
    smape = 2 * absDf ./ max(abs(yhat_kw) + abs(ytrue_kw), eps0);

    metrics.MAPE  = mean(ape,   'all', 'omitnan') * 100;
    metrics.SMAPE = mean(smape, 'all', 'omitnan') * 100;
    metrics.RMSE  = sqrt(mean(diff.^2, 'all', 'omitnan'));

    % --- WAPE (volume-weighted absolute percentage error) -------------------
    % WAPE = sum(|err|) / sum(|true|).  Unlike MAPE, WAPE is not inflated by
    % low-load buses (each error is weighted by the magnitude of the true
    % value it accompanies).  Industry-standard metric for skewed/clustered
    % load datasets; recommended by Baeldung (CS) and the M-competitions
    % literature whenever some series carry much lower volume than others.
    metrics.WAPE  = sum(absDf, 'all', 'omitnan') / ...
                    max(sum(abs(ytrue_kw), 'all', 'omitnan'), eps0) * 100;

    % --- per-bus WAPE (useful for ablation tables) --------------------------
    metrics.perBus.WAPE = squeeze(sum(absDf, [1 3], 'omitnan')) ./ ...
                          max(squeeze(sum(abs(ytrue_kw), [1 3], 'omitnan')), eps0) * 100;

    ss_res = sum(diff.^2, 'all', 'omitnan');
    ybar   = mean(ytrue_kw, 'all', 'omitnan');
    ss_tot = sum((ytrue_kw - ybar).^2, 'all', 'omitnan');
    metrics.R2 = 1 - ss_res / max(ss_tot, eps0);

    % peak-hour mask using the timestamp of each forecast position
    ts = feats.ts_y(:, opts.idx);                  % [H x S]
    hh = hour(ts);
    pkMask  = ismember(hh, opts.peakHours);        % [H x S]
    pkMask3 = repmat(reshape(pkMask, [H, 1, S]), 1, N, 1);
    ape_pk  = ape(pkMask3);
    if ~isempty(ape_pk)
        metrics.PH_MAPE = mean(ape_pk, 'all', 'omitnan') * 100;
    else
        metrics.PH_MAPE = NaN;
    end

    metrics.perHorizon.MAPE = squeeze(mean(ape,    [2 3], 'omitnan')) * 100;
    metrics.perHorizon.RMSE = squeeze(sqrt(mean(diff.^2, [2 3], 'omitnan')));
    metrics.perBus.MAPE     = squeeze(mean(ape,    [1 3], 'omitnan')) * 100;
    metrics.perBus.RMSE     = squeeze(sqrt(mean(diff.^2, [1 3], 'omitnan')));
    metrics.n_samples       = S;
    metrics.n_buses         = N;
    metrics.horizons        = (1:H)';
    metrics.scale_eps_kW    = eps0;
    metrics.peak_load_kW    = double(peak);
end
