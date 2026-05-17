function exp2b_horizons(quick)
%EXP2B_HORIZONS  Multi-horizon evaluation at the system-aggregate level.
%
%   Implements the horizon-sensitivity table promised in Section 6.3.2.
%   Following industry STLF benchmarking conventions:
%     - Rows = models (ARIMA, CNN-LSTM, Transformer, TA-Transformer)
%     - Cols = forecast horizons H in {1, 4, 16, 96}
%             corresponding to 15-min, 1-h, 4-h, 24-h ahead.
%     - Sub-cols = {MAPE, WAPE, RMSE, R^2}
%
%   References:
%     Industry KPI standard: Horizon-Europe load forecasting projects
%     report 15-min + day-ahead pairs (arXiv 2506.04294).
%     Lewis qualitative bands: < 10% highly accurate, 10-20% good,
%     20-50% reasonable, > 50% inaccurate (Lewis 1982).
%
%   USAGE
%     exp2b_horizons()         % full mode, ~3-4 h GPU
%     exp2b_horizons(true)     % quick mode, ~10 min
%
%   Output:
%     results/tables/exp2b_horizons_metrics.csv  (long-form: model x horizon)
%     results/tables/exp2b_horizons_results.mat  (full yhat / metrics structs)
%     results/figures/fig_horizon_mape_aggregate_multimodel.pdf

    if nargin < 1; quick = false; end

    here    = fileparts(mfilename('fullpath'));
    outDir  = fullfile(here, '..', 'results', 'tables');
    figDir  = fullfile(here, '..', 'results', 'figures');
    if ~isfolder(outDir); mkdir(outDir); end
    if ~isfolder(figDir); mkdir(figDir); end

    %% 1. Configure horizons + context lengths --------------------------------
    % Rule: L >= max(96, 2*H) so that the context window encompasses the
    % forecast horizon + at least one daily seasonality.  For H=96 (24h),
    % we use L=192 (48h) -- a compromise between covering weekly seasonality
    % and keeping attention O(L^2) manageable on consumer GPUs.
    horizons.H = [1, 4, 16, 96];        % steps
    horizons.label = {'15-min', '1-h', '4-h', '24-h'};
    horizons.L = [96, 96, 96, 192];     % context length per horizon
    horizons.stride = [4, 4, 4, 16];    % to keep sample count tractable

    nH = numel(horizons.H);

    %% 2. Load UMass + aggregate ---------------------------------------------
    optsLd.maxApts       = ifThen(quick, 20, Inf);
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topo  = load_topology('case33bw');

    [busLoad, ~, ~, ~] = preprocess_aggregate( ...
        umass.load_kW, umass.apartment_id, topo);
    P_total = sum(busLoad, 2);          % [T x 1]
    fprintf('  system-level series: T=%d, mean=%.1f kW, peak=%.1f kW\n', ...
            numel(P_total), mean(P_total), max(P_total));

    exo = build_exogenous_from_umass(umass);
    if quick
        n3 = min(numel(exo.timestamp), 96 * 60);   % 60 days
        exo = trim_exo(exo, n3);
        P_total = P_total(1:n3);
    end

    %% 3. Run each horizon ---------------------------------------------------
    % Rolling-origin protocol: build_features with stride = H matches the
    % standard "predict next H at each test step" convention used in
    % Horizon-Europe day-ahead benchmarks.
    rows = {};
    allResults = struct();
    for hi = 1:nH
        H = horizons.H(hi);
        L = horizons.L(hi);
        s = horizons.stride(hi);
        label = horizons.label{hi};

        fprintf('\n========================================\n');
        fprintf('  Horizon %s (H=%d, L=%d, stride=%d)\n', label, H, L, s);
        fprintf('========================================\n');

        feats = build_features(P_total, exo, 'L', L, 'H', H, 'stride', s);
        fprintf('  feats: L=%d, H=%d, S=%d\n', L, H, feats.dims.S);

        % Train 4 core models (skip LSTM -- unstable at N=1, already shown)
        fprintf('  -- ARIMA --\n');
        [~, yh_arima] = train_arima(feats);

        fprintf('  -- CNN-LSTM --\n');
        [~, ~, yh_cnnlstm] = train_transformer(feats, ...
            'epochs', ifThen(quick,3,100), 'K_spatial', 0);

        fprintf('  -- Vanilla Transformer --\n');
        [~, ~, yh_tx] = train_transformer(feats, ...
            'epochs', ifThen(quick,3,100));

        fprintf('  -- TA-Transformer --\n');
        A = ones(1, 1);     % degenerate at N=1
        [~, ~, predFn] = train_ta_transformer(feats, A, ...
            'epochs', ifThen(quick,3,100), 'beta', 0.0);
        yh_ta = predFn(feats.X(:,:,:,feats.idx.test));

        % Evaluate
        yt = feats.Y(:, :, feats.idx.test);
        modelYh = {'ARIMA', yh_arima; 'CNN_LSTM', yh_cnnlstm; ...
                   'Transformer', yh_tx; 'TA_Transformer', yh_ta};
        for k = 1:size(modelYh, 1)
            name = modelYh{k, 1};
            yh   = modelYh{k, 2};
            m    = evaluate_metrics(yh, yt, feats);
            rows(end+1, :) = {name, H, label, ...
                m.MAPE, m.WAPE, m.RMSE, m.R2, m.PH_MAPE}; %#ok<AGROW>
            allResults.(sprintf('H%d', H)).(name) = ...
                struct('yhat', yh, 'metrics', m);
        end
        allResults.(sprintf('H%d', H)).feats = feats;
    end

    %% 4. Long-form CSV ------------------------------------------------------
    T = cell2table(rows, 'VariableNames', ...
        {'Model','H_steps','Horizon_label', ...
         'MAPE_percent','WAPE_percent','RMSE','R2','PH_MAPE_percent'});
    csvOut = fullfile(outDir, 'exp2b_horizons_metrics.csv');
    writetable(T, csvOut);
    save(fullfile(outDir, 'exp2b_horizons_results.mat'), 'allResults', '-v7.3');
    fprintf('\nWrote %s\n', csvOut);

    %% 5. Horizon-vs-MAPE figure --------------------------------------------
    try
        plot_horizons_figure(T, fullfile(figDir, 'fig_horizon_mape_aggregate_multimodel.pdf'));
    catch ME
        fprintf(2, '[plot_horizons] FAILED: %s\n', ME.message);
    end

    %% 6. Pretty-print + Lewis quality bands --------------------------------
    fprintf('\n=========================================================\n');
    fprintf('  AGGREGATE MULTI-HORIZON SUMMARY (paper Table 4)\n');
    fprintf('=========================================================\n');
    fprintf('  Lewis band: <10%% [+++ highly accurate], 10-20%% [++ good],\n');
    fprintf('              20-50%% [+ reasonable], >50%% [-- inaccurate]\n\n');
    fprintf('  %-16s | %-8s | %-9s | %-9s\n', 'Model', 'Horizon', 'MAPE(%)', 'WAPE(%)');
    fprintf('  ---------------- | -------- | --------- | ---------\n');
    for r = 1:height(T)
        band = lewis_band(T.MAPE_percent(r));
        fprintf('  %-16s | %-8s | %7.2f %s | %7.2f\n', ...
                T.Model{r}, T.Horizon_label{r}, T.MAPE_percent(r), band, ...
                T.WAPE_percent(r));
    end

    fprintf('\n  exp2b_horizons done.\n');
end

% =========================================================================
function exo = build_exogenous_from_umass(umass)
    tz = umass.timestamp.TimeZone;
    if isempty(tz); tz = 'Asia/Shanghai'; end
    grid_ts = datetime(umass.timestamp, 'TimeZone', tz);
    T = numel(grid_ts);
    wx = umass.weather;
    if isempty(wx) || height(wx) == 0
        exo.timestamp  = grid_ts;
        exo.temp_C     = zeros(T, 1);
        exo.humidity   = zeros(T, 1);
        exo.solar_W    = zeros(T, 1);
    else
        wxTs = datetime(wx.timestamp, 'TimeZone', tz);
        TTwx = timetable(wxTs, wx.temperature_C, wx.humidity, wx.cloudCover, ...
                         'VariableNames', {'temperature_C','humidity','cloudCover'});
        TTwx = retime(TTwx, grid_ts, 'linear');
        exo.timestamp = grid_ts;
        exo.temp_C    = fillmissing(TTwx.temperature_C, 'nearest');
        exo.humidity  = fillmissing(TTwx.humidity,      'nearest');
        cc            = fillmissing(TTwx.cloudCover,    'nearest');
        exo.solar_W   = (1 - cc) * 800;
    end
    exo.ebike_kW   = zeros(T, 1);
    exo.is_weekend = ismember(weekday(grid_ts), [1 7]);
    exo.is_holiday = false(T, 1);
    exo.is_sf      = false(T, 1);
end

function exo = trim_exo(exo, n)
    f = fieldnames(exo);
    for i = 1:numel(f)
        v = exo.(f{i});
        if size(v,1) >= n; exo.(f{i}) = v(1:n, :); end
    end
end

function band = lewis_band(mape)
    if mape < 10;       band = '[+++]';
    elseif mape < 20;   band = '[++ ]';
    elseif mape < 50;   band = '[+  ]';
    else;               band = '[-- ]';
    end
end

function plot_horizons_figure(T, outPath)
    models = unique(T.Model, 'stable');
    horizons = unique(T.H_steps, 'stable');
    nM = numel(models);

    figure('Color', 'w', 'Position', [100 100 720 380]);
    colors = lines(nM);
    hold on;
    for m = 1:nM
        sub = T(strcmp(T.Model, models{m}), :);
        plot(sub.H_steps, sub.MAPE_percent, '-o', ...
             'Color', colors(m, :), 'LineWidth', 1.5, ...
             'MarkerFaceColor', colors(m, :), 'DisplayName', models{m});
    end
    set(gca, 'XScale', 'log', 'XTick', horizons, ...
             'XTickLabel', arrayfun(@(h) sprintf('%dx15m', h), ...
                                    horizons, 'UniformOutput', false));
    grid on;
    xlabel('Forecast horizon H (15-min steps)');
    ylabel('MAPE (%)');
    title('Aggregate forecast accuracy vs. horizon (UMass Smart*, N=1)');
    legend('Location', 'northwest', 'NumColumns', 2, 'Box', 'on', ...
           'EdgeColor', [0.7 0.7 0.7], 'FontSize', 9);
    exportgraphics(gcf, outPath, 'ContentType', 'vector');
end

function y = ifThen(c, a, b)
    if c; y = a; else; y = b; end
end
