function exp2_k_partial()
%EXP2_K_PARTIAL  Run K = {5, 10, 20} cases of the aggregation sensitivity
%   sweep.  This complements exp2_k33_only.m which already wrote the K=33
%   row to results/tables/exp2_aggregation_sensitivity.csv.
%
%   We run each K in its own fully-isolated iteration, explicitly clearing
%   intermediate state between iterations to avoid the heap-accumulation
%   OOM that crashed the original exp2_aggregation_sensitivity.m.  K=20 is
%   the largest case here (33 was the largest before) so memory pressure
%   is substantially lower.
%
%   Output: the existing exp2_aggregation_sensitivity.csv is read,
%   merged with the new K = {5, 10, 20} rows, deduplicated on (Model, K),
%   and written back.

    here   = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));
    addpath(fullfile(here, 'experiments'));
    outDir = fullfile(here, 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    %% Load once -----------------------------------------------------------
    optsLd.maxApts = Inf;
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topoOrig = load_topology('case33bw');

    K_values = [5, 10, 20];
    rows = {};

    for K = K_values
        fprintf('\n========================================\n');
        fprintf('  K = %d clusters (partial-sweep run)\n', K);
        fprintf('========================================\n');

        % Reduce topology to top-(K+1) buses by Pd, exactly mirroring the
        % protocol in exp2_aggregation_sensitivity.m.
        topo = topoOrig;
        [~, idx_sorted] = sort(topo.bus_Pd_kW, 'descend');
        keep = idx_sorted(1:min(K+1, numel(idx_sorted)));
        topo.N = numel(keep);
        topo.bus = topo.bus(keep, :);
        topo.bus_Pd_kW = topo.bus_Pd_kW(keep);
        topo.adj_weighted = topo.adj_weighted(keep, keep);

        [busLoad, ~, ~, ~] = preprocess_aggregate( ...
            umass.load_kW, umass.apartment_id, topo);
        fprintf('  bus load matrix: [%d x %d]\n', size(busLoad,1), size(busLoad,2));

        exo = build_exo(umass);
        feats = build_features(busLoad, exo, 'L', 96, 'H', 4, 'stride', 4);
        A = topo.adj_weighted;
        yt = feats.Y(:, :, feats.idx.test);

        fprintf('  -- Transformer baseline --\n');
        [~, ~, yh_tx] = train_transformer(feats, 'epochs', 60);
        m_tx = evaluate_metrics(yh_tx, yt, feats);
        rows(end+1, :) = {'Transformer', K, m_tx.MAPE, m_tx.WAPE, m_tx.RMSE, m_tx.R2}; %#ok<AGROW>
        fprintf('  Transformer K=%d:    MAPE %.2f%%  WAPE %.2f%%  R^2 %.3f\n', ...
                K, m_tx.MAPE, m_tx.WAPE, m_tx.R2);

        % Free Transformer-baseline state before training TA-T
        clear yh_tx
        if canUseGPU(); try; reset(gpuDevice); catch; end; end %#ok<TRYNC>

        fprintf('  -- TA-Transformer --\n');
        [~, ~, predFn] = train_ta_transformer(feats, A, 'epochs', 60, 'beta', 1.0);
        yh_ta = predFn(feats.X(:,:,:,feats.idx.test));
        m_ta = evaluate_metrics(yh_ta, yt, feats);
        rows(end+1, :) = {'TA_Transformer', K, m_ta.MAPE, m_ta.WAPE, m_ta.RMSE, m_ta.R2}; %#ok<AGROW>
        fprintf('  TA-Transformer K=%d: MAPE %.2f%%  WAPE %.2f%%  R^2 %.3f\n', ...
                K, m_ta.MAPE, m_ta.WAPE, m_ta.R2);

        % --- Inter-iteration memory cleanup ----------------------------------
        % Workspace clears + GPU device reset prevent the heap accumulation
        % that crashed the original full-sweep script at K=33.
        clear feats yh_ta predFn busLoad exo A yt m_tx m_ta
        if canUseGPU(); try; reset(gpuDevice); catch; end; end %#ok<TRYNC>
    end

    %% Append rows to existing CSV -----------------------------------------
    newRows = cell2table(rows, 'VariableNames', ...
        {'Model','K_clusters','MAPE_percent','WAPE_percent','RMSE','R2'});

    csvOut = fullfile(outDir, 'exp2_aggregation_sensitivity.csv');
    if isfile(csvOut)
        existing = readtable(csvOut);
        % Drop any prior K = {5, 10, 20} rows in case of re-runs
        toDrop = ismember(existing.K_clusters, K_values);
        existing(toDrop, :) = [];
        combined = [existing; newRows];
    else
        combined = newRows;
    end
    % Sort by (K_clusters, Model) for readability
    [~, ord] = sortrows([combined.K_clusters, double(strcmp(combined.Model, 'TA_Transformer'))]);
    combined = combined(ord, :);
    writetable(combined, csvOut);

    fprintf('\n=========================================================\n');
    fprintf('  K-SWEEP TABLE (paper Section 6.7)\n');
    fprintf('=========================================================\n');
    fprintf('  %-16s | %5s | %-8s | %-8s | %-6s\n', ...
            'Model', 'K', 'MAPE %', 'WAPE %', 'R^2');
    fprintf('  ---------------- | ----- | -------- | -------- | ------\n');
    for r = 1:height(combined)
        fprintf('  %-16s | %5d | %8.2f | %8.2f | %6.3f\n', ...
            combined.Model{r}, combined.K_clusters(r), ...
            combined.MAPE_percent(r), combined.WAPE_percent(r), combined.R2(r));
    end
    fprintf('\n  Wrote %s\n', csvOut);
    fprintf('  exp2_k_partial done.\n');
end

% ============================================================================
function exo = build_exo(umass)
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
