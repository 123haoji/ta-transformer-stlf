function exp2_k33_only()
%EXP2_K33_ONLY  Run K=33 case of aggregation sensitivity in isolation.
%
%   The full exp2_aggregation_sensitivity.m crashed with an out-of-memory
%   error at K=33 due to MATLAB workspace state accumulated over the
%   K = {5, 10, 20} iterations.  This script runs ONLY K=33 (the original
%   case33bw topology) in a fresh MATLAB process so no prior state
%   pollutes the RAM/GPU heaps.
%
%   The output is appended (not overwritten) to the partial CSV produced
%   by the earlier run so the full K = {5, 10, 20, 33} table is recovered.

    here   = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));
    addpath(fullfile(here, 'experiments'));
    outDir = fullfile(here, 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    %% Load fresh -----------------------------------------------------------
    optsLd.maxApts = Inf;
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topo  = load_topology('case33bw');

    K = 33;
    fprintf('\n========================================\n');
    fprintf('  K = %d clusters (isolated run)\n', K);
    fprintf('========================================\n');

    [busLoad, ~, ~, ~] = preprocess_aggregate( ...
        umass.load_kW, umass.apartment_id, topo);
    fprintf('  bus load matrix: [%d x %d]\n', size(busLoad,1), size(busLoad,2));

    exo = build_exo(umass);

    feats = build_features(busLoad, exo, 'L', 96, 'H', 4, 'stride', 4);
    A = topo.adj_weighted;

    fprintf('  -- Transformer baseline --\n');
    [~, ~, yh_tx] = train_transformer(feats, 'epochs', 60);

    fprintf('  -- TA-Transformer --\n');
    [~, ~, predFn] = train_ta_transformer(feats, A, 'epochs', 60, 'beta', 1.0);
    yh_ta = predFn(feats.X(:,:,:,feats.idx.test));

    yt = feats.Y(:, :, feats.idx.test);
    m_tx = evaluate_metrics(yh_tx, yt, feats);
    m_ta = evaluate_metrics(yh_ta, yt, feats);

    %% Append to partial CSV ------------------------------------------------
    csvOut = fullfile(outDir, 'exp2_aggregation_sensitivity.csv');
    newRows = cell2table( ...
        {'Transformer',    K, m_tx.MAPE, m_tx.WAPE, m_tx.RMSE, m_tx.R2; ...
         'TA_Transformer', K, m_ta.MAPE, m_ta.WAPE, m_ta.RMSE, m_ta.R2}, ...
        'VariableNames', ...
        {'Model','K_clusters','MAPE_percent','WAPE_percent','RMSE','R2'});

    if isfile(csvOut)
        existing = readtable(csvOut);
        % drop any prior K=33 rows in case of re-runs, then concatenate
        existing(existing.K_clusters == K, :) = [];
        combined = [existing; newRows];
        writetable(combined, csvOut);
        fprintf('  Appended K=33 to existing CSV.\n');
    else
        writetable(newRows, csvOut);
        fprintf('  Wrote new CSV (K=33 only).\n');
    end

    fprintf('\n  K=33 results:\n');
    fprintf('    Transformer:    MAPE %.2f%%  WAPE %.2f%%  R^2 %.3f\n', ...
            m_tx.MAPE, m_tx.WAPE, m_tx.R2);
    fprintf('    TA-Transformer: MAPE %.2f%%  WAPE %.2f%%  R^2 %.3f\n', ...
            m_ta.MAPE, m_ta.WAPE, m_ta.R2);
    fprintf('  exp2_k33_only done.\n');
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
