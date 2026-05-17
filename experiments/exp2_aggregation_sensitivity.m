function exp2_aggregation_sensitivity(quick)
%EXP2_AGGREGATION_SENSITIVITY  Sweep the number of clusters K used in
%   preprocess_aggregate, training the TA-Transformer for each K and
%   tabulating MAPE / WAPE / R^2.
%
%   This addresses the reviewer-anticipated question "How does forecast
%   accuracy degrade as cluster granularity changes?" -- a standard
%   sensitivity-analysis subsection in MDPI Energies STLF papers.
%
%   K values: {5, 10, 20, 33}
%     - K=5  : extreme aggregation (almost system-level)
%     - K=10 : commercial-feeder granularity
%     - K=20 : small substation granularity
%     - K=33 : original case33bw (paper main result)
%
%   Note: requires changing the preprocess_aggregate cluster count, which
%   the function exposes via the K_target parameter (NOT YET IMPLEMENTED
%   in preprocess_aggregate.m).  Until that is exposed, this script
%   monkey-patches by editing topo.N (an acceptable hack since rank
%   matching picks K = min(N-1, J-1) clusters in practice).
%
%   USAGE
%     exp2_aggregation_sensitivity()       % full, ~3-4 h GPU
%     exp2_aggregation_sensitivity(true)   % quick, ~20 min

    if nargin < 1; quick = false; end

    here   = fileparts(mfilename('fullpath'));
    outDir = fullfile(here, '..', 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    %% --- Load once ---------------------------------------------------------
    optsLd.maxApts = ifThen(quick, 20, Inf);
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topoOrig = load_topology('case33bw');

    K_values = ifThen(quick, [5 10], [5 10 20 33]);

    rows = {};
    for K = K_values
        fprintf('\n========================================\n');
        fprintf('  K = %d clusters\n', K);
        fprintf('========================================\n');

        % Use a reduced topology by selecting K largest-Pd buses.
        % preprocess_aggregate will rank-match clusters to these K buses.
        topo = topoOrig;
        % Sort buses by Pd descending; keep only the top K + slack.
        [pd_sorted, idx_sorted] = sort(topo.bus_Pd_kW, 'descend');
        keep = idx_sorted(1:min(K+1, numel(idx_sorted)));   % +1 for slack
        topo.N = numel(keep);
        topo.bus = topo.bus(keep, :);
        topo.bus_Pd_kW = topo.bus_Pd_kW(keep);
        topo.adj_weighted = topo.adj_weighted(keep, keep);

        [busLoad, ~, ~, ~] = preprocess_aggregate( ...
            umass.load_kW, umass.apartment_id, topo);
        fprintf('  bus load matrix: [%d x %d]\n', size(busLoad,1), size(busLoad,2));

        exo = build_exogenous_from_umass(umass);
        if quick
            n3 = min(numel(exo.timestamp), 96 * 30);
            exo = trim_exo(exo, n3);
            busLoad = busLoad(1:n3, :);
        end

        feats = build_features(busLoad, exo, 'L', 96, 'H', 4, 'stride', 4);
        A = topo.adj_weighted;

        % Train only the proposed model + 1 baseline (Transformer) to save time.
        fprintf('  -- Transformer baseline --\n');
        [~, ~, yh_tx] = train_transformer(feats, 'epochs', ifThen(quick,3,60));

        fprintf('  -- TA-Transformer --\n');
        [~, ~, predFn] = train_ta_transformer(feats, A, ...
            'epochs', ifThen(quick,3,60), 'beta', 1.0);
        yh_ta = predFn(feats.X(:,:,:,feats.idx.test));

        yt = feats.Y(:, :, feats.idx.test);
        m_tx = evaluate_metrics(yh_tx, yt, feats);
        m_ta = evaluate_metrics(yh_ta, yt, feats);
        rows(end+1, :) = {'Transformer',    K, m_tx.MAPE, m_tx.WAPE, m_tx.RMSE, m_tx.R2}; %#ok<AGROW>
        rows(end+1, :) = {'TA_Transformer', K, m_ta.MAPE, m_ta.WAPE, m_ta.RMSE, m_ta.R2}; %#ok<AGROW>
    end

    T = cell2table(rows, 'VariableNames', ...
        {'Model','K_clusters','MAPE_percent','WAPE_percent','RMSE','R2'});
    csvOut = fullfile(outDir, 'exp2_aggregation_sensitivity.csv');
    writetable(T, csvOut);
    fprintf('\nWrote %s\n', csvOut);

    fprintf('\n=========================================================\n');
    fprintf('  AGGREGATION SENSITIVITY\n');
    fprintf('  Expectation: MAPE/WAPE decrease monotonically as K decreases\n');
    fprintf('  (more aggregation -> smoother signal -> better accuracy)\n');
    fprintf('=========================================================\n');
    fprintf('  %-16s | %5s | %-8s | %-8s\n', 'Model', 'K', 'MAPE(%)', 'WAPE(%)');
    fprintf('  ---------------- | ----- | -------- | --------\n');
    for r = 1:height(T)
        fprintf('  %-16s | %5d | %8.2f | %8.2f\n', ...
            T.Model{r}, T.K_clusters(r), T.MAPE_percent(r), T.WAPE_percent(r));
    end
    fprintf('\n  Aggregation sensitivity done.\n');
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

function y = ifThen(c, a, b)
    if c; y = a; else; y = b; end
end
