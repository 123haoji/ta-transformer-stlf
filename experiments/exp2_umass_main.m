function exp2_umass_main(quick)
%EXP2_UMASS_MAIN  Multi-bus main result: UMass apartments -> case33bw.
%
%   This is the empirical headline of the paper.  Loads 114 UMass 2016
%   apartments, applies the deterministic UMass-to-33-bus aggregation
%   protocol, trains all baselines + TA-Transformer, and reports the
%   table that appears in Section 6.2 of the paper.

    if nargin < 1; quick = false; end

    here    = fileparts(mfilename('fullpath'));
    outDir  = fullfile(here, '..', 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    % 1. Load UMass apartments + topology
    optsLd.maxApts       = ifThen(quick, 20, Inf);
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topo  = load_topology('case33bw');

    % 2. Aggregate to 33 buses
    [busLoad, ~, ~, aggMeta] = preprocess_aggregate( ...
        umass.load_kW, umass.apartment_id, topo);
    fprintf('  bus load matrix: [%d x %d]\n', size(busLoad,1), size(busLoad,2));

    % 3. Exogenous: align UMass weather to 15-min grid
    exo = build_exogenous_from_umass(umass);

    if quick
        n3 = min(numel(exo.timestamp), 96*30);   % 30 days
        exo = trim_exo(exo, n3);
        busLoad = busLoad(1:n3, :);
    end

    % 4. Feature tensors
    feats = build_features(busLoad, exo, 'L', 96, 'H', 4, 'stride', 4);
    fprintf('  feats: L=%d, H=%d, N=%d, S=%d\n', ...
        feats.dims.L, feats.dims.H, feats.dims.N, feats.dims.S);

    % 5. Adjacency
    A = topo.adj_weighted;

    % 6. Train models
    %    Tier1-2 fix: bumped full-mode epochs to 80-100 with patience
    %    so transformer-class models actually converge.  ARIMA is per-bus
    %    closed-form so epochs N/A.
    fprintf('  -- ARIMA --\n');
    [~, yh_arima] = train_arima(feats);

    fprintf('  -- LSTM --\n');
    [~, ~, yh_lstm] = train_lstm(feats, 'epochs', ifThen(quick,3,80));

    fprintf('  -- CNN-LSTM (vanilla Transformer placeholder) --\n');
    [~, ~, yh_cnnlstm] = train_transformer(feats, 'epochs', ifThen(quick,3,100), ...
                                            'K_spatial', 0);  % no spatial

    fprintf('  -- Vanilla Transformer --\n');
    [~, ~, yh_tx] = train_transformer(feats, 'epochs', ifThen(quick,3,100));

    fprintf('  -- TA-Transformer (proposed) --\n');
    [taModel, ~, predFn] = train_ta_transformer(feats, A, ...
        'epochs', ifThen(quick,3,100), 'beta', 1.0);
    yh_ta = predFn(feats.X(:,:,:,feats.idx.test));

    % Ablation (c): Pure GAT (K_temporal=0, K_spatial=2).  Removes the
    % temporal Transformer backbone, retaining only the topology-aware
    % spatial GAT.  This completes the paper's three-way ablation
    % alongside (a) w/o e-bike and (b) w/o GAT (= Vanilla Transformer).
    % Refer to init_params() in train_ta_transformer.m -- it handles Kt=0
    % by allocating an empty struct array p.t, so the temporal loop in
    % forwardPass() naturally short-circuits.
    fprintf('  -- Pure GAT (ablation c: K_temporal=0) --\n');
    [~, ~, predFn_pgat] = train_ta_transformer(feats, A, ...
        'epochs', ifThen(quick,3,100), 'beta', 1.0, ...
        'K_spatial', 2, 'K_temporal', 0);
    yh_pgat = predFn_pgat(feats.X(:,:,:,feats.idx.test));

    % 7. Evaluate
    yt = feats.Y(:, :, feats.idx.test);
    results.feats = feats;
    results.ARIMA           = pack(yh_arima,   yt, feats);
    results.LSTM            = pack(yh_lstm,    yt, feats);
    results.CNN_LSTM        = pack(yh_cnnlstm, yt, feats);
    results.Transformer     = pack(yh_tx,      yt, feats);
    results.TA_Transformer  = pack(yh_ta,      yt, feats);
    results.PureGAT         = pack(yh_pgat,    yt, feats);

    % save model state so exp4 can do TRUE zero-shot (no retraining)
    stage2_model.params  = taModel;
    stage2_model.A_norm  = double(A) + eye(size(A,1));
    stage2_model.A_norm  = stage2_model.A_norm ./ ...
                            sqrt(sum(stage2_model.A_norm, 2)) ./ ...
                            sqrt(sum(stage2_model.A_norm, 2))';
    stage2_model.beta    = 1.0;
    stage2_model.K_spatial  = 2;
    stage2_model.K_temporal = 2;
    stage2_model.norm    = feats.norm;
    stage2_model.F_total = feats.dims.F_total;
    stage2_model.L       = feats.dims.L;
    stage2_model.H       = feats.dims.H;
    save(fullfile(outDir,'exp2_model.mat'), 'stage2_model', '-v7.3');

    save(fullfile(outDir,'exp2_results.mat'), 'results', 'aggMeta', '-v7.3');
    plot_results(results, fullfile(here,'..','results','figures'));
    write_metric_table(results, fullfile(outDir,'exp2_metrics.csv'));
    fprintf('  Stage 2 done.\n');
end

% ---------------------------------------------------------------------------
function exo = build_exogenous_from_umass(umass)
    % Force-align all timestamps to a common timezone before retiming.
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
    % ebike_kW must be either [T, 1] (broadcast) or [T, N_buses].  Using [T, 1]
    % of zeros marks "no e-bike data for this dataset"; build_features will
    % broadcast it across all buses.
    exo.ebike_kW   = zeros(T, 1);
    exo.is_weekend = ismember(weekday(grid_ts), [1 7]);
    exo.is_holiday = false(T, 1);
    exo.is_sf      = false(T, 1);
end

function s = pack(yhat, ytrue, feats)
    s.yhat    = yhat;
    s.metrics = evaluate_metrics(yhat, ytrue, feats);
end

function exo = trim_exo(exo, n)
    f = fieldnames(exo);
    for i = 1:numel(f)
        v = exo.(f{i});
        if size(v,1) >= n
            exo.(f{i}) = v(1:n, :);
        end
    end
end

function write_metric_table(results, csvPath)
    names = setdiff(fieldnames(results), {'feats'}, 'stable');
    rows = {};
    for k = 1:numel(names)
        m = results.(names{k}).metrics;
        % Add WAPE (volume-weighted absolute percent error) as a column
        % alongside MAPE.  WAPE is the recommended primary metric on
        % clustered/skewed loads where small-bus errors inflate MAPE.
        wape_val = NaN;
        if isfield(m, 'WAPE'); wape_val = m.WAPE; end
        rows(end+1, :) = {names{k}, m.MAPE, wape_val, m.RMSE, m.R2, m.PH_MAPE}; %#ok<AGROW>
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Model','MAPE_percent','WAPE_percent','RMSE','R2','PH_MAPE_percent'});
    writetable(T, csvPath);
end

function y = ifThen(cond, a, b)
    if cond; y = a; else; y = b; end
end
