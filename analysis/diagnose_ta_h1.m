function diagnose_ta_h1()
%DIAGNOSE_TA_H1  Anomaly investigation: TA-Transformer at H=1 (15-min)
%   underperforms its own H=4 (1-h) result on the system-aggregate task.
%
%   Hypothesis under test:
%     The "TA-T" called in exp2b_horizons.m is invoked with beta=0 (so the
%     topology bias is structurally absent), making it architecturally
%     identical to the Vanilla Transformer except for the default learning
%     rate (5e-4 vs 1e-3).  The H=1 anomaly is therefore predicted to be a
%     pure under-fitting artifact that vanishes once lr is matched.
%
%   Design (4 runs, ~12 min total):
%     A: TA-T H=1, lr=5e-4, seed=1   -- reproducibility of default
%     B: TA-T H=1, lr=5e-4, seed=2   -- reproducibility of default
%     C: TA-T H=1, lr=1e-3, seed=42  -- lr fix
%     D: TA-T H=1, lr=1e-3, seed=1   -- lr fix reproducibility
%
%   Reference numbers (existing single-seed exp2b_horizons run, seed=42):
%     TA-T H=1: MAPE 13.11, WAPE 11.98  (anomalous)
%     Vanilla H=1: MAPE 11.61, WAPE 10.68
%     TA-T H=4: MAPE 11.47, WAPE 10.85  (normal)
%
%   Output:
%     results/tables_paper/diagnose_ta_h1.csv
%     console summary

    here   = fileparts(mfilename('fullpath'));
    outDir = fullfile(here, 'results', 'tables_paper');
    if ~isfolder(outDir); mkdir(outDir); end

    % --- Load UMass + aggregate (identical to exp2b_horizons.m) ----------
    addpath(fullfile(here, 'src'));
    addpath(fullfile(here, 'experiments'));

    optsLd.maxApts       = Inf;
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topo  = load_topology('case33bw');

    [busLoad, ~, ~, ~] = preprocess_aggregate( ...
        umass.load_kW, umass.apartment_id, topo);
    P_total = sum(busLoad, 2);
    fprintf('  system-level series: T=%d, mean=%.1f kW, peak=%.1f kW\n', ...
            numel(P_total), mean(P_total), max(P_total));

    exo = build_exo(umass);

    feats = build_features(P_total, exo, 'L', 96, 'H', 1, 'stride', 4);
    fprintf('  feats: L=%d, H=%d, S=%d\n', 96, 1, feats.dims.S);

    A = ones(1, 1);
    yt = feats.Y(:, :, feats.idx.test);

    % --- Run plan ---------------------------------------------------------
    plan = {
    %   tag      lr        seed
        'A',    5e-4,      1;
        'B',    5e-4,      2;
        'C',    1e-3,     42;
        'D',    1e-3,      1;
    };

    rows = {};
    for i = 1:size(plan, 1)
        tag  = plan{i, 1};
        lr_v = plan{i, 2};
        sd   = plan{i, 3};
        fprintf('\n========================================\n');
        fprintf('  [%s] TA-T H=1, lr=%.1e, seed=%d\n', tag, lr_v, sd);
        fprintf('========================================\n');
        t0 = tic;
        [~, ~, predFn] = train_ta_transformer(feats, A, ...
            'epochs', 100, ...
            'beta',   0.0, ...        % match exp2b_horizons.m semantics
            'lr',     lr_v, ...
            'seed',   sd, ...
            'verbose', false);
        yh = predFn(feats.X(:,:,:,feats.idx.test));
        m  = evaluate_metrics(yh, yt, feats);
        elapsed = toc(t0);
        rows(end+1, :) = {tag, lr_v, sd, m.MAPE, m.WAPE, m.RMSE, m.R2, ...
                          m.PH_MAPE, elapsed}; %#ok<AGROW>
        fprintf('  [%s]  MAPE=%.2f  WAPE=%.2f  RMSE=%.1f  R2=%.3f  (%.0fs)\n', ...
                tag, m.MAPE, m.WAPE, m.RMSE, m.R2, elapsed);
    end

    T = cell2table(rows, 'VariableNames', ...
        {'Tag','LR','Seed','MAPE_percent','WAPE_percent', ...
         'RMSE','R2','PH_MAPE_percent','Sec'});
    csvOut = fullfile(outDir, 'diagnose_ta_h1.csv');
    writetable(T, csvOut);

    fprintf('\n=========================================================\n');
    fprintf('  TA-T H=1 ANOMALY DIAGNOSIS SUMMARY\n');
    fprintf('=========================================================\n');
    fprintf('  Tag | LR     | Seed |  MAPE  |  WAPE  |   R2   | PH-MAPE\n');
    fprintf('  --- | ------ | ---- | ------ | ------ | ------ | -------\n');
    for r = 1:height(T)
        fprintf('   %s  | %.0e | %4d | %6.2f | %6.2f | %6.3f | %7.2f\n', ...
            T.Tag{r}, T.LR(r), T.Seed(r), ...
            T.MAPE_percent(r), T.WAPE_percent(r), T.R2(r), T.PH_MAPE_percent(r));
    end
    fprintf('\n  Reference (existing seed=42, lr=5e-4): MAPE=13.11  WAPE=11.98\n');
    fprintf('  Reference Vanilla (seed=42, lr=1e-3):    MAPE=11.61  WAPE=10.68\n');
    fprintf('\n  Decision rule:\n');
    fprintf('    If C/D MAPE drops to ~11.6%%  -> lr is the SOLE cause (apply fix)\n');
    fprintf('    If C/D MAPE stays ~13%%       -> structural (mean-pool dilution)\n');
    fprintf('    If A/B MAPE varies by >1pp   -> seed noise, not systematic\n\n');

    fprintf('  Wrote %s\n', csvOut);
    fprintf('  diagnose_ta_h1 done.\n');
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
