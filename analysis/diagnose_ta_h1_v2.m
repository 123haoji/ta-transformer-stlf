function diagnose_ta_h1_v2()
%DIAGNOSE_TA_H1_V2  Verify the lr=1e-3 / seed=1 outlier result (MAPE=10.03)
%   by running one more seed at the same configuration.
%
%   If the verify run gives MAPE near 10%, lr=1e-3 is genuinely better and
%   we may consider raising the default; if it lands near 12-13%, the 10.03
%   was a lucky-seed outlier and we keep the current convention.

    here   = fileparts(mfilename('fullpath'));
    outDir = fullfile(here, 'results', 'tables_paper');
    if ~isfolder(outDir); mkdir(outDir); end

    addpath(fullfile(here, 'src'));

    optsLd.maxApts = Inf;
    optsLd.missingThresh = 0.10;
    optsLd.resampleMinutes = 15;
    umass = load_umass_apartment(2016, [], optsLd);
    topo  = load_topology('case33bw');

    [busLoad, ~, ~, ~] = preprocess_aggregate( ...
        umass.load_kW, umass.apartment_id, topo);
    P_total = sum(busLoad, 2);

    exo = build_exo(umass);
    feats = build_features(P_total, exo, 'L', 96, 'H', 1, 'stride', 4);

    A  = ones(1, 1);
    yt = feats.Y(:, :, feats.idx.test);

    fprintf('\n========================================\n');
    fprintf('  [D-verify] TA-T H=1, lr=1.0e-03, seed=2\n');
    fprintf('========================================\n');
    t0 = tic;
    [~, ~, predFn] = train_ta_transformer(feats, A, ...
        'epochs', 100, 'beta', 0.0, 'lr', 1e-3, 'seed', 2, 'verbose', false);
    yh = predFn(feats.X(:,:,:,feats.idx.test));
    m  = evaluate_metrics(yh, yt, feats);

    fprintf('\n=========================================================\n');
    fprintf('  D-VERIFY RESULT (lr=1e-3, seed=2)\n');
    fprintf('=========================================================\n');
    fprintf('  MAPE=%.2f  WAPE=%.2f  RMSE=%.1f  R2=%.3f  PH-MAPE=%.2f  (%.0fs)\n', ...
        m.MAPE, m.WAPE, m.RMSE, m.R2, m.PH_MAPE, toc(t0));

    fprintf('\n  Previous lr=1e-3 results: seed=42 -> 13.07,  seed=1 -> 10.03\n');
    fprintf('  Decision:\n');
    fprintf('    If MAPE ~10%%  -> lr=1e-3 genuinely better (default upgrade candidate)\n');
    fprintf('    If MAPE >12%%  -> 10.03 was lucky-seed outlier, keep current default\n\n');

    T = table({'D-verify'}, 1e-3, 2, m.MAPE, m.WAPE, m.RMSE, m.R2, m.PH_MAPE, ...
              'VariableNames', {'Tag','LR','Seed','MAPE_percent','WAPE_percent', ...
                                'RMSE','R2','PH_MAPE_percent'});
    writetable(T, fullfile(outDir, 'diagnose_ta_h1_v2.csv'));
end

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
