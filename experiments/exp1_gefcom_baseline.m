function exp1_gefcom_baseline(quick)
%EXP1_GEFCOM_BASELINE  Single-zone temporal-backbone sanity check.
%
%   This experiment verifies that the temporal blocks of TA-Transformer
%   match published GEFCom2014 Track-L performance when the spatial axis
%   is degenerate (N=1).  It is NOT the main result of the paper.

    if nargin < 1; quick = false; end

    here    = fileparts(mfilename('fullpath'));
    outDir  = fullfile(here, '..', 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    % 1. Load GEFCom
    gef = load_gefcom(1:12);
    fprintf('  GEFCom rows: %d, target rows: %d\n', ...
        gef.meta.n_rows, gef.meta.n_target_rows);

    % 2. Pack as a (T x 1) load matrix; weather as exogenous
    P = gef.load_MW;
    % drop rows with NaN load (target rows; we evaluate on train/val/test in-distribution)
    keep = ~gef.is_target;
    ts   = gef.timestamp(keep);
    P    = P(keep);
    tempC = mean(gef.temp_C(keep, :), 2, 'omitnan');

    exo.timestamp  = ts;
    exo.temp_C     = tempC;
    exo.humidity   = zeros(size(ts));     % not in GEFCom
    exo.solar_W    = zeros(size(ts));
    exo.ebike_kW   = zeros(numel(ts), 1);
    exo.is_weekend = ismember(weekday(ts), [1 7]);
    exo.is_holiday = false(numel(ts), 1);
    exo.is_sf      = false(numel(ts), 1);

    if quick
        % truncate to first ~3 months for debug
        n3 = min(numel(ts), 24*30*3);
        ts = ts(1:n3); P = P(1:n3);
        exo = trim_exo(exo, n3);
    end

    feats = build_features(reshape(P,[],1), exo, ...
                            'L', 168, 'H', 24, 'stride', 24);
    fprintf('  feats: L=%d, H=%d, S=%d\n', feats.dims.L, feats.dims.H, feats.dims.S);

    % 3. Single-zone -> dummy adjacency = scalar 1
    A = 1;

    % 4. Train baselines + proposed
    fprintf('  -- ARIMA --\n');
    [~, yh_arima] = train_arima(feats, 'pRange', 0:2, 'qRange', 0:2);

    fprintf('  -- LSTM --\n');
    [~, ~, yh_lstm] = train_lstm(feats, 'epochs', 10);

    fprintf('  -- Vanilla Transformer --\n');
    [~, ~, yh_tx] = train_transformer(feats, 'epochs', 10);

    fprintf('  -- TA-Transformer (degenerate spatial) --\n');
    [~, ~, predFn] = train_ta_transformer(feats, A, 'epochs', 10);
    yh_ta = predFn(feats.X(:,:,:,feats.idx.test));

    % 5. Evaluate and save
    yt = feats.Y(:, :, feats.idx.test);
    results.feats = feats;
    results.ARIMA           = pack(yh_arima, yt, feats);
    results.LSTM            = pack(yh_lstm,  yt, feats);
    results.Transformer     = pack(yh_tx,    yt, feats);
    results.TA_Transformer  = pack(yh_ta,    yt, feats);

    save(fullfile(outDir, 'exp1_results.mat'), 'results', '-v7.3');
    plot_results(results, fullfile(here, '..', 'results', 'figures'));

    write_metric_table(results, fullfile(outDir, 'exp1_metrics.csv'));
    fprintf('  Stage 1 done. Tables under %s\n', outDir);
end

% ---------------------------------------------------------------------------
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
        rows(end+1, :) = {names{k}, m.MAPE, m.RMSE, m.R2, m.PH_MAPE}; %#ok<AGROW>
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Model','MAPE_percent','RMSE','R2','PH_MAPE_percent'});
    writetable(T, csvPath);
end
