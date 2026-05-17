function holiday_sensitivity(stage)
%HOLIDAY_SENSITIVITY  Slice saved forecast results by US holiday weeks
%   to quantify model robustness on special calendar dates.
%
%   stage = 'cluster'   (default) | 'aggregate'
%
%   Holiday windows considered (US calendar, UMass dataset is Massachusetts):
%     - Thanksgiving week:  Nov 24-30 (Wed-Sun anchored)
%     - Christmas week:     Dec 22-28
%     - Independence Day:   Jul 1-7
%     - New Year:           Jan 1-7
%
%   For each holiday window + a matched non-holiday baseline week, we
%   compute per-window MAPE/WAPE.  This is the US analogue of the
%   Chinese-paper-style "Spring Festival vs. typical workday" comparison
%   originally proposed in the paper template.
%
%   No retraining required -- consumes saved test predictions only.

    if nargin < 1; stage = 'cluster'; end

    here   = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));
    tabDir = fullfile(here, 'results', 'tables');
    outDir = fullfile(here, 'results', 'tables_paper');
    if ~isfolder(outDir); mkdir(outDir); end

    switch lower(stage)
        case {'cluster','stage2'}
            matFile = fullfile(tabDir, 'exp2_results.mat');
            label   = 'cluster';
        case {'aggregate','stage2b','stage5'}
            matFile = fullfile(tabDir, 'exp2b_aggregate_results.mat');
            label   = 'aggregate';
        otherwise
            error('Unknown stage: %s', stage);
    end
    assert(isfile(matFile), 'Result file not found: %s', matFile);

    S = load(matFile);
    R = S.results;
    feats = R.feats;
    modelNames = setdiff(fieldnames(R), {'feats'}, 'stable');

    % Per-sample timestamps (use the start of each forecast window).
    ts_test = feats.ts_y(1, feats.idx.test);    % [1 x S_test]
    ts_test = datetime(ts_test, 'TimeZone', 'America/New_York');

    %% --- Define holiday windows --------------------------------------------
    win = struct();
    win.Thanksgiving = (month(ts_test) == 11 & day(ts_test) >= 24 & day(ts_test) <= 30);
    win.Christmas    = (month(ts_test) == 12 & day(ts_test) >= 22 & day(ts_test) <= 28);
    win.IndepDay     = (month(ts_test) == 7  & day(ts_test) >= 1  & day(ts_test) <= 7);
    win.NewYear      = (month(ts_test) == 1  & day(ts_test) >= 1  & day(ts_test) <= 7);
    % Non-holiday baseline: test samples that fall in none of the above
    win.Baseline     = ~(win.Thanksgiving | win.Christmas | win.IndepDay | win.NewYear);

    winNames = fieldnames(win);

    %% --- Per-window metric computation ------------------------------------
    rows = {};
    yt   = feats.Y(:, :, feats.idx.test);
    for k = 1:numel(modelNames)
        yh = R.(modelNames{k}).yhat;
        for w = 1:numel(winNames)
            mask = win.(winNames{w});
            if sum(mask) < 5
                rows(end+1, :) = {modelNames{k}, winNames{w}, sum(mask), ...
                                  NaN, NaN, NaN, NaN}; %#ok<AGROW>
                continue;
            end
            feats_sub = feats;
            feats_sub.idx.test = feats.idx.test(mask);
            yh_sub = yh(:, :, mask);
            yt_sub = yt(:, :, mask);
            m = evaluate_metrics(single(yh_sub), single(yt_sub), feats_sub);
            rows(end+1, :) = {modelNames{k}, winNames{w}, sum(mask), ...
                              m.MAPE, m.WAPE, m.RMSE, m.R2}; %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Model','Window','N_samples','MAPE_percent','WAPE_percent','RMSE','R2'});

    csvOut = fullfile(outDir, sprintf('holiday_sensitivity_%s.csv', label));
    writetable(T, csvOut);

    %% --- Pretty-print ------------------------------------------------------
    fprintf('\n=========================================================\n');
    fprintf('  HOLIDAY-WEEK SENSITIVITY (Stage %s)\n', label);
    fprintf('=========================================================\n');
    fprintf('  %-16s | %-12s | %-3s | %-8s | %-8s\n', ...
            'Model', 'Window', 'N', 'MAPE(%)', 'WAPE(%)');
    fprintf('  ---------------- | ------------ | --- | -------- | --------\n');
    for r = 1:height(T)
        fprintf('  %-16s | %-12s | %3d | %8.2f | %8.2f\n', ...
            T.Model{r}, T.Window{r}, T.N_samples(r), ...
            T.MAPE_percent(r), T.WAPE_percent(r));
    end
    fprintf('\n  Wrote %s\n', csvOut);
end
