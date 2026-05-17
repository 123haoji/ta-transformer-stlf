function reeval_exp2_wape()
%REEVAL_EXP2_WAPE  Re-score saved Stage 2 results using WAPE (no retraining).
%
%   Loads results/tables/exp2_results.mat (contains feats + each model's
%   yhat tensor), re-runs evaluate_metrics.m -- which now reports WAPE
%   alongside MAPE -- and prints/writes a new CSV.
%
%   This is the FASTEST way to confirm the diagnosis that MAPE 44% is
%   metric-distortion (small-cluster inflation) and not a model failure.

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'), fullfile(here, 'experiments'));

    matFile = fullfile(here, 'results', 'tables', 'exp2_results.mat');
    if ~isfile(matFile)
        error('Stage 2 results not found at %s. Run main(''Stage'',2) first.', matFile);
    end
    S = load(matFile);
    R = S.results;
    feats = R.feats;
    modelNames = setdiff(fieldnames(R), {'feats'}, 'stable');

    yt = feats.Y(:, :, feats.idx.test);

    fprintf('\n=========================================================\n');
    fprintf('  Stage 2 RE-EVALUATION with WAPE\n');
    fprintf('=========================================================\n');

    rows = {};
    for k = 1:numel(modelNames)
        name = modelNames{k};
        yhat = R.(name).yhat;
        m    = evaluate_metrics(yhat, yt, feats);
        fprintf('  %-16s  MAPE=%6.2f%%  WAPE=%6.2f%%  RMSE=%7.2f  R2=%.3f  PH_MAPE=%6.2f%%\n', ...
                name, m.MAPE, m.WAPE, m.RMSE, m.R2, m.PH_MAPE);
        rows(end+1, :) = {name, m.MAPE, m.WAPE, m.RMSE, m.R2, m.PH_MAPE}; %#ok<AGROW>
    end
    T = cell2table(rows, 'VariableNames', ...
        {'Model','MAPE_percent','WAPE_percent','RMSE','R2','PH_MAPE_percent'});
    out = fullfile(here, 'results', 'tables', 'exp2_metrics_wape.csv');
    writetable(T, out);
    fprintf('\nWrote %s\n', out);

    %% -------- per-bus WAPE distribution -----------------------------------
    fprintf('\n--- TA-Transformer per-bus WAPE distribution (sorted) ---\n');
    yhat = R.TA_Transformer.yhat;
    m    = evaluate_metrics(yhat, yt, feats);
    if isfield(m, 'perBus') && isfield(m.perBus, 'WAPE')
        wpb = sort(m.perBus.WAPE);
        fprintf('  min  = %6.2f%%\n', wpb(1));
        fprintf('  p25  = %6.2f%%\n', wpb(max(1, round(0.25*numel(wpb)))));
        fprintf('  median %6.2f%%\n', wpb(max(1, round(0.50*numel(wpb)))));
        fprintf('  p75  = %6.2f%%\n', wpb(max(1, round(0.75*numel(wpb)))));
        fprintf('  max  = %6.2f%%\n', wpb(end));
        fprintf('\n  Heads (low-WAPE buses):  %s\n', mat2str(round(wpb(1:min(5,end))*10)/10));
        fprintf('  Tails (high-WAPE buses): %s\n', mat2str(round(wpb(max(end-4,1):end)*10)/10));
    end

    fprintf('\n=========================================================\n');
    fprintf('  INTERPRETATION:\n');
    fprintf('  - MAPE > WAPE  -> small-bus inflation is the dominant issue\n');
    fprintf('  - Headline metric to report: WAPE\n');
    fprintf('=========================================================\n\n');
end
