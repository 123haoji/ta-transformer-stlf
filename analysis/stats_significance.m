function stats_significance(stage)
%STATS_SIGNIFICANCE  Compute pairwise statistical significance tests
%  on saved forecast results for paper-grade comparison.
%
%   stats_significance()        % default: Stage 2 (cluster)
%   stats_significance('cluster')
%   stats_significance('aggregate')
%
%   Methods implemented:
%     1. Lilliefors test  -- a Kolmogorov-Smirnov-based normality test
%                            (does not require pre-specified mean/variance;
%                            preferred over Shapiro-Wilk in MATLAB built-in
%                            since R2009a -- equivalent purpose here).
%     2. Wilcoxon signed-rank test -- nonparametric paired comparison of
%                                     per-sample absolute percentage error
%                                     (APE) across competing models. Robust
%                                     to non-normal MAPE distributions.
%     3. Modified Diebold-Mariano (Harvey-Leybourne-Newbold 1997) -- the
%                                     standard parametric test for paired
%                                     forecast accuracy comparison in
%                                     time-series forecasting literature.
%
%   References:
%     Diebold & Mariano (1995), J. Bus. Econ. Stat. 13(3).
%     Harvey, Leybourne, Newbold (1997), Int. J. Forecasting 13(2).
%     Bao, Xiong, Hu (2014), Energies 7(7):4185 -- Refined DM for wind
%                                                  power forecasting.
%     Tsamardinos et al. -- Wilcoxon for paired CV comparisons.
%
%   Output:
%     Prints a formatted table of all pairwise p-values to console and
%     writes results/tables_paper/significance_<stage>.csv

    if nargin < 1; stage = 'cluster'; end

    here   = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));
    tabDir = fullfile(here, 'results', 'tables');
    outDir = fullfile(here, 'results', 'tables_paper');
    if ~isfolder(outDir); mkdir(outDir); end

    switch lower(stage)
        case {'cluster','stage2','exp2'}
            matFile = fullfile(tabDir, 'exp2_results.mat');
            label   = 'cluster';
        case {'aggregate','stage2b','stage5','exp2b'}
            matFile = fullfile(tabDir, 'exp2b_aggregate_results.mat');
            label   = 'aggregate';
        otherwise
            error('Unknown stage: %s', stage);
    end

    assert(isfile(matFile), 'Result file not found: %s\nRun the experiment first.', matFile);
    S = load(matFile);
    R = S.results;
    feats = R.feats;
    yt    = feats.Y(:, :, feats.idx.test);            % [H, N, S]

    modelNames = setdiff(fieldnames(R), {'feats'}, 'stable');
    fprintf('\n=========================================================\n');
    fprintf('  STATISTICAL SIGNIFICANCE -- Stage %s\n', label);
    fprintf('  Models: %s\n', strjoin(modelNames, ', '));
    fprintf('=========================================================\n');

    %% --- Compute per-sample APE for each model -----------------------------
    % Goal: collapse [H, N, S] -> [S, 1] so pairs can be compared sample-wise.
    % APE is averaged over (H, N) to yield a single scalar per test window.
    nS = size(yt, 3);
    perSampleAPE = struct();
    for k = 1:numel(modelNames)
        yh = double(R.(modelNames{k}).yhat);
        % De-normalize both (already handled inside evaluate_metrics, but here
        % we keep them in normalized space since APE ratios are invariant
        % under per-bus z-score multiplication).
        diff = yh - double(yt);
        absDf = abs(diff);
        ape  = absDf ./ max(abs(double(yt)), 1e-6);    % per-element APE
        ape  = min(ape, 5.0);                          % clip 500% per evaluate_metrics
        perSampleAPE.(modelNames{k}) = squeeze(mean(ape, [1 2])) * 100;  % [S x 1]
    end

    %% --- Lilliefors normality test per model -------------------------------
    fprintf('\n--- Lilliefors normality test on per-sample APE ---\n');
    fprintf('  (H0: APE distribution is normal; reject => use nonparametric)\n');
    normalityRows = {};
    for k = 1:numel(modelNames)
        x = perSampleAPE.(modelNames{k});
        try
            [h, p] = lillietest(x);
        catch
            h = NaN; p = NaN;
        end
        verdict = 'non-normal';
        if ~isnan(h) && h == 0; verdict = 'normal'; end
        fprintf('  %-16s  p = %.4g   verdict: %s\n', modelNames{k}, p, verdict);
        normalityRows(end+1, :) = {modelNames{k}, p, verdict}; %#ok<AGROW>
    end
    Tnorm = cell2table(normalityRows, ...
        'VariableNames', {'Model','Lilliefors_p','Verdict'});

    %% --- Pairwise Wilcoxon signed-rank --------------------------------------
    fprintf('\n--- Wilcoxon signed-rank test (paired, two-sided) ---\n');
    fprintf('  (H0: median APE difference = 0; p<0.05 => significantly different)\n');
    n = numel(modelNames);
    wilcoxon_p  = NaN(n, n);
    wilcoxon_h  = NaN(n, n);
    for i = 1:n
        for j = (i+1):n
            a = perSampleAPE.(modelNames{i});
            b = perSampleAPE.(modelNames{j});
            [p, h] = signrank(a, b);
            wilcoxon_p(i, j) = p; wilcoxon_p(j, i) = p;
            wilcoxon_h(i, j) = h; wilcoxon_h(j, i) = h;
            sig = ternary(h==1, '*', ' ');
            fprintf('  %-16s vs %-16s   p = %.4g  %s\n', ...
                modelNames{i}, modelNames{j}, p, sig);
        end
    end

    %% --- Diebold-Mariano test (Harvey-Leybourne-Newbold corrected) ---------
    fprintf('\n--- Diebold-Mariano (HLN-corrected) test, h=1, q=2 ---\n');
    fprintf('  (H0: forecasts have equal accuracy; p<0.05 => sig. different)\n');
    dm_p = NaN(n, n);
    for i = 1:n
        for j = (i+1):n
            % Use squared-error loss differential for DM (standard q=2)
            e1 = squeeze(mean((double(R.(modelNames{i}).yhat) - double(yt)).^2, [1 2]));
            e2 = squeeze(mean((double(R.(modelNames{j}).yhat) - double(yt)).^2, [1 2]));
            p = dm_test_hln(e1, e2, 1);
            dm_p(i, j) = p; dm_p(j, i) = p;
            sig = ternary(p < 0.05, '*', ' ');
            fprintf('  %-16s vs %-16s   p = %.4g  %s\n', ...
                modelNames{i}, modelNames{j}, p, sig);
        end
    end

    %% --- Save CSV ----------------------------------------------------------
    % Pairwise long-form table
    pairs = {};
    for i = 1:n
        for j = (i+1):n
            pairs(end+1, :) = {modelNames{i}, modelNames{j}, ...
                wilcoxon_p(i, j), wilcoxon_h(i, j), dm_p(i, j)}; %#ok<AGROW>
        end
    end
    Tpairs = cell2table(pairs, 'VariableNames', ...
        {'ModelA','ModelB','Wilcoxon_p','Wilcoxon_sig','DM_HLN_p'});
    csvPath = fullfile(outDir, sprintf('significance_%s.csv', label));
    writetable(Tpairs, csvPath);
    writetable(Tnorm,  fullfile(outDir, sprintf('normality_%s.csv', label)));
    fprintf('\nWrote %s and normality_%s.csv\n', csvPath, label);

    %% --- Paper-ready summary ----------------------------------------------
    fprintf('\n=========================================================\n');
    fprintf('  PAPER-READY INTERPRETATION (Stage %s)\n', label);
    fprintf('=========================================================\n');
    n_tied = 0; n_diff = 0;
    for i = 1:n
        for j = (i+1):n
            if wilcoxon_p(i, j) > 0.05 && dm_p(i, j) > 0.05
                n_tied = n_tied + 1;
            elseif wilcoxon_p(i, j) < 0.05 || dm_p(i, j) < 0.05
                n_diff = n_diff + 1;
            end
        end
    end
    fprintf('  Total pairs: %d | statistically tied: %d | distinguishable: %d\n', ...
        n*(n-1)/2, n_tied, n_diff);
    if n_tied >= n_diff
        fprintf('  >> Most model pairs are STATISTICALLY INDISTINGUISHABLE\n');
        fprintf('  >> Paper claim "DL models converge to similar accuracy" is SUPPORTED.\n');
    else
        fprintf('  >> Most model pairs differ significantly; rank ordering is meaningful.\n');
    end
end

% =========================================================================
function p = dm_test_hln(e1, e2, h)
%DM_TEST_HLN  Modified Diebold-Mariano test (Harvey, Leybourne, Newbold 1997).
%
%   p = dm_test_hln(e1, e2, h) returns two-sided p-value testing H0 that
%   the two forecast error series have equal accuracy under squared loss.
%
%   The HLN correction adjusts for small-sample bias by:
%     1. Multiplying the DM statistic by sqrt((T+1-2h+h(h-1)/T)/T)
%     2. Using t-distribution with T-1 d.o.f. instead of standard normal.
%
%   Inputs:
%     e1, e2 : column vectors of per-sample forecast errors (squared loss
%              terms are e1.^2 and e2.^2; we expect *errors*, not losses)
%              -- but here we already pass squared MSE per-sample, so we
%              use loss differentials d = e1 - e2 directly.
%     h      : forecast horizon (=1 for one-step-ahead)
%
%   Reference: Harvey, Leybourne, Newbold (1997).
%
    d = e1 - e2;
    T = numel(d);
    d = d(:);
    dbar = mean(d);
    % Newey-West HAC variance with lag h-1 (=0 for h=1)
    if h == 1
        var_d = var(d, 1);                          % MLE variance
    else
        gamma0 = var(d, 1);
        gamma = zeros(h-1, 1);
        for k = 1:(h-1)
            gamma(k) = sum((d(1:end-k) - dbar) .* (d(1+k:end) - dbar)) / T;
        end
        var_d = gamma0 + 2 * sum(gamma);
    end
    if var_d <= 0
        p = NaN; return;
    end
    DM = dbar / sqrt(var_d / T);
    % HLN correction factor
    corr = sqrt((T + 1 - 2*h + h*(h-1)/T) / T);
    DM_hln = DM * corr;
    % Two-sided p-value using t-distribution with T-1 d.o.f.
    p = 2 * (1 - tcdf(abs(DM_hln), T - 1));
end

% =========================================================================
function y = ternary(c, a, b)
    if c; y = a; else; y = b; end
end
