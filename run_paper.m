function run_paper(varargin)
%RUN_PAPER  Production-grade experiment runner for the TA-Transformer paper.
%
%   This is the entry point for results intended for publication.  Unlike
%   MAIN('Quick', true) it uses the full datasets, longer training, and
%   multi-seed averaging.  Expect a full run to take 3-6 hours on a single
%   modern GPU.
%
%   USAGE
%     run_paper                          % all 4 stages, 3 seeds
%     run_paper('Stage', 2)              % only stage 2
%     run_paper('Seeds', [42 1 2 3 4])   % five seeds
%     run_paper('Stage', 3, 'Seeds', 1:5)
%
%   The runner writes seed-averaged metrics tables to results/tables_paper/
%   so that they don't overwrite Quick-mode results.

    p = inputParser;
    addParameter(p, 'Stage', [1 2 3 4]);
    addParameter(p, 'Seeds', [42 1 2]);            % 3 seeds by default
    addParameter(p, 'OutDir', '');                 % override default
    parse(p, varargin{:});
    stages = p.Results.Stage;
    seeds  = p.Results.Seeds;

    here    = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'), fullfile(here, 'experiments'));
    if isempty(p.Results.OutDir)
        outDir = fullfile(here, 'results', 'tables_paper');
    else
        outDir = p.Results.OutDir;
    end
    if ~isfolder(outDir); mkdir(outDir); end

    fprintf('======================================================\n');
    fprintf('  TA-Transformer PAPER pipeline (full mode)\n');
    fprintf('  Stages: %s   Seeds: %s\n', mat2str(stages), mat2str(seeds));
    fprintf('  Out: %s\n', outDir);
    fprintf('======================================================\n');

    %% Stage 3 with multi-seed averaging is the most paper-relevant.
    if ismember(3, stages)
        fprintf('\n>>> Stage 3 (multi-seed): Urban-village e-bike sensitivity\n');
        run_stage3_multiseed(seeds, outDir);
    end

    if ismember(1, stages)
        fprintf('\n>>> Stage 1 (multi-seed): GEFCom2014 single-zone benchmark\n');
        run_stage1_multiseed(seeds, outDir);
    end

    if ismember(2, stages)
        fprintf('\n>>> Stage 2 (multi-seed): UMass-33 main experiment\n');
        run_stage2_multiseed(seeds, outDir);
    end

    if ismember(4, stages)
        fprintf('\n>>> Stage 4 (multi-seed): Zero-shot 443-household generalization\n');
        run_stage4_multiseed(seeds, outDir);
    end

    if ismember(5, stages)
        fprintf('\n>>> Stage 2b (multi-seed): System aggregate forecast (headline)\n');
        run_stage5_multiseed(seeds, outDir);
    end

    fprintf('\n[run_paper] done.\n');
end

% ---------------------------------------------------------------------------
function run_stage3_multiseed(seeds, outDir)
    % We re-run exp3 across seeds and aggregate (mean ± std) the MAPE table.
    nS = numel(seeds);
    multipliers = [0.5, 1.0, 2.0, 4.0];
    nM = numel(multipliers);
    all_MAPE_with    = zeros(nM, nS);
    all_MAPE_wo      = zeros(nM, nS);
    all_PH_MAPE_with = zeros(nM, nS);
    all_PH_MAPE_wo   = zeros(nM, nS);

    for si = 1:nS
        rng(seeds(si));
        fprintf('  --- seed %d ---\n', seeds(si));
        % We call exp3 in non-quick mode but capture per-seed results.
        tbl = exp3_ebike_sensitivity(false);
        all_MAPE_with(:, si)    = tbl.MAPE_with;
        all_MAPE_wo(:, si)      = tbl.MAPE_wo;
        all_PH_MAPE_with(:, si) = tbl.PH_MAPE_with;
        all_PH_MAPE_wo(:, si)   = tbl.PH_MAPE_wo;
    end

    Tagg = table(multipliers', ...
                 mean(all_MAPE_with, 2),    std(all_MAPE_with, 0, 2), ...
                 mean(all_MAPE_wo,   2),    std(all_MAPE_wo,   0, 2), ...
                 mean(all_PH_MAPE_with, 2), std(all_PH_MAPE_with, 0, 2), ...
                 mean(all_PH_MAPE_wo,   2), std(all_PH_MAPE_wo,   0, 2), ...
        'VariableNames', {'multiplier', ...
                          'MAPE_with_mean','MAPE_with_std', ...
                          'MAPE_wo_mean',  'MAPE_wo_std',   ...
                          'PH_MAPE_with_mean','PH_MAPE_with_std', ...
                          'PH_MAPE_wo_mean','PH_MAPE_wo_std'});
    writetable(Tagg, fullfile(outDir, 'exp3_ebike_sensitivity_aggregated.csv'));
    disp(Tagg);
end

% ---------------------------------------------------------------------------
function run_stage1_multiseed(seeds, outDir)
    here       = fileparts(mfilename('fullpath'));
    perSeedCsv = fullfile(here, 'results', 'tables', 'exp1_metrics.csv');
    seedTables = cell(numel(seeds), 1);
    for si = 1:numel(seeds)
        rng(seeds(si));
        fprintf('  --- seed %d ---\n', seeds(si));
        exp1_gefcom_baseline(false);
        seedTables{si} = readtable(perSeedCsv);
    end
    Tagg = aggregate_by_model(seedTables);
    writetable(Tagg, fullfile(outDir, 'exp1_metrics_aggregated.csv'));
    disp(Tagg);
end

% ---------------------------------------------------------------------------
function run_stage2_multiseed(seeds, outDir)
    here       = fileparts(mfilename('fullpath'));
    perSeedCsv = fullfile(here, 'results', 'tables', 'exp2_metrics.csv');
    seedTables = cell(numel(seeds), 1);
    for si = 1:numel(seeds)
        rng(seeds(si));
        fprintf('  --- seed %d ---\n', seeds(si));
        exp2_umass_main(false);
        seedTables{si} = readtable(perSeedCsv);
    end
    Tagg = aggregate_by_model(seedTables);
    writetable(Tagg, fullfile(outDir, 'exp2_metrics_aggregated.csv'));
    disp(Tagg);
end

% ---------------------------------------------------------------------------
function run_stage4_multiseed(seeds, outDir)
    here       = fileparts(mfilename('fullpath'));
    perSeedCsv = fullfile(here, 'results', 'tables', 'exp4_zeroshot.csv');
    seedRows   = cell(numel(seeds), 1);
    for si = 1:numel(seeds)
        rng(seeds(si));
        fprintf('  --- seed %d ---\n', seeds(si));
        exp4_zeroshot_443(false);
        seedRows{si} = readtable(perSeedCsv);
    end
    Tagg = aggregate_single_row(vertcat(seedRows{:}));
    writetable(Tagg, fullfile(outDir, 'exp4_zeroshot_aggregated.csv'));
    disp(Tagg);
end

% ---------------------------------------------------------------------------
function run_stage5_multiseed(seeds, outDir)
    % Stage 2b (system aggregate, headline paper number) multi-seed.
    % Mirrors run_stage2_multiseed but reads exp2b_aggregate_metrics.csv.
    here       = fileparts(mfilename('fullpath'));
    perSeedCsv = fullfile(here, 'results', 'tables', 'exp2b_aggregate_metrics.csv');
    seedTables = cell(numel(seeds), 1);
    for si = 1:numel(seeds)
        rng(seeds(si));
        fprintf('  --- seed %d ---\n', seeds(si));
        exp2b_aggregate(false);
        seedTables{si} = readtable(perSeedCsv);
    end
    Tagg = aggregate_by_model(seedTables);
    writetable(Tagg, fullfile(outDir, 'exp2b_aggregate_metrics_aggregated.csv'));
    disp(Tagg);
end

% ---------------------------------------------------------------------------
function Tagg = aggregate_by_model(seedTables)
    % Stack per-seed tables, group by Model, output mean and std per numeric column.
    T_all   = vertcat(seedTables{:});
    models  = unique(T_all.Model, 'stable');
    valCols = T_all.Properties.VariableNames(2:end);          % drop 'Model'
    nM = numel(models); nC = numel(valCols);
    Vmean = zeros(nM, nC); Vstd = zeros(nM, nC);
    for mi = 1:nM
        sub = T_all(strcmp(T_all.Model, models{mi}), 2:end);
        Vmean(mi, :) = mean(sub{:, :}, 1, 'omitnan');
        Vstd(mi, :)  = std(sub{:, :}, 0, 1, 'omitnan');
    end
    Tagg = table(models, 'VariableNames', {'Model'});
    for ci = 1:nC
        Tagg.([valCols{ci} '_mean']) = Vmean(:, ci);
        Tagg.([valCols{ci} '_std'])  = Vstd(:, ci);
    end
end

% ---------------------------------------------------------------------------
function Tagg = aggregate_single_row(T_all)
    % exp4-shape: single-row-per-seed CSV with no key column.
    keep = varfun(@isnumeric, T_all, 'OutputFormat', 'uniform');
    cols = T_all.Properties.VariableNames(keep);
    Tagg = table();
    for ci = 1:numel(cols)
        v = T_all{:, cols{ci}};
        Tagg.([cols{ci} '_mean']) = mean(v, 'omitnan');
        Tagg.([cols{ci} '_std'])  = std(v, 0, 'omitnan');
    end
end
