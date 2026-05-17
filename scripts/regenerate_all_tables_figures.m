function regenerate_all_tables_figures(varargin)
%REGENERATE_ALL_TABLES_FIGURES  One-click reproduction of every paper artefact.
%
%   This is the script promised in §dataavailability of the accompanying
%   Energies paper. Given that the upstream stages
%   (run_paper / exp1 / exp2 / exp2b / exp3) have already produced their
%   *.mat caches under results/tables/, this script walks the cache and
%   re-emits every CSV (results/tables_paper/) and PDF
%   (results/figures/) that appears in the paper.
%
%   USAGE
%     regenerate_all_tables_figures                       % run everything
%     regenerate_all_tables_figures('Skip', {'figures'})  % tables only
%     regenerate_all_tables_figures('Skip', {'tables'})   % figures only
%
%   Total wall-clock budget on a modern laptop: ≈ 5 minutes.

    p = inputParser;
    addParameter(p, 'Skip', {}, @iscellstr);
    parse(p, varargin{:});
    skip = p.Results.Skip;

    here     = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(here);                          % parent of scripts/
    addpath(fullfile(repoRoot, 'src'));
    addpath(fullfile(repoRoot, 'experiments'));
    addpath(fullfile(repoRoot, 'analysis'));

    fprintf('======================================================\n');
    fprintf('  TA-Transformer paper · one-click regeneration\n');
    fprintf('  Repo: %s\n', repoRoot);
    fprintf('======================================================\n\n');

    tableDir  = fullfile(repoRoot, 'results', 'tables');
    paperDir  = fullfile(repoRoot, 'results', 'tables_paper');
    figureDir = fullfile(repoRoot, 'results', 'figures');
    for d = {tableDir, paperDir, figureDir}
        if ~isfolder(d{1}); mkdir(d{1}); end
    end

    % ---- prerequisite check ----
    must_have = { ...
        'exp1_results.mat',           'Stage 1 (GEFCom2014)'; ...
        'exp2_results.mat',           'Stage 2 (UMass-33 cluster)'; ...
        'exp2b_aggregate_results.mat','Stage 2b (system-aggregate)'; ...
        'exp2b_horizons_results.mat', 'Stage 2b multi-horizon'};
    missing = {};
    for k = 1:size(must_have, 1)
        if ~isfile(fullfile(tableDir, must_have{k, 1}))
            missing{end+1} = sprintf('  - %s  (%s)', ...
                must_have{k, 1}, must_have{k, 2}); %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        fprintf(2, '[regenerate] Missing upstream caches:\n');
        fprintf(2, '%s\n', strjoin(missing, char(10)));
        fprintf(2, '  → run "run_paper" first, or download the result-tensor\n');
        fprintf(2, '    archive from the GitHub Release page.\n');
        error('regenerate:missingCache', 'Cannot proceed without upstream caches.');
    end

    % ---- tables (CSV) ----
    if ~ismember('tables', skip)
        section('Regenerating CSV tables');

        % Significance + normality (Tables 5, 6)
        run_step('stats_significance', @() stats_significance('cluster'));
        run_step('stats_significance', @() stats_significance('aggregate'));

        % Holiday-week sensitivity (Table 9 in §5.5)
        run_step('holiday_sensitivity', @holiday_sensitivity);

        % Complexity / latency (Table in §4.5 paragraph)
        run_step('complexity_table', @complexity_table);

        % Re-evaluated WAPE for Table 4 (if not already there)
        run_step('reeval_exp2_wape', @reeval_exp2_wape);
    end

    % ---- figures (PDF) ----
    if ~ismember('figures', skip)
        section('Regenerating PDF figures');
        run_step('replot_horizons',    @replot_horizons);
        run_step('replot_predictions', @replot_predictions);
        run_step('make_aux_figures',   @make_aux_figures);
    end

    % ---- summary ----
    section('Output inventory');
    list_dir('CSV (tables_paper/)', paperDir, '*.csv');
    list_dir('CSV (tables/)',       tableDir, '*.csv');
    list_dir('PDF (figures/)',      figureDir, '*.pdf');

    fprintf('\n[regenerate] complete.\n');
end

% =====================================================================
function section(s)
    fprintf('\n------ %s ------\n', s);
end

function run_step(name, fn)
    t0 = tic;
    fprintf('  · %-30s ... ', name);
    try
        fn();
        fprintf('OK (%.1fs)\n', toc(t0));
    catch ME
        fprintf(2, 'FAILED: %s\n', ME.message);
    end
end

function list_dir(title, d, pat)
    fprintf('  %s\n', title);
    items = dir(fullfile(d, pat));
    if isempty(items)
        fprintf('    (none)\n');
        return;
    end
    for k = 1:numel(items)
        fprintf('    %s  (%.1f kB)\n', items(k).name, items(k).bytes/1024);
    end
end
