function main(varargin)
%MAIN  Top-level entry point for the TA-Transformer paper experiments.
%
%   main()                              run the full pipeline (Exp 1..4)
%   main('Stage', 2)                    run only Stage 2 (UMass-33 main)
%   main('Stage', [1 2])                run Stages 1 and 2
%   main('Stage', 2, 'Quick', true)     small-scale debug pass
%
%   Outputs (results, figures, tables) are written to ../results/.

    p = inputParser;
    % Stage 5 = exp2b_aggregate (system-level aggregate forecast supplement).
    % We allow 1..5 so users can invoke it via main('Stage', 5).
    addParameter(p, 'Stage', 1:4, @(x) isnumeric(x) && all(x>=1 & x<=5));
    addParameter(p, 'Quick', false, @islogical);
    parse(p, varargin{:});
    cfg = p.Results;

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));
    addpath(fullfile(here, 'experiments'));
    addpath(fullfile(here, 'analysis'));
    addpath(fullfile(here, 'scripts'));

    fprintf('======================================================\n');
    fprintf('  TA-Transformer paper pipeline\n');
    fprintf('  Stages: %s   Quick: %d\n', mat2str(cfg.Stage), cfg.Quick);
    fprintf('======================================================\n\n');

    if ismember(1, cfg.Stage)
        fprintf('\n>>> Stage 1: GEFCom2014 single-zone benchmark\n');
        try
            exp1_gefcom_baseline(cfg.Quick);
        catch ME
            fprintf(2, '[Stage 1 FAILED] %s\n', ME.message);
        end
    end

    if ismember(2, cfg.Stage)
        fprintf('\n>>> Stage 2: UMass-33 multi-bus main experiment\n');
        try
            exp2_umass_main(cfg.Quick);
        catch ME
            fprintf(2, '[Stage 2 FAILED] %s\n', ME.message);
        end
    end

    if ismember(5, cfg.Stage)
        fprintf('\n>>> Stage 2b: System-aggregate forecast (paper headline)\n');
        try
            exp2b_aggregate(cfg.Quick);
        catch ME
            fprintf(2, '[Stage 2b FAILED] %s\n', ME.message);
            for i = 1:length(ME.stack)
                fprintf(2, '   at %s:%d\n', ME.stack(i).name, ME.stack(i).line);
            end
        end
    end

    if ismember(3, cfg.Stage)
        fprintf('\n>>> Stage 3: E-bike penetration sensitivity (Dataset C)\n');
        try
            exp3_ebike_sensitivity(cfg.Quick);
        catch ME
            fprintf(2, '[Stage 3 FAILED] %s\n', ME.message);
        end
    end

    if ismember(4, cfg.Stage)
        fprintf('\n>>> Stage 4: Zero-shot 443-household generalization\n');
        try
            exp4_zeroshot_443(cfg.Quick);
        catch ME
            fprintf(2, '[Stage 4 FAILED] %s\n', ME.message);
        end
    end

    print_summary(here);
    fprintf('\n[main] pipeline complete.\n');
end

% ============================================================================
function print_summary(here)
    tableDir = fullfile(here, 'results', 'tables');
    if ~isfolder(tableDir); return; end
    csvs = dir(fullfile(tableDir, '*.csv'));
    if isempty(csvs)
        fprintf('\n  (no result CSVs produced)\n');
        return;
    end
    fprintf('\n======================================================\n');
    fprintf('  RESULTS SUMMARY  (CSV files in %s)\n', tableDir);
    fprintf('======================================================\n');
    for i = 1:numel(csvs)
        f = fullfile(csvs(i).folder, csvs(i).name);
        try
            T = readtable(f);
            fprintf('\n--- %s ---\n', csvs(i).name);
            disp(T);
        catch ME
            fprintf('  could not read %s: %s\n', csvs(i).name, ME.message);
        end
    end
end
