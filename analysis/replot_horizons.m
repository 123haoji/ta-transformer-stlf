function replot_horizons(outDir)
%REPLOT_HORIZONS  Regenerate Figure 2 (multi-horizon MAPE) from the existing
%   exp2b_horizons_metrics.csv with the legend pinned to the upper-left
%   corner so it no longer overlaps the data lines.
%
%   Usage:
%     >> replot_horizons()                     % writes to results/figures
%     >> replot_horizons('/some/other/dir')    % custom output dir
%
%   No GPU work, no retraining; reads the saved CSV and re-renders in <2 s.

    here  = fileparts(mfilename('fullpath'));
    csvIn = fullfile(here, 'results', 'tables', 'exp2b_horizons_metrics.csv');

    if nargin < 1 || isempty(outDir)
        outDir = fullfile(here, 'results', 'figures');
    end
    if ~isfolder(outDir); mkdir(outDir); end

    if ~isfile(csvIn)
        error('replot_horizons:noCSV', ...
              'Cannot find %s.  Run exp2b_horizons() first.', csvIn);
    end
    T = readtable(csvIn);

    models   = unique(T.Model,    'stable');
    horizons = unique(T.H_steps,  'stable');
    nM       = numel(models);

    fig = figure('Color', 'w', 'Position', [100 100 720 380]);
    colors = lines(nM);
    hold on;
    for m = 1:nM
        sub = T(strcmp(T.Model, models{m}), :);
        plot(sub.H_steps, sub.MAPE_percent, '-o', ...
             'Color', colors(m, :), 'LineWidth', 1.5, ...
             'MarkerFaceColor', colors(m, :), ...
             'DisplayName', strrep(models{m}, '_', '-'));
    end
    set(gca, 'XScale', 'log', 'XTick', horizons, ...
             'XTickLabel', arrayfun(@(h) sprintf('%dx15m', h), ...
                                    horizons, 'UniformOutput', false));
    grid on;
    xlabel('Forecast horizon H (15-min steps)');
    ylabel('MAPE (%)');
    title('Aggregate forecast accuracy vs. horizon (UMass Smart*, N=1)');

    % --- Pin legend to upper-left + tighten ---------------------------------
    lh = legend('Location', 'northwest', 'NumColumns', 2, ...
                'Box', 'on', 'EdgeColor', [0.7 0.7 0.7], ...
                'FontSize', 9);
    lh.ItemTokenSize = [18, 10];           % shorten the marker stub

    % Small y-axis padding so legend never touches the top axis frame.
    yl = ylim();
    ylim([yl(1), yl(2) * 1.05]);

    outPath = fullfile(outDir, 'fig_horizon_mape_aggregate_multimodel.pdf');
    exportgraphics(fig, outPath, 'ContentType', 'vector');
    fprintf('  Wrote %s\n', outPath);

    fprintf('  replot_horizons done.\n');
end
