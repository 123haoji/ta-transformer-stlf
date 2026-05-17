function make_aux_figures()
%MAKE_AUX_FIGURES  Generate three auxiliary figures for the paper:
%
%   Fig (A) fig_cluster_errorbar.pdf   -- 6-model MAPE +/- std bar chart (Stage 2)
%   Fig (B) fig_ebike_bar.pdf          -- E-bike feature with vs without bar chart (Stage 3)
%   Fig (C) fig_wilcoxon_heatmap.pdf   -- 6x6 Wilcoxon signed-rank heatmap (Stage 2)
%
%   Reads:
%     results/tables_paper/exp2_metrics_aggregated.csv
%     results/tables_paper/exp3_ebike_sensitivity_aggregated.csv
%     results/tables_paper/significance_cluster.csv
%
%   Writes:
%     results/figures/fig_cluster_errorbar.pdf
%     results/figures/fig_ebike_bar.pdf
%     results/figures/fig_wilcoxon_heatmap.pdf

    here   = fileparts(mfilename('fullpath'));
    inDir  = fullfile(here, 'results', 'tables_paper');
    outDir = fullfile(here, 'results', 'figures');
    if ~isfolder(outDir); mkdir(outDir); end

    %% =========================================================
    %% Fig A: 6-model cluster MAPE +/- std errorbar
    %% =========================================================
    T = readtable(fullfile(inDir, 'exp2_metrics_aggregated.csv'));
    modelOrder = {'ARIMA','LSTM','CNN_LSTM','Transformer','TA_Transformer','PureGAT'};
    [~, ord] = ismember(modelOrder, T.Model);
    T = T(ord(ord > 0), :);

    nM   = height(T);
    mape = T.MAPE_percent_mean;
    sd   = T.MAPE_percent_std;
    wape = T.WAPE_percent_mean;
    wsd  = T.WAPE_percent_std;

    fig = figure('Color','w','Position',[100 100 760 380]);
    x = 1:nM;
    w = 0.36;
    cm = lines(6);
    cBar1 = cm(1, :);    % MAPE blue
    cBar2 = [0.85 0.45 0.10];  % WAPE orange

    % Group bars: MAPE (left), WAPE (right)
    b1 = bar(x - w/2, mape, w, 'FaceColor', cBar1, 'EdgeColor', 'none', ...
             'DisplayName', 'MAPE');
    hold on;
    b2 = bar(x + w/2, wape, w, 'FaceColor', cBar2, 'EdgeColor', 'none', ...
             'DisplayName', 'WAPE');
    errorbar(x - w/2, mape, sd, 'k', 'LineStyle','none', 'LineWidth',1.0, 'CapSize',6);
    errorbar(x + w/2, wape, wsd, 'k', 'LineStyle','none', 'LineWidth',1.0, 'CapSize',6);

    set(gca, 'XTick', x, ...
             'XTickLabel', strrep(modelOrder, '_', '-'), ...
             'FontName', 'Helvetica', 'FontSize', 10);
    ylabel('Error (%)');
    title('Cluster-level forecasting error on UMass-33 (3 seeds, mean \pm std)');
    legend([b1 b2], 'Location', 'northwest', 'Box', 'on', ...
           'EdgeColor', [0.7 0.7 0.7]);
    grid on; box on;
    yl = ylim(); ylim([0, yl(2) * 1.05]);

    outA = fullfile(outDir, 'fig_cluster_errorbar.pdf');
    exportgraphics(fig, outA, 'ContentType', 'vector');
    fprintf('  Wrote %s\n', outA);

    %% =========================================================
    %% Fig B: E-bike with/without grouped bar
    %% =========================================================
    T = readtable(fullfile(inDir, 'exp3_ebike_sensitivity_aggregated.csv'));
    mu = T.multiplier;
    nMu = numel(mu);

    fig = figure('Color','w','Position',[100 100 720 380]);
    x = 1:nMu;
    w = 0.36;
    cWith = [0.20 0.55 0.85];   % blue
    cWo   = [0.65 0.65 0.65];   % gray

    b1 = bar(x - w/2, T.MAPE_with_mean, w, 'FaceColor', cWith, 'EdgeColor','none', ...
             'DisplayName', 'with e-bike covariate');
    hold on;
    b2 = bar(x + w/2, T.MAPE_wo_mean,   w, 'FaceColor', cWo,   'EdgeColor','none', ...
             'DisplayName', 'without e-bike covariate');
    errorbar(x - w/2, T.MAPE_with_mean, T.MAPE_with_std, 'k', 'LineStyle','none', ...
             'LineWidth', 1.0, 'CapSize', 6);
    errorbar(x + w/2, T.MAPE_wo_mean,   T.MAPE_wo_std,   'k', 'LineStyle','none', ...
             'LineWidth', 1.0, 'CapSize', 6);

    set(gca, 'XTick', x, ...
             'XTickLabel', arrayfun(@(v) sprintf('\\mu = %.1f', v), mu, ...
                                    'UniformOutput', false), ...
             'TickLabelInterpreter','tex', ...
             'FontName', 'Helvetica', 'FontSize', 10);
    xlabel('E-bike penetration multiplier');
    ylabel('MAPE (%)');
    title('E-bike feature sensitivity on Dataset C (3 seeds, mean \pm std)');
    legend([b1 b2], 'Location', 'northwest', 'Box', 'on', ...
           'EdgeColor', [0.7 0.7 0.7]);
    grid on; box on;
    yl = ylim(); ylim([0, yl(2) * 1.10]);

    outB = fullfile(outDir, 'fig_ebike_bar.pdf');
    exportgraphics(fig, outB, 'ContentType', 'vector');
    fprintf('  Wrote %s\n', outB);

    %% =========================================================
    %% Fig C: 6x6 Wilcoxon p-value heatmap (cluster level)
    %% =========================================================
    T = readtable(fullfile(inDir, 'significance_cluster.csv'));
    models = {'ARIMA','LSTM','CNN_LSTM','Transformer','TA_Transformer','PureGAT'};
    nM = numel(models);
    P  = NaN(nM, nM);

    for r = 1:height(T)
        a = T.ModelA{r};
        b = T.ModelB{r};
        ia = find(strcmp(models, a));
        ib = find(strcmp(models, b));
        if ~isempty(ia) && ~isempty(ib)
            P(ia, ib) = T.Wilcoxon_p(r);
            P(ib, ia) = T.Wilcoxon_p(r);
        end
    end
    P(1:nM+1:end) = 1;     % diagonal = 1 (model vs itself)

    % Heatmap data: log10(p), clipped at -50 for color stability
    L  = log10(P);
    Lc = max(L, -50);

    fig = figure('Color','w','Position',[100 100 580 500]);
    imagesc(Lc);
    colormap(flipud(parula));
    cb = colorbar;
    cb.Label.String   = 'log_{10} p  (Wilcoxon signed-rank)';
    cb.Label.FontSize = 11;
    caxis([-50 0]);

    set(gca, 'XTick', 1:nM, 'YTick', 1:nM, ...
             'XTickLabel', strrep(models, '_', '-'), ...
             'YTickLabel', strrep(models, '_', '-'), ...
             'FontName', 'Helvetica', 'FontSize', 9, ...
             'TickLength', [0 0]);
    xtickangle(35);
    axis square; box on;
    title('Pairwise Wilcoxon p-values, cluster level (UMass-33)', ...
          'FontSize', 12);

    % Cell annotations (skip diagonal). Use plain TeX (default interpreter)
    % so superscripts and \times render without LaTeX parser dependencies.
    for i = 1:nM
        for j = 1:nM
            if i == j
                txt = '---';
                col = [0.5 0.5 0.5];
            else
                if P(i, j) >= 1e-3
                    txt = sprintf('%.3f', P(i, j));
                else
                    e        = floor(log10(P(i, j)));
                    mantissa = P(i, j) / 10^e;
                    txt = sprintf('%.1f\\times10^{%d}', mantissa, e);
                end
                if Lc(i, j) < -25; col = [1 1 1]; else; col = [0 0 0]; end
            end
            text(j, i, txt, 'HorizontalAlignment','center', ...
                 'VerticalAlignment','middle', ...
                 'FontName','Helvetica', 'FontSize', 7.5, ...
                 'Color', col, 'Interpreter', 'tex');
        end
    end

    outC = fullfile(outDir, 'fig_wilcoxon_heatmap.pdf');
    exportgraphics(fig, outC, 'ContentType', 'vector');
    fprintf('  Wrote %s\n', outC);

    fprintf('\n  make_aux_figures done.\n');
end
