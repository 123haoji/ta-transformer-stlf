function plot_results(results, outDir, opts)
%PLOT_RESULTS  Generate paper figures from a results bundle.
%
%   Expects results.<model> = struct(yhat, metrics, name) and a feats
%   reference passed in results.feats.
%
%   USAGE
%     plot_results(results)                          % default cluster level
%     plot_results(results, outDir)
%     plot_results(results, outDir, 'Suffix','_aggregate')   % avoid overwrite
%     plot_results(results, outDir, 'Suffix','_aggregate', 'Title','Aggregate')
%
%   When feats.dims.N == 1 (aggregate single-series), the multi-bus
%   prediction figure adapts to show a longer time window of the single
%   series instead of multiple bus panels.

    arguments
        results (1,1) struct
        outDir         = ''
        opts.Suffix    = ''
        opts.Title     = ''       % optional title prefix (e.g. 'Aggregate')
    end

    if isempty(outDir)
        here = fileparts(mfilename('fullpath'));
        outDir = fullfile(here, '..', 'results', 'figures');
    end
    if ~isfolder(outDir); mkdir(outDir); end

    sfx = opts.Suffix;
    titlePrefix = opts.Title;
    if ~isempty(titlePrefix); titlePrefix = [titlePrefix ' - ']; end

    modelNames = setdiff(fieldnames(results), {'feats'}, 'stable');
    feats = results.feats;
    N = feats.dims.N;

    % ---- Figure 1: bar comparison MAPE / RMSE / PH-MAPE -----------------
    nM = numel(modelNames);
    metricNames = {'MAPE','RMSE','PH_MAPE'};
    values = zeros(nM, numel(metricNames));
    for m = 1:nM
        for n = 1:numel(metricNames)
            values(m, n) = results.(modelNames{m}).metrics.(metricNames{n});
        end
    end
    figure('Color','w','Position',[100 100 900 320]);
    tiledlayout(1, 3, 'TileSpacing','compact','Padding','compact');
    for n = 1:3
        nexttile;
        bar(values(:, n));
        set(gca,'XTickLabel', modelNames,'XTickLabelRotation',30);
        ylabel(metricNames{n});
        grid on;
    end
    sgtitle([titlePrefix 'Model comparison on test set']);
    exportgraphics(gcf, fullfile(outDir, ['fig_model_comparison' sfx '.pdf']), ...
                   'ContentType','vector');

    % ---- Figure 2: predicted vs real -------------------------------------
    % Behaviour adapts to N:
    %   N == 1  -> show a longer time window of the single (aggregate)
    %              series so the reader can compare model trajectories.
    %   N >  1  -> show 4 selected buses over one typical-day window.
    nTest        = numel(feats.idx.test);
    H  = feats.dims.H;

    figure('Color','w','Position',[100 100 1000 420]);

    if N == 1
        % ---- Single-series (aggregate) view ------------------------------
        % Concatenate the first forecast step (h=1) across many consecutive
        % test samples to reconstruct an approximate continuous trajectory.
        nShow = min(nTest, 96 * 3);                 % up to 3 days
        sampleRange = 1:nShow;
        sampleIdx_fu = feats.idx.test(sampleRange);

        % Truth: first step of each sample's Y
        yt_norm = squeeze(feats.Y(1, 1, sampleIdx_fu));   % [nShow, 1]
        tt      = feats.ts_y(1, sampleIdx_fu);            % [1, nShow]
        tt      = tt(:);
        y_true  = denorm_bus(yt_norm, 1, feats);

        plot(tt, y_true, 'k', 'LineWidth', 1.8, 'DisplayName','Truth');
        hold on;
        for m = 1:numel(modelNames)
            yh_norm = squeeze(results.(modelNames{m}).yhat(1, 1, sampleRange));
            yh = denorm_bus(double(yh_norm), 1, feats);
            plot(tt, yh, '--', 'LineWidth', 1.0, 'DisplayName', modelNames{m});
        end
        ylabel('System load (kW)');
        xlabel('Time');
        legend('Location','best','NumColumns',3);
        grid on;
        sgtitle([titlePrefix 'Aggregate forecast vs. measurement (1-step ahead, test set)']);
    else
        % ---- Multi-bus cluster view (existing) ---------------------------
        sampleIdx_te = max(1, round(nTest * 0.85));
        sampleIdx_fu = feats.idx.test(sampleIdx_te);
        busSel       = unique(min(N, [1, max(1, ceil(N/4)), max(1, ceil(N/2)), N]));
        tiledlayout(numel(busSel), 1, 'TileSpacing','compact');
        tt = feats.ts_y(:, sampleIdx_fu);
        for i = 1:numel(busSel)
            nexttile;
            b = busSel(i);
            y_true = denorm_bus(feats.Y(:, b, sampleIdx_fu), b, feats);
            plot(tt, y_true, 'k', 'LineWidth', 1.6, 'DisplayName','Truth');
            hold on;
            for m = 1:numel(modelNames)
                yhat = denorm_bus(double(results.(modelNames{m}).yhat(:, b, sampleIdx_te)), b, feats);
                plot(tt, yhat, '--', 'DisplayName', modelNames{m});
            end
            ylabel(sprintf('Bus %d (kW)', b));
            if i == 1; legend('Location','best','NumColumns',2); end
            grid on;
        end
        xlabel('Time');
        sgtitle([titlePrefix 'Forecast versus measurement, typical test-set day']);
    end

    exportgraphics(gcf, fullfile(outDir, ['fig_predictions_day' sfx '.pdf']), ...
                   'ContentType','vector');

    % ---- Figure 3: per-horizon MAPE for the proposed model ---------------
    if any(strcmp(modelNames,'TA_Transformer'))
        per = results.TA_Transformer.metrics.perHorizon;
        figure('Color','w','Position',[100 100 600 320]);
        plot(per.MAPE, '-o', 'LineWidth', 1.5); grid on;
        xlabel('Forecast horizon (15-min steps)');
        ylabel('MAPE (%)');
        title([titlePrefix 'TA-Transformer accuracy degradation with horizon']);
        exportgraphics(gcf, fullfile(outDir, ['fig_horizon_mape' sfx '.pdf']), ...
                       'ContentType','vector');
    end

    fprintf('[plot_results] Figures saved under %s%s\n', outDir, ...
            ifThen(isempty(sfx), '', [' (suffix ' sfx ')']));
end

% --------------------------------------------------------------------------
% Per-bus de-normalization helper: handles both scalar (legacy) and [1xN]
% vector (per-bus) normalization stats produced by build_features.m.
function v = denorm_bus(z, busIdx, feats)
    sd = feats.norm.sd_load;  mu = feats.norm.mu_load;
    if isscalar(sd); v = z * sd + mu; return; end
    v = z * sd(busIdx) + mu(busIdx);
end

function y = ifThen(c, a, b)
    if c; y = a; else; y = b; end
end
