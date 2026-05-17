function replot_predictions(outDir)
%REPLOT_PREDICTIONS  Regenerate Figure 1 (system-aggregate prediction-vs-truth)
%   from the saved Stage 2b results, with English datetime tick labels.
%
%   Root cause of the previous garbled "Time" axis: MATLAB's datetime axis
%   ruler inherited the system locale (zh_CN on Windows Chinese), producing
%   non-ASCII month/day labels (e.g., "1月") that lacked CJK glyphs in the
%   exported PDF's font.  We force the tick format to ASCII English here
%   and additionally pin the datetime locale to 'en_US' for safety.
%
%   Usage:
%     >> replot_predictions()

    here  = fileparts(mfilename('fullpath'));
    matIn = fullfile(here, 'results', 'tables', 'exp2b_aggregate_results.mat');

    if nargin < 1 || isempty(outDir)
        outDir = fullfile(here, 'results', 'figures');
    end
    if ~isfolder(outDir); mkdir(outDir); end
    if ~isfile(matIn)
        error('replot_predictions:noMat', ...
              'Cannot find %s.  Run exp2b_aggregate() first.', matIn);
    end

    S       = load(matIn);
    results = S.results;
    feats   = results.feats;

    modelNames = {'ARIMA','CNN_LSTM','Transformer','TA_Transformer'};   % skip LSTM (diverged)

    nTest   = numel(feats.idx.test);
    nShow   = min(nTest, 96 * 3);                  % up to 3 days
    rng     = 1:nShow;
    sIdx_fu = feats.idx.test(rng);

    % Truth (first-step head of each sample's Y)
    yt_norm = squeeze(feats.Y(1, 1, sIdx_fu));
    tt      = feats.ts_y(1, sIdx_fu);
    tt      = tt(:);
    y_true  = denorm_bus(yt_norm, 1, feats);

    % --- Robust locale fix: convert datetime to numeric x and build tick
    %     labels via datestr (always English, locale-independent). ----------
    if isdatetime(tt)
        x = datenum(tt);                          % serial date number
        % Pick ~7 tick positions: midnight of each calendar day in range.
        tStart = dateshift(tt(1),  'start', 'day');
        tEnd   = dateshift(tt(end),'end',   'day');
        tickDt = (tStart:hours(12):tEnd)';
        % Keep only tick positions inside the displayed range.
        tickDt = tickDt(tickDt >= tt(1) & tickDt <= tt(end));
        tickX  = datenum(tickDt);
        tickL  = cellstr(datestr(tickDt, 'mmm dd HH:MM'));   % 'Jan 15 12:00'
    else
        x      = tt;
        tickX  = [];
        tickL  = {};
    end

    fig = figure('Color','w','Position',[100 100 1000 420]);
    plot(x, y_true, 'k', 'LineWidth', 1.8, 'DisplayName', 'Truth');
    hold on;
    for m = 1:numel(modelNames)
        if ~isfield(results, modelNames{m}); continue; end
        yh_norm = squeeze(results.(modelNames{m}).yhat(1, 1, rng));
        yh = denorm_bus(double(yh_norm), 1, feats);
        plot(x, yh, '--', 'LineWidth', 1.0, ...
             'DisplayName', strrep(modelNames{m}, '_', '-'));
    end
    ylabel('System load (kW)');
    xlabel('Time');

    ax = gca;
    if ~isempty(tickX)
        set(ax, 'XTick', tickX, 'XTickLabel', tickL);
    end
    set(ax, 'FontName', 'Helvetica', 'FontSize', 10);
    xtickangle(ax, 25);
    xlim([x(1), x(end)]);

    legend('Location', 'best', 'NumColumns', 3, 'Box', 'on', ...
           'EdgeColor', [0.7 0.7 0.7]);
    grid on;
    title('System-aggregate forecast on three representative test-set days');

    outPath = fullfile(outDir, 'fig_aggregate_predictions.pdf');
    exportgraphics(fig, outPath, 'ContentType', 'vector');
    fprintf('  Wrote %s\n', outPath);
    fprintf('  replot_predictions done.\n');
end

% ============================================================================
function v = denorm_bus(v_norm, b, feats)
% Mirror of plot_results.m's denorm_bus (per-bus z-score inverse).
    if isfield(feats, 'norm') && isfield(feats.norm, 'mu_y') ...
            && isfield(feats.norm, 'sd_y')
        mu = feats.norm.mu_y(b);
        sd = feats.norm.sd_y(b);
        v  = v_norm .* sd + mu;
    else
        v = v_norm;
    end
end
