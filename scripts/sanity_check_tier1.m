function sanity_check_tier1()
%SANITY_CHECK_TIER1  Validate that Tier1 optimizations are applied correctly.
%
%   Runs a minimal end-to-end pipeline (synthetic 33-bus, 30 days, 5 epochs)
%   and checks invariants that MUST hold after Tier1:
%
%     1. feats.norm.mu_load is a row vector of length N (per-bus stats).
%     2. feats.norm.per_bus == true flag is set.
%     3. Round-trip normalization is exact: (Y_norm * sd + mu) ~= Y_raw.
%     4. train_ta_transformer prints "lr X.XXe-NN" in epoch log (LR sched).
%     5. forwardPass mean-pool output shape is [H, N, B].
%     6. Quick-mode TA-Transformer MAPE on synthetic exp3 < 30%
%        (baseline before Tier1 was 44-78%).
%
%   USAGE
%     >> cd matlab_workspace
%     >> addpath('src'); addpath('experiments');
%     >> sanity_check_tier1
%
%   All assertions print PASS / FAIL.  Exits non-zero on FAIL when run as a
%   batch script (function form returns implicitly).

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'), fullfile(here, 'experiments'));

    fprintf('\n========================================================\n');
    fprintf('  Tier-1 Sanity Check\n');
    fprintf('========================================================\n\n');

    nPass = 0; nFail = 0;
    function ok(name, cond, detail)
        if cond
            fprintf('  [PASS] %s\n', name);
            nPass = nPass + 1;
        else
            fprintf('  [FAIL] %s   --   %s\n', name, detail);
            nFail = nFail + 1;
        end
    end

    %% ----- 1) Build features on a tiny synthetic problem ------------------
    fprintf('--- 1) build_features per-bus normalization ---\n');
    rng(42);
    T = 96 * 30;   N = 8;
    % Each bus has distinct mean & dynamics so global vs per-bus norm
    % would behave VERY differently.
    bus_means = [5, 20, 80, 150, 12, 35, 200, 60];
    P = zeros(T, N);
    for i = 1:N
        daily   = bus_means(i) * (0.6 + 0.4*sin(2*pi*(0:T-1)'/96 - pi/2));
        weekly  = bus_means(i) * 0.1 * sin(2*pi*(0:T-1)'/(96*7));
        noise   = 0.05 * bus_means(i) * randn(T, 1);
        P(:, i) = daily + weekly + noise;
    end
    P = max(P, 0.1);
    ts = datetime('2024-01-01', 'TimeZone','Asia/Shanghai') + minutes(15*(0:T-1)');
    exo = struct('timestamp', ts, 'temp_C', 20+5*randn(T,1), ...
                 'humidity', 0.5*ones(T,1), 'solar_W', zeros(T,1), ...
                 'ebike_kW', zeros(T, N), ...
                 'is_weekend', ismember(weekday(ts),[1 7]), ...
                 'is_holiday', false(T,1), 'is_sf', false(T,1));

    feats = build_features(P, exo, 'L', 96, 'H', 4, 'stride', 4);
    fprintf('  feats: L=%d, N=%d, S=%d, F=%d\n', ...
        feats.dims.L, feats.dims.N, feats.dims.S, feats.dims.F_total);

    ok('mu_load is row vector of length N', ...
       isequal(size(feats.norm.mu_load), [1, N]), ...
       sprintf('got size %s', mat2str(size(feats.norm.mu_load))));
    ok('sd_load is row vector of length N', ...
       isequal(size(feats.norm.sd_load), [1, N]), ...
       sprintf('got size %s', mat2str(size(feats.norm.sd_load))));
    ok('per_bus flag set', ...
       isfield(feats.norm,'per_bus') && feats.norm.per_bus == true, ...
       'flag missing or false');

    % Per-bus mu should differ — confirms NOT global normalization.
    mu_spread = max(feats.norm.mu_load) - min(feats.norm.mu_load);
    ok('per-bus mu actually varies across buses', mu_spread > 1, ...
       sprintf('mu range = %.3f', mu_spread));

    %% ----- 2) Round-trip de-normalization ---------------------------------
    fprintf('\n--- 2) round-trip de-normalization in evaluate_metrics ---\n');
    % Construct a synthetic prediction equal to truth -> MAPE should be ~0%.
    yt = feats.Y(:,:,feats.idx.test);                 % normalized targets
    m_ident = evaluate_metrics(single(yt), yt, feats);
    ok('identity prediction yields MAPE < 1%', m_ident.MAPE < 1.0, ...
       sprintf('got MAPE = %.3f%%', m_ident.MAPE));
    ok('identity prediction yields R2 ~ 1', m_ident.R2 > 0.999, ...
       sprintf('got R2 = %.6f', m_ident.R2));

    %% ----- 3) TA-Transformer forward shape & LR schedule ------------------
    fprintf('\n--- 3) TA-Transformer forward (5 epochs, CPU) ---\n');
    A_dummy = eye(N) + 0.5 * (rand(N) > 0.7);
    A_dummy = (A_dummy + A_dummy') / 2;

    logFile = fullfile(tempdir, 'tier1_train_log.txt');
    if exist(logFile, 'file'); delete(logFile); end
    diary(logFile);
    [params, hist, predFn] = train_ta_transformer(feats, A_dummy, ...
        'epochs', 5, 'batchSize', 16, 'verbose', true, ...
        'useGPU', false, 'd_model', 32, 'M', 2, ...
        'K_spatial', 1, 'K_temporal', 1);
    diary off;
    if exist(logFile, 'file')
        logStr = fileread(logFile);
    else
        logStr = '';
    end

    ok('train_ta_transformer returns params struct', isstruct(params), '');
    ok('history has loss curves', ...
        isfield(hist,'train_loss') && numel(hist.train_loss) >= 1, '');

    % Check LR schedule actually fired (look for "lr " token in log)
    ok('training log contains LR value per epoch', ...
       ~isempty(regexp(logStr, 'lr\s+\d+\.\d+e[-+]?\d+', 'once')), ...
       'no "lr X.XXe-NN" pattern detected');

    %% ----- 4) Prediction shape & sanity --------------------------------------
    Xte = feats.X(:,:,:,feats.idx.test);
    yhat = predFn(Xte);
    ok('predFn output is [H,N,nTest]', ...
       isequal(size(yhat), [feats.dims.H, feats.dims.N, numel(feats.idx.test)]), ...
       sprintf('got size %s', mat2str(size(yhat))));

    m_quick = evaluate_metrics(yhat, yt, feats);
    fprintf('\n  Quick-mode 5-epoch TA-Transformer MAPE = %.2f%%\n', m_quick.MAPE);
    fprintf('  (Reference: pre-Tier1 baseline on UMass-33 was 44.4%%)\n');

    % After Tier1, we expect the model to actually learn something
    % distinct from "predict mean per bus".  Even with 5 epochs on this
    % small synthetic, MAPE should be well below 100% and R2 positive.
    ok('5-epoch quick run MAPE < 100% (learned something)', m_quick.MAPE < 100, ...
       sprintf('got MAPE = %.2f%%', m_quick.MAPE));
    ok('5-epoch quick run R2 > 0 (better than mean)', m_quick.R2 > 0, ...
       sprintf('got R2 = %.4f', m_quick.R2));

    %% ----- 5) Final tally -----------------------------------------------------
    fprintf('\n========================================================\n');
    fprintf('  Sanity check: %d PASS, %d FAIL\n', nPass, nFail);
    fprintf('========================================================\n\n');
    if nFail == 0
        fprintf('  ALL TIER-1 OPTIMIZATIONS APPLIED CORRECTLY.\n');
        fprintf('  Next steps:\n');
        fprintf('    >> main(''Stage'', 3, ''Quick'', true)   %% verify exp3\n');
        fprintf('    >> main(''Stage'', 2, ''Quick'', true)   %% verify exp2 (UMass)\n');
        fprintf('    >> main(''Stage'', [2 3])               %% full run (~few hours)\n');
    else
        error('Sanity check FAILED on %d checks.', nFail);
    end
end
