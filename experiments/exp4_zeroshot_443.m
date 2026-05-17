function exp4_zeroshot_443(quick)
%EXP4_ZEROSHOT_443  Zero-shot transfer to UMass microgrid (443 households).
%
%   Loads the 443 single-day household CSVs from
%   data/03_UMass_SMART/extracted/microgrid/ and projects the TA-Transformer
%   trained on UMass-33 onto them.  Reports the in-vs-out-of-distribution
%   degradation in MAPE.
%
%   NOTE: this is a SANITY-CHECK experiment.  The microgrid dataset is only
%   24 hours long, so we evaluate one-step-ahead prediction rather than the
%   4-step horizon used elsewhere.

    if nargin < 1; quick = false; end

    here     = fileparts(mfilename('fullpath'));
    dataRoot = fullfile(here, '..', '..', 'data');
    mgDir    = fullfile(dataRoot, '03_UMass_SMART', 'extracted', 'microgrid');
    outDir   = fullfile(here, '..', 'results', 'tables');
    if ~isfolder(outDir); mkdir(outDir); end

    flist = dir(fullfile(mgDir, '2011-04-02-b*.csv'));
    assert(~isempty(flist), 'Microgrid CSVs not found under %s', mgDir);
    if quick; flist = flist(1:50); end

    % We load each house, resample to 15-min mean, store column-wise
    N_h = numel(flist);
    fprintf('  loading %d microgrid households...\n', N_h);

    common_ts = [];
    P = [];
    for k = 1:N_h
        T  = readtable(fullfile(flist(k).folder, flist(k).name), ...
                         'ReadVariableNames', false);
        T.Properties.VariableNames = {'ts_unix', 'power_kW'};
        ts = datetime(T.ts_unix, 'ConvertFrom', 'posixtime', ...
                                 'TimeZone',     'America/New_York');
        TT = retime(table2timetable(table(ts, T.power_kW, 'VariableNames', ...
                                          {'ts','p'})), 'regular', 'mean', ...
                    'TimeStep', minutes(15));
        if isempty(common_ts)
            common_ts = TT.ts;
            P = zeros(numel(common_ts), N_h);
        end
        if numel(TT.p) == numel(common_ts)
            P(:, k) = fillmissing(TT.p, 'linear');
        end
    end
    common_ts.TimeZone = 'Asia/Shanghai';

    % For zero-shot test we need to crop/pad to 33 buses by aggregation
    % (sum-aggregate 443 -> 33 buckets, fixed deterministic mapping)
    rng(42);
    bucket = mod(0:N_h-1, 33) + 1;            % round-robin assignment
    P33 = zeros(size(P,1), 33);
    for b = 1:33
        P33(:, b) = sum(P(:, bucket == b), 2);
    end

    % Build exo with calendar only (no e-bike, no weather for this day)
    exo.timestamp  = common_ts;
    exo.temp_C     = zeros(size(common_ts));
    exo.humidity   = zeros(size(common_ts));
    exo.solar_W    = zeros(size(common_ts));
    exo.ebike_kW   = zeros(size(P33));
    exo.is_weekend = ismember(weekday(common_ts), [1 7]);
    exo.is_holiday = false(size(common_ts));
    exo.is_sf      = false(size(common_ts));

    feats = build_features(P33, exo, 'L', 8, 'H', 2, 'stride', 1, ...
                            'minSamples', 30);

    % --- True zero-shot only if dimensions match -----------------------------
    modelFile = fullfile(outDir, 'exp2_model.mat');
    can_zero_shot = false;
    if isfile(modelFile)
        S = load(modelFile);
        m = S.stage2_model;
        if m.F_total == feats.dims.F_total ...
                && m.L == feats.dims.L && m.H == feats.dims.H ...
                && size(m.params.in_proj, 2) == 64   % d_model must match
            can_zero_shot = true;
            fprintf('  zero-shot: Stage 2 model dims match (L=%d, H=%d, F=%d) -> apply weights without retraining.\n', ...
                    m.L, m.H, m.F_total);
        else
            fprintf(['  Stage 2 model has L=%d H=%d F=%d but Stage 4 needs ', ...
                     'L=%d H=%d F=%d ; falling back to fresh training.\n'], ...
                    m.L, m.H, m.F_total, feats.dims.L, feats.dims.H, feats.dims.F_total);
        end
    else
        fprintf('  exp2_model.mat not found; training on microgrid as a fresh-trained baseline.\n');
    end

    if can_zero_shot
        Xtest = feats.X(:, :, :, feats.idx.test);
        Xtest = dlarray(single(Xtest));
        if canUseGPU(); Xtest = gpuArray(Xtest); end
        yh = predict_external(m.params, Xtest, m.A_norm, m.beta, ...
                              m.K_spatial, m.K_temporal);
    else
        topo = load_topology('case33bw');
        [~, ~, predFn] = train_ta_transformer(feats, topo.adj_weighted, ...
            'epochs', ifThen(quick,2,10), 'beta', 1.0);
        yh = predFn(feats.X(:,:,:,feats.idx.test));
    end
    yt = feats.Y(:,:,feats.idx.test);
    m  = evaluate_metrics(yh, yt, feats);

    fprintf('  Zero-shot 443->33  MAPE = %.2f%%,  PH_MAPE = %.2f%%,  RMSE = %.2f\n', ...
            m.MAPE, m.PH_MAPE, m.RMSE);

    writetable(struct2table(rmfield(m, {'perHorizon','perBus','horizons'})), ...
               fullfile(outDir,'exp4_zeroshot.csv'));
    fprintf('  Stage 4 done.\n');
end

function y = ifThen(c,a,b); if c; y=a; else; y=b; end; end
