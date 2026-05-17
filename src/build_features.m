function feats = build_features(P, exogenous, opts)
%BUILD_FEATURES  Assemble model input tensors from raw load + exogenous data.
%
%   feats = BUILD_FEATURES(P, EXOGENOUS, OPTS)
%
%   Produces:
%     feats.X       [L, N, F, S]  input tensor of past context per sample
%     feats.Y       [H, N, S]     target per sample
%     feats.ts_y    [H, S]        target-window starting timestamps (datetime)
%     feats.exo_y   [H, N, F_exo, S]  future exogenous (known-future portion)
%     feats.norm    struct(mu, sigma)   normalization stats from train split
%     feats.idx     struct(train, val, test) sample indices in chronological order
%     feats.opts    echoed options
%
%   INPUTS
%     P            [T x N] active power load (kW), 15-min grid
%     exogenous    struct with fields
%        .timestamp   [T x 1] datetime aligned to P rows
%        .temp_C      [T x 1] or [T x N] (will be broadcast)
%        .humidity    [T x 1]
%        .solar_W     [T x 1]
%        .ebike_kW    [T x N] (may be all zeros if unavailable)
%        .is_holiday  [T x 1] logical
%        .is_weekend  [T x 1] logical
%        .is_sf       [T x 1] logical (spring festival)
%     opts (struct)
%       .L     context window length in steps   (default 96  = 24 h)
%       .H     forecast horizon in steps         (default 4   = 1 h)
%       .stride sample stride in steps           (default 4   = 1 h)
%       .splitFrac  [train, val, test] fractions (default [0.70 0.15 0.15])
%       .normalize  'zscore' | 'minmax' (default 'zscore')
%
%   Normalization is fit on the train split only (no leakage); val and test
%   are transformed using train statistics.

    arguments
        P           (:, :) double
        exogenous   (1, 1) struct
        opts.L      (1, 1) double = 96
        opts.H      (1, 1) double = 4
        opts.stride (1, 1) double = 4
        opts.splitFrac (1, 3) double = [0.70 0.15 0.15]
        opts.normalize char {mustBeMember(opts.normalize,{'zscore','minmax'})} = 'zscore'
        opts.minSamples (1, 1) double = 50
        opts.lagFeatures (1, 1) logical = true   % NEW: Fan-Hyndman style lag features
    end

    [T, N] = size(P);
    assert(numel(exogenous.timestamp) == T, ...
        'exogenous.timestamp must align with P rows');

    L = opts.L;  H = opts.H;  s = opts.stride;

    % ---- Standard STLF lag features (Fan & Hyndman 2010, IEEE TPS) ---------
    % - lag-168h (one week back, same hour/day-of-week)  -- captures weekly
    %   seasonality not in the L=96 (24h) context window.
    % - rolling 24h mean   -- captures recent baseline level
    % - rolling 168h mean  -- captures weekly average baseline
    % These three features are the empirically validated baseline used by
    % AEMO (Australian Energy Market Operator) for half-hourly forecasting
    % (Fan & Hyndman 2010, "Short-Term Load Forecasting Based on a
    % Semi-Parametric Additive Model", IEEE Trans. Power Systems).
    LAG_1W_STEPS   = 672;     % 1 week at 15-min resolution = 7*24*4
    ROLL_24H_STEPS = 96;
    ROLL_168H_STEPS= 672;

    lag_1w   = zeros(T, N);
    roll_24  = zeros(T, N);
    roll_168 = zeros(T, N);
    if opts.lagFeatures
        for t = (LAG_1W_STEPS+1):T
            lag_1w(t, :)   = P(t - LAG_1W_STEPS, :);
        end
        for t = (ROLL_24H_STEPS+1):T
            roll_24(t, :)  = mean(P((t-ROLL_24H_STEPS):(t-1), :), 1);
        end
        for t = (ROLL_168H_STEPS+1):T
            roll_168(t, :) = mean(P((t-ROLL_168H_STEPS):(t-1), :), 1);
        end
        % For early t where lag is undefined, back-fill with first valid value
        % to avoid all-zero rows (which would distort the standardization).
        first_valid_1w  = lag_1w(LAG_1W_STEPS+1, :);
        first_valid_24  = roll_24(ROLL_24H_STEPS+1, :);
        first_valid_168 = roll_168(ROLL_168H_STEPS+1, :);
        for n = 1:N
            lag_1w(1:LAG_1W_STEPS, n)     = first_valid_1w(n);
            roll_24(1:ROLL_24H_STEPS, n)  = first_valid_24(n);
            roll_168(1:ROLL_168H_STEPS, n)= first_valid_168(n);
        end
    end

    % construct sliding-window starts
    % When lag features are enabled, samples must start at >= LAG_1W_STEPS+1
    % so that every position in the context window has a valid lag.
    if opts.lagFeatures
        sampleStart0 = LAG_1W_STEPS + 1;
    else
        sampleStart0 = 1;
    end
    sampleStarts = sampleStart0 : s : (T - L - H + 1);
    S            = numel(sampleStarts);
    assert(S > opts.minSamples, ...
        'Too few samples (%d, min %d); reduce L/H/stride or get more data.', ...
        S, opts.minSamples);

    % ---- chronological train/val/test split before fitting normalization
    cum   = cumsum(opts.splitFrac);
    nTr   = floor(cum(1) * S);
    nVa   = floor(cum(2) * S) - nTr;
    nTe   = S - nTr - nVa;
    idx.train = 1 : nTr;
    idx.val   = nTr + (1 : nVa);
    idx.test  = nTr + nVa + (1 : nTe);

    % ---- exogenous tensor preparation -------------------------------------
    exoMat = build_exogenous_matrix(exogenous, T, N);    % [T, N, F_exo]

    % Append lag features (per-bus) if enabled.  These are [T x N] each.
    if opts.lagFeatures
        exoMat = cat(3, exoMat, lag_1w, roll_24, roll_168);   % +3 channels
    end
    F_exo  = size(exoMat, 3);
    F_total = 1 + F_exo;                                  % +1 for the load itself

    % ---- build samples ----------------------------------------------------
    X = zeros(L, N, F_total, S, 'single');
    Y = zeros(H, N, S, 'single');
    exoY = zeros(H, N, F_exo, S, 'single');
    ts_y = NaT(H, S, 'TimeZone', exogenous.timestamp.TimeZone);

    for k = 1:S
        a    = sampleStarts(k);          % context start
        b    = a + L - 1;                % context end (inclusive)
        c    = b + 1;                    % target start
        d    = c + H - 1;                % target end
        X(:, :, 1,     k) = P(a:b, :);
        X(:, :, 2:end, k) = exoMat(a:b, :, :);
        Y(:, :, k)        = P(c:d, :);
        exoY(:, :, :, k)  = exoMat(c:d, :, :);
        ts_y(:, k)        = exogenous.timestamp(c:d);
    end

    % ---- normalize using train split only ---------------------------------
    if strcmp(opts.normalize, 'zscore')
        % Per-bus z-score on the load channel.  Statistics are taken across
        % time (dim 1) and samples (dim 4) but retained per node (dim 2).
        % Rationale: global statistics collapse buses with very different
        % load magnitudes (e.g. 5 kW residential vs 150 kW commercial) onto
        % a single scale, after which a shared model can only learn the
        % average dynamic and predicts a near-constant for every bus.
        % Per-bus z-score keeps each node on its own scale and lets the
        % network learn shared *normalized* dynamics. See Kim et al. (RevIN,
        % ICLR 2022) and BLformer (2025) for the broader rationale.
        blkL     = X(:,:,1,idx.train);                  % [L, N, 1, nTr]
        mu_load4 = mean(blkL, [1 4]);                   % [1, N, 1, 1]
        sd_load4 = std (blkL, 0, [1 4]);                % [1, N, 1, 1]
        sd_load4(sd_load4 < 1e-9) = 1;

        % apply on X load channel: [L,N,1,S] broadcasts with [1,N,1,1]
        X(:,:,1,:) = (X(:,:,1,:) - mu_load4) ./ sd_load4;
        % apply on Y: [H,N,S] broadcasts with [1,N,1]
        mu_Y = reshape(mu_load4, [1, N, 1]);
        sd_Y = reshape(sd_load4, [1, N, 1]);
        Y    = (Y - mu_Y) ./ sd_Y;

        % per-feature stats for exogenous (kept global; exogenous channels
        % are already comparable across nodes by construction).
        mu_exo = zeros(1, 1, F_exo);
        sd_exo = ones (1, 1, F_exo);
        for f = 1:F_exo
            blk = X(:,:,1+f,idx.train);
            mu_exo(1,1,f) = mean(blk, 'all');
            sd_exo(1,1,f) = max(std(blk, 0, 'all'), 1e-9);
        end
        X(:,:,2:end,:) = (X(:,:,2:end,:) - mu_exo) ./ sd_exo;
        exoY           = (exoY          - mu_exo) ./ sd_exo;

        % Store norm stats as [1 x N] row vectors so downstream code can
        % reshape() them as needed without ambiguity.
        feats.norm.mu_load   = reshape(mu_load4, 1, N);
        feats.norm.sd_load   = reshape(sd_load4, 1, N);
        feats.norm.mu_exo    = squeeze(mu_exo);
        feats.norm.sd_exo    = squeeze(sd_exo);
        feats.norm.per_bus   = true;     % flag for downstream consumers
    else
        % minmax: omitted in this scaffold for brevity
        error('opts.normalize=''minmax'' not implemented in scaffold.');
    end

    feats.X     = X;
    feats.Y     = Y;
    feats.exo_y = exoY;
    feats.ts_y  = ts_y;
    feats.idx   = idx;
    feats.opts  = opts;
    feats.dims  = struct('L',L,'H',H,'N',N,'F_total',F_total,'F_exo',F_exo,'S',S);
end

% ---------------------------------------------------------------------------
function exoMat = build_exogenous_matrix(exo, T, N)
    % broadcast scalar-per-time fields across N nodes; per-node fields kept
    function M = bcast(v)
        v = v(:);
        if numel(v) ~= T
            error('exogenous field length mismatch (%d vs T=%d)', numel(v), T);
        end
        M = repmat(v, 1, N);
    end
    function M = bcast_or_keep(X)
        if size(X,1) == T && size(X,2) == N
            M = X;
        else
            M = bcast(X);
        end
    end

    feats = {};
    if isfield(exo,'temp_C');     feats{end+1} = bcast(exo.temp_C);                 end
    if isfield(exo,'humidity');   feats{end+1} = bcast(exo.humidity);               end
    if isfield(exo,'solar_W');    feats{end+1} = bcast(exo.solar_W);                end
    if isfield(exo,'ebike_kW');   feats{end+1} = bcast_or_keep(exo.ebike_kW);       end
    if isfield(exo,'is_weekend'); feats{end+1} = bcast(double(exo.is_weekend));     end
    if isfield(exo,'is_holiday'); feats{end+1} = bcast(double(exo.is_holiday));     end
    if isfield(exo,'is_sf');      feats{end+1} = bcast(double(exo.is_sf));          end

    % time-of-day (sinusoid pair) -- always added
    hh = hour(exo.timestamp) + minute(exo.timestamp)/60;
    feats{end+1} = bcast(sin(2*pi*hh/24));
    feats{end+1} = bcast(cos(2*pi*hh/24));
    % day-of-week (sinusoid pair)
    dw = weekday(exo.timestamp);
    feats{end+1} = bcast(sin(2*pi*dw/7));
    feats{end+1} = bcast(cos(2*pi*dw/7));

    F_exo = numel(feats);
    exoMat = zeros(T, N, F_exo);
    for f = 1:F_exo
        exoMat(:, :, f) = feats{f};
    end
end
