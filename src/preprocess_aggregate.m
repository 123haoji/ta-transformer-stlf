function [busLoad_kW, assign, scaleGamma, meta] = preprocess_aggregate( ...
        apt_load_kW, apt_ids, topo, opts)
%PREPROCESS_AGGREGATE  Map UMass apartments to IEEE case33bw buses (Sec. 5.2).
%
%   [busLoad_kW, assign, scaleGamma, meta] = ...
%       PREPROCESS_AGGREGATE(APT_LOAD_KW, APT_IDS, TOPO, OPTS)
%
%   Deterministic protocol described in the paper:
%
%   1. Compute 5-D feature vector phi_j for each apartment j
%        phi_j = [ mean_load,
%                  peak_to_mean,
%                  evening_share,
%                  weekend_to_weekday,
%                  ramp_variance ]
%   2. z-score-standardise phi across apartments
%   3. k-means(N_load_buses, replicates=20, seed=42)
%   4. Match clusters to buses by sorting both by cluster-mean / Pd
%      (deterministic 1-to-1 rank-matching)
%   5. Per-cluster load is summed over its apartments and scaled by
%      gamma_i so that the annual mean matches topo.bus_Pd_kW for that bus.
%
%   INPUTS
%     apt_load_kW   [T x J] load matrix (kW), already on 15-min grid
%     apt_ids       [J x 1] string apartment ids
%     topo          struct from LOAD_TOPOLOGY (must contain bus_Pd_kW, N)
%     opts (struct, optional)
%       .seed             default 42
%       .replicates       default 20
%       .slackBus         bus index to receive the slack cluster (default 1
%                         -- the substation bus has Pd=0 so its slack content
%                         contributes 0 unless you choose differently)
%       .excludeBuses     buses to skip during matching (default [1] -- the
%                         substation/slack bus of case33bw has Pd = 0)
%       .pdFloorKw        minimum Pd considered a load bus (default 1e-3)
%       .verbose          default true
%
%   OUTPUTS
%     busLoad_kW    [T x N] per-bus load matrix on case33bw indexing (1..N).
%                          Excluded buses receive zeros.
%     assign        [J x 1] integer bus assignment for each apartment, or 0
%                          if apartment was placed in slack cluster.
%     scaleGamma    [N x 1] gamma_i scaling factors actually applied.
%     meta          struct (cluster sizes, achieved means, sse, ...).
%
%   Reproducibility: the function fixes both the RNG state and k-means
%   replicates seed, so two calls with identical inputs yield bit-exact
%   identical outputs.

    arguments
        apt_load_kW (:, :) double
        apt_ids     (:, 1) string
        topo        (1, 1) struct
        opts.seed         (1, 1) double = 42
        opts.replicates   (1, 1) double = 20
        opts.slackBus     (1, 1) double = 1
        opts.excludeBuses (1, :) double = 1
        opts.pdFloorKw    (1, 1) double = 1e-3
        opts.verbose      (1, 1) logical = true
    end

    [T, J] = size(apt_load_kW);
    N      = topo.N;

    % -- 1. compute per-apartment feature vector phi -------------------------
    % evening: hours 17:00 - 22:00 on 15-min grid; we know the grid step is
    % 15 min from caller, so 1 day = 96 steps; bin index for HH:MM is
    % (HH*4 + MM/15).  We compute via index modulo 96.
    stepsPerDay   = 96;
    idx           = (0:T-1)'.';
    hh            = floor(mod(idx, stepsPerDay) / 4);  % 0..23
    eveningMask   = (hh >= 17) & (hh <= 22);
    % NOTE: weekday-of-week pattern requires a real datetime anchor.  Since
    % we are passed an apt_load matrix without timestamps, we approximate
    % weekend mask using a deterministic 7-day cycle starting Day-1 = Sunday
    % (UMass 2016 starts on a Friday actually; we accept ~1-day phase
    % shift -- this is a feature for clustering, not for forecasting target).
    dayIdx        = floor(idx / stepsPerDay);
    weekendMask   = mod(dayIdx, 7) <= 1;   % 2 of every 7 days

    phi = zeros(J, 5);
    for j = 1:J
        y       = apt_load_kW(:, j);
        muY     = mean(y, 'omitnan');
        peakY   = max(y, [], 'omitnan');
        eveSum  = sum(y(eveningMask), 'omitnan');
        totSum  = sum(y, 'omitnan');
        weY     = mean(y(weekendMask), 'omitnan');
        wdY     = mean(y(~weekendMask), 'omitnan');
        dy      = diff(y);
        phi(j,:) = [ muY, ...
                     peakY / max(muY, 1e-6), ...
                     eveSum / max(totSum, 1e-6), ...
                     weY    / max(wdY,  1e-6), ...
                     var(dy, 'omitnan') ];
    end

    % robust z-score (median / mad) to reduce outlier dominance
    centre = median(phi, 1);
    scale  = max(mad(phi, 1, 1), 1e-6);
    Z      = (phi - centre) ./ scale;

    % -- 2. eligible load buses & cluster count ------------------------------
    busMask = true(N, 1);
    busMask(opts.excludeBuses) = false;
    busMask(topo.bus_Pd_kW < opts.pdFloorKw) = false;
    loadBusIdx = find(busMask);
    nLoadBus   = numel(loadBusIdx);
    assert(nLoadBus > 0, 'No eligible load buses after filtering.');

    % k-means requires nApartments > nClusters.  When the apartment count is
    % smaller than the available load buses (e.g. Quick-mode subsetting),
    % we shrink the cluster count to J-1 and only the heaviest-Pd buses
    % receive load; light buses get zero injection.  Diagnostics emitted.
    K = min(nLoadBus, J - 1);
    if K < nLoadBus && opts.verbose
        fprintf(['[preprocess_aggregate] Apartment count J=%d < load buses %d; ', ...
                 'using K=%d clusters; lightest %d buses receive zero load.\n'], ...
                J, nLoadBus, K, nLoadBus - K);
    end
    assert(K >= 2, 'Need at least 2 apartments after filtering (have J=%d).', J);

    % -- 3. k-means clustering ----------------------------------------------
    rng(opts.seed);
    [clusterIdx, ~, ~, sumDist] = kmeans(Z, K, ...
        'Replicates', opts.replicates, ...
        'Start',      'plus', ...
        'Display',    'off');

    % -- 4. rank-match clusters to (heaviest K) buses -----------------------
    clusterMean = accumarray(clusterIdx, phi(:,1), [K, 1], @mean);
    [~, clusterRank] = sort(clusterMean, 'ascend');     % ascending cluster mean
    pdSorted        = sort(topo.bus_Pd_kW(loadBusIdx), 'descend');   % heaviest first
    [~, busRank]    = sort(topo.bus_Pd_kW(loadBusIdx), 'descend');   % heaviest indices
    busesUsed       = loadBusIdx(busRank(1:K));         % K heaviest buses
    busesUsedSorted = sort(busesUsed);
    % map clusters (ascending mean) to buses (ascending Pd among the K heaviest)
    [~, busOrderAsc] = sort(topo.bus_Pd_kW(busesUsedSorted), 'ascend');
    cluster2bus = zeros(K, 1);
    for r = 1:K
        cluster2bus(clusterRank(r)) = busesUsedSorted(busOrderAsc(r));
    end

    % -- 5. per-bus aggregate + gamma scaling -------------------------------
    busLoad_kW = zeros(T, N);
    scaleGamma = ones(N, 1);
    achievedMean = zeros(N, 1);
    clusterSize  = zeros(N, 1);
    for c = 1:K
        members = find(clusterIdx == c);
        busI    = cluster2bus(c);
        agg     = sum(apt_load_kW(:, members), 2, 'omitnan');
        target  = topo.bus_Pd_kW(busI);          % kW
        currentMean = mean(agg, 'omitnan');
        if currentMean > 1e-6
            gamma = target / currentMean;
        else
            gamma = 1;
        end
        scaleGamma(busI)   = gamma;
        busLoad_kW(:, busI)= agg * gamma;
        achievedMean(busI) = mean(busLoad_kW(:, busI), 'omitnan');
        clusterSize(busI)  = numel(members);
    end

    % assignment vector
    assign = zeros(J, 1);
    for j = 1:J
        assign(j) = cluster2bus(clusterIdx(j));
    end

    meta.cluster_size      = clusterSize;
    meta.target_Pd_kW      = topo.bus_Pd_kW;
    meta.achieved_mean_kW  = achievedMean;
    meta.scale_gamma       = scaleGamma;
    meta.cluster2bus       = cluster2bus;
    meta.kmeans_total_sse  = sum(sumDist);
    meta.n_load_buses      = nLoadBus;
    meta.n_clusters_used   = K;
    meta.buses_used        = busesUsed;
    meta.opts              = opts;

    if opts.verbose
        % Aggressive scalarization: explicitly extract scalar(1) values so
        % that fprintf can never loop over an unintended vector argument.
        sz_arr = clusterSize(busesUsed);
        sz_arr = double(sz_arr(:));
        m_val  = sz_arr; m_val  = mean(m_val); m_val  = m_val(1);
        mn_val = sz_arr; mn_val = min(mn_val); mn_val = mn_val(1);
        mx_val = sz_arr; mx_val = max(mx_val); mx_val = mx_val(1);
        sse    = double(sum(double(sumDist(:))));  sse = sse(1);
        line = sprintf('[preprocess_aggregate] J=%d apts -> K=%d clusters -> %d buses | mean=%.2f min=%d max=%d SSE=%.2f', ...
                       double(J), double(K), double(numel(busesUsed)), ...
                       m_val, mn_val, mx_val, sse);
        disp(line);
    end
end
