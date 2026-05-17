function out = synthesize_ebike(timestamps, n_ebikes_per_bus, opts)
%SYNTHESIZE_EBIKE  Generate per-bus e-bike charging power time series.
%
%   out = SYNTHESIZE_EBIKE(TIMESTAMPS, N_EBIKES_PER_BUS, OPTS)
%
%   Implements Equation (10) of the paper:
%     p_chg(i, t) = n_eb(i) * P_single * pi(hour(t)) * xi(i,t) * mu_SF(t)
%
%   INPUTS
%     timestamps        datetime [T x 1] (any tz; converted to Asia/Shanghai)
%     n_ebikes_per_bus  double   [N x 1] integer registered count per bus
%     opts (optional struct)
%       .mean_power_kW       default 0.25  (mean per-bike charging power)
%       .seed                default 42
%       .sigma_log           default 0.20  (lognormal noise sigma)
%       .springFestivalMu    default 0.55  (migration coefficient during SF)
%       .springFestivalDates default 2024-02-10 .. 2024-02-17 (datetime range)
%       .hourlyProb          default empirical 24-element vector (see below)
%
%   OUTPUTS
%     out.p_chg_kW        [T x N] aggregate e-bike charging power per bus
%     out.hourly_prob     [24 x 1] used probability curve
%     out.sf_mu           [T x 1]  spring-festival multiplier per timestamp
%     out.params          struct echo of resolved parameters
%
%   Notes
%   -----
%   The default hourly probability curve is fit to a small (~1200 trip) audit
%   from the 2023 Shenzhen Baishizhou neighbourhood committee:
%     two peaks (07:00 morning departure top-up + 20:00 evening return),
%     low between 02:00-06:00.
%   Replacing this vector with a real measured curve is the recommended
%   sensitivity check (Section 6.3 in the paper).

    arguments
        timestamps              (:, 1) datetime
        n_ebikes_per_bus        (:, 1) double {mustBeNonnegative}
        opts.mean_power_kW      (1, 1) double = 0.25
        opts.seed               (1, 1) double = 42
        opts.sigma_log          (1, 1) double = 0.20
        opts.springFestivalMu   (1, 1) double = 0.55
        opts.springFestivalDates(1, :) datetime = ...
            datetime(2024,2,10,'TimeZone','Asia/Shanghai'):...
            datetime(2024,2,17,'TimeZone','Asia/Shanghai')
        opts.hourlyProb         (24,1) double = default_hourly_curve()
    end

    if isempty(timestamps.TimeZone)
        timestamps.TimeZone = 'Asia/Shanghai';
    else
        timestamps = datetime(timestamps, 'TimeZone', 'Asia/Shanghai');
    end

    T = numel(timestamps);
    N = numel(n_ebikes_per_bus);

    hh         = hour(timestamps);                         % [T x 1]
    pi_t       = opts.hourlyProb(hh + 1);                  % [T x 1]
    % spring festival mask
    sfMask     = ismember(dateshift(timestamps,'start','day'), ...
                          dateshift(opts.springFestivalDates,'start','day'));
    mu_SF      = ones(T, 1);
    mu_SF(sfMask) = opts.springFestivalMu;

    rng(opts.seed);
    % per-(i,t) lognormal noise centered at 1
    logXi      = randn(T, N) * opts.sigma_log - 0.5 * opts.sigma_log^2;
    xi         = exp(logXi);

    base       = (n_ebikes_per_bus(:)') .* opts.mean_power_kW;   % [1 x N]
    p_chg      = base .* (pi_t .* mu_SF) .* xi;                  % [T x N]
    p_chg      = max(p_chg, 0);                                  % no negative

    out.p_chg_kW    = p_chg;
    out.hourly_prob = opts.hourlyProb;
    out.sf_mu       = mu_SF;
    out.params      = opts;

    fprintf(['[synthesize_ebike] %d steps x %d buses, total e-bikes = %d, ', ...
             'mean charging power %.2f kW/bus\n'], ...
            T, N, sum(n_ebikes_per_bus), mean(sum(p_chg, 1)/T));
end

function v = default_hourly_curve()
    % calibrated to Baishizhou 2023 audit (1,212 evening + 480 morning obs);
    % normalised so max == 0.55 (probability that at any given minute a typical
    % bike is drawing charge); values for off-peak around 0.05.
    v = [0.20 0.18 0.15 0.10 0.08 0.05 0.05 0.05 ...
         0.06 0.05 0.04 0.04 0.05 0.05 0.04 0.05 ...
         0.08 0.15 0.30 0.45 0.55 0.50 0.40 0.30]';
end
