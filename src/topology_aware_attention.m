function Y = topology_aware_attention(Q, K, V, A, beta)
%TOPOLOGY_AWARE_ATTENTION  Single- or multi-head attention with topology bias.
%
%   Y = TOPOLOGY_AWARE_ATTENTION(Q, K, V, A, BETA)
%
%   Implements Equation (5) of the paper:
%       alpha_{ij} = softmax_j(  Q_i K_j^T / sqrt(d)  +  beta * log(A_ij)  )
%       Y_i        = sum_j alpha_{ij} V_j
%
%   INPUT TENSOR LAYOUT
%     Q, K, V   any of the following shapes are accepted:
%                 [N, d, B]            -- single-head, B batched pages
%                 [N, d_h, M, B]       -- multi-head with M heads
%     A         [N, N]  non-negative adjacency (0 = no edge)
%     beta      scalar >= 0;  beta = 0 yields vanilla self-attention
%
%   OUTPUT shape matches input shape of V (without changing N).
%
%   NUMERICAL NOTES
%   * log(0) is treated as -1e9 so exp(.) underflows cleanly.
%   * Stable softmax: subtract per-row max before exp.
%   * Works on plain numeric arrays AND dlarrays (stays in AD graph).

    if nargin < 5; beta = 1.0; end

    sz = size(Q);
    N  = sz(1);
    d  = sz(2);

    % flatten any trailing dims (M, B, ...) into one page axis
    pageDims = sz(3:end);
    if isempty(pageDims); pageDims = 1; end
    P = prod(pageDims);

    Qf = reshape(Q, [N, d, P]);
    Kf = reshape(K, [N, d, P]);
    Vf = reshape(V, [N, d, P]);

    scores = pagemtimes(Qf, 'none', Kf, 'transpose') / sqrt(d);  % [N, N, P]

    if beta > 0
        % AD-safe construction of the topology bias (no indexed assignment).
        connMask   = double(A > 0);                 % [N,N] 1 if edge
        A_clamped  = max(A, 1e-30);                 % avoid log(0)
        logA_full  = log(A_clamped);                % real values
        disconnPen = -1e9 * (1 - connMask);         % -1e9 if no edge, 0 otherwise
        bias       = beta * (logA_full .* connMask + disconnPen);
        scores     = scores + bias;                 % broadcasts over P
    end

    % numerically stable softmax over dim 2 (j-node)
    s_max = max(scores, [], 2);
    s_exp = exp(scores - s_max);
    alpha = s_exp ./ sum(s_exp, 2);

    Yf = pagemtimes(alpha, Vf);                     % [N, d, P]
    Y  = reshape(Yf, [N, d, pageDims]);
end
