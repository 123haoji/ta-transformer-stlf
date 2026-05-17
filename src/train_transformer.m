function [model, history, yhat] = train_transformer(feats, opts)
%TRAIN_TRANSFORMER  Vanilla (non-topology-aware) Transformer baseline.
%
%   Identical architecture and hyperparameters to TA-Transformer except
%   beta = 0, so the spatial attention has no topological prior.
%   By construction, the difference between TA-Transformer and this
%   baseline isolates the contribution of the topology mask.

    arguments
        feats  (1, 1) struct
        opts.d_model     (1, 1) double = 64
        opts.M           (1, 1) double = 4
        opts.K_spatial   (1, 1) double = 2
        opts.K_temporal  (1, 1) double = 2
        opts.dropout     (1, 1) double = 0.10
        opts.batchSize   (1, 1) double = 64      % 32 -> 64 (GPU utilization; 128 OOMs on 8GB)
        opts.epochs      (1, 1) double = 25
        opts.lr          (1, 1) double = 1e-3
        opts.useGPU      (1, 1) logical = true
        opts.seed        (1, 1) double = 42
    end

    N = feats.dims.N;
    A_dummy = ones(N, N);     % uniform adjacency, beta=0 makes it inert

    [model, history, predFn] = train_ta_transformer(feats, A_dummy, ...
        'd_model', opts.d_model, 'M', opts.M, ...
        'K_spatial', opts.K_spatial, 'K_temporal', opts.K_temporal, ...
        'beta', 0.0, ...
        'dropout', opts.dropout, ...
        'batchSize', opts.batchSize, ...
        'epochs', opts.epochs, ...
        'lr', opts.lr, ...
        'useGPU', opts.useGPU, ...
        'seed', opts.seed);

    % predict on test set in same shape conventions
    Xte = feats.X(:, :, :, feats.idx.test);
    yhat = predFn(Xte);    % [H, N, nTest]
end
