function [net, history, yhat] = train_lstm(feats, opts)
%TRAIN_LSTM  Per-bus shared-weight LSTM baseline.
%
%   Input layout: model treats time as sequence dimension and (node, feature)
%   as channel dimension.  Weights are shared across nodes (same network sees
%   each node's series), which corresponds to the standard practice in
%   single-area STLF benchmarks.

    arguments
        feats (1,1) struct
        opts.hidden    (1,1) double = 128
        opts.layers    (1,1) double = 2
        opts.dropout   (1,1) double = 0.20
        opts.epochs    (1,1) double = 80          % 20 -> 80 (Tier1-2)
        opts.batch     (1,1) double = 128        % 64 -> 128 (GPU utilization; halved from 256 after OOM)
        opts.lr        (1,1) double = 1e-3
        opts.l2        (1,1) double = 1e-4
        opts.patience  (1,1) double = 8           % 3 -> 8 (more patience)
        opts.seed      (1,1) double = 42
        opts.useGPU    (1,1) logical = true
        opts.verbose   (1,1) logical = true
    end

    rng(opts.seed);

    L = feats.dims.L;   H = feats.dims.H;
    N = feats.dims.N;   F = feats.dims.F_total;

    % We reshape so that each (sample, bus) becomes a separate sequence:
    %   X_seq:  [F, L, S*N]
    %   Y_seq:  [H, S*N]
    function [Xs, Ys] = reshape_seq(idx)
        nS = numel(idx);
        Xs = permute(feats.X(:,:,:,idx), [3 1 2 4]);     % [F, L, N, S]
        Xs = reshape(Xs, [F, L, N*nS]);
        Ys = permute(feats.Y(:,:,idx), [2 1 3]);          % [N, H, S]
        Ys = reshape(Ys, [N*nS, H]);
    end

    [Xtr, Ytr] = reshape_seq(feats.idx.train);
    [Xva, Yva] = reshape_seq(feats.idx.val);
    [Xte, ~ ]  = reshape_seq(feats.idx.test);

    XtrC = squeeze(num2cell(Xtr, [1 2]));    % cell array of [F x L]
    XvaC = squeeze(num2cell(Xva, [1 2]));
    XteC = squeeze(num2cell(Xte, [1 2]));

    YtrC = num2cell(Ytr, 2);                 % cell of [1 x H]
    YvaC = num2cell(Yva, 2);

    layers = [
        sequenceInputLayer(F, 'Name', 'in')
    ];
    for k = 1:opts.layers
        if k < opts.layers
            layers = [layers; lstmLayer(opts.hidden, 'OutputMode','sequence', ...
                                         'Name', sprintf('lstm%d',k))]; %#ok<AGROW>
            layers = [layers; dropoutLayer(opts.dropout, 'Name', sprintf('do%d',k))]; %#ok<AGROW>
        else
            layers = [layers; lstmLayer(opts.hidden, 'OutputMode','last', ...
                                         'Name', sprintf('lstm%d',k))]; %#ok<AGROW>
        end
    end
    layers = [layers
              fullyConnectedLayer(H, 'Name', 'fc')
              regressionLayer('Name', 'reg')];

    executionEnv = 'auto';
    if opts.useGPU && canUseGPU(); executionEnv = 'gpu'; else; executionEnv = 'cpu'; end
    % Loud diagnostic: confirm device + batch the user is actually getting
    if strcmp(executionEnv, 'gpu')
        try
            gd = gpuDevice;
            fprintf('[train_lstm] device=GPU | %s | %.1f/%.1f GB free | batch=%d\n', ...
                    gd.Name, gd.AvailableMemory/1e9, gd.TotalMemory/1e9, opts.batch);
        catch
        end
    else
        fprintf('[train_lstm] device=CPU | batch=%d  (GPU unavailable)\n', opts.batch);
    end

    trOpts = trainingOptions('adam', ...
        'InitialLearnRate', opts.lr, ...
        'MaxEpochs',        opts.epochs, ...
        'MiniBatchSize',    opts.batch, ...
        'Shuffle',          'every-epoch', ...
        'ExecutionEnvironment', executionEnv, ...
        'ValidationData',   {XvaC, cell2mat(YvaC)}, ...
        'ValidationFrequency', max(1, floor(numel(XtrC)/opts.batch/4)), ...
        'ValidationPatience',  opts.patience, ...
        'OutputNetwork',       'best-validation-loss', ...
        'L2Regularization',    opts.l2, ...
        'GradientThreshold',   0.5, ...
        'GradientThresholdMethod', 'global-l2norm', ...
        'LearnRateSchedule',   'piecewise', ...
        'LearnRateDropPeriod', max(10, floor(opts.epochs/4)), ...
        'LearnRateDropFactor', 0.5, ...
        'Verbose',          opts.verbose, ...
        'Plots',            'none');

    [net, info] = trainNetwork(XtrC, cell2mat(YtrC), layers, trOpts);
    history = info;

    % predict test
    YpredFlat = predict(net, XteC, 'MiniBatchSize', opts.batch, ...
                        'ExecutionEnvironment', executionEnv);   % [S*N, H]
    nTe = numel(feats.idx.test);
    yhat = reshape(YpredFlat', [H, N, nTe]);
    yhat = single(yhat);
end
