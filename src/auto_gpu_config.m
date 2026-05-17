function cfg = auto_gpu_config(verbose)
%AUTO_GPU_CONFIG  Detect GPU and recommend optimal batch sizes.
%
%   cfg = AUTO_GPU_CONFIG(verbose)
%
%   Returns a struct with fields:
%     .useGPU         logical, true if GPU is available and selected
%     .device         'gpu' or 'cpu'
%     .deviceName     string (e.g. 'NVIDIA GeForce RTX 4070 Laptop GPU')
%     .vram_GB        total VRAM in GB (NaN if no GPU)
%     .avail_GB       currently available VRAM in GB (NaN if no GPU)
%     .computeCap     compute capability string (e.g. '8.9')
%     .train_batch    recommended TA-Transformer training batch size
%     .lstm_batch     recommended LSTM mini-batch size
%     .predict_batch  recommended inference batch size
%
%   Sizing logic (based on TA-Transformer temporal-attention tensor
%   [L=96, L=96, N=33, B] and 2 spatial + 2 temporal blocks with backprop
%   graph; rough peak VRAM scales as 0.025 * B GB):
%
%       avail_GB        train_batch   lstm_batch   predict_batch
%       < 3 GB              32            64            32
%       3 - 6 GB            64           128            64
%       6 - 10 GB          128           256           128
%       > 10 GB            256           512           256
%
%   If no GPU is found the function returns CPU defaults (small batches).

    if nargin < 1; verbose = true; end

    cfg.useGPU      = false;
    cfg.device      = 'cpu';
    cfg.deviceName  = '';
    cfg.vram_GB     = NaN;
    cfg.avail_GB    = NaN;
    cfg.computeCap  = '';
    % CPU defaults (safe small)
    cfg.train_batch   = 32;
    cfg.lstm_batch    = 64;
    cfg.predict_batch = 32;

    try
        n = gpuDeviceCount;
    catch
        n = 0;
    end
    if n == 0
        if verbose
            fprintf('[auto_gpu_config] No CUDA GPU detected. Using CPU defaults.\n');
        end
        return;
    end

    try
        g = gpuDevice;
    catch ME
        if verbose
            fprintf(2, '[auto_gpu_config] gpuDevice() failed: %s. Using CPU.\n', ...
                    ME.message);
        end
        return;
    end

    cfg.useGPU      = true;
    cfg.device      = 'gpu';
    cfg.deviceName  = char(g.Name);
    cfg.vram_GB     = g.TotalMemory / 1e9;
    cfg.avail_GB    = g.AvailableMemory / 1e9;
    cfg.computeCap  = char(g.ComputeCapability);

    % VRAM-aware batch sizing.  After empirical OOM on RTX 4070 Laptop (8 GB)
    % with batch=128 the per-sample peak was revised up to ~60 MB; these
    % thresholds are now conservative for FP32 autograd training.
    av = cfg.avail_GB;
    if av < 3
        cfg.train_batch = 16;  cfg.lstm_batch = 32;  cfg.predict_batch = 16;
    elseif av < 6
        cfg.train_batch = 32;  cfg.lstm_batch = 64;  cfg.predict_batch = 32;
    elseif av < 10
        cfg.train_batch = 64;  cfg.lstm_batch = 128; cfg.predict_batch = 64;
    else
        cfg.train_batch = 128; cfg.lstm_batch = 256; cfg.predict_batch = 128;
    end

    if verbose
        fprintf('[auto_gpu_config] %s | %.1f GB total, %.1f GB free | CC %s\n', ...
                cfg.deviceName, cfg.vram_GB, cfg.avail_GB, cfg.computeCap);
        fprintf('[auto_gpu_config] Recommended batch sizes: train=%d  lstm=%d  predict=%d\n', ...
                cfg.train_batch, cfg.lstm_batch, cfg.predict_batch);
    end
end
