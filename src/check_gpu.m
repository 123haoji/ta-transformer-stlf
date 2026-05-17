function check_gpu()
%CHECK_GPU  Standalone GPU diagnostic for the TA-Transformer pipeline.
%
%   Prints a detailed report on the CUDA GPU detected by MATLAB and the
%   batch sizes that AUTO_GPU_CONFIG recommends for this hardware.
%
%   Run this BEFORE launching a training run to confirm:
%     (a) MATLAB can see your GPU
%     (b) Enough VRAM is free (no other process hogging it)
%     (c) Recommended batch sizes are appropriate
%
%   USAGE
%     >> check_gpu

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src'));

    fprintf('\n==================================================\n');
    fprintf('  TA-Transformer GPU diagnostic\n');
    fprintf('==================================================\n');

    fprintf('\nMATLAB version: %s\n', version);
    fprintf('Parallel Toolbox installed: %s\n', ternary(license('test','Distrib_Computing_Toolbox'), 'yes', 'NO'));

    try
        n = gpuDeviceCount;
    catch
        n = 0;
    end
    fprintf('CUDA GPU count: %d\n', n);

    if n == 0
        fprintf(2, '\n[!] No CUDA GPU detected by MATLAB.\n');
        fprintf('    Possible causes:\n');
        fprintf('      - Parallel Computing Toolbox not installed\n');
        fprintf('      - GPU driver too old (need CUDA 11+ for R2023a+)\n');
        fprintf('      - Running in a remote session that hides the GPU\n');
        fprintf('    Pipeline will fall back to CPU (~10x slower).\n\n');
        return;
    end

    g = gpuDevice;
    fprintf('\nGPU Device:\n');
    fprintf('  Name:               %s\n', g.Name);
    fprintf('  Compute Capability: %s\n', g.ComputeCapability);
    fprintf('  Total VRAM:         %.2f GB\n', g.TotalMemory / 1e9);
    fprintf('  Free VRAM:          %.2f GB\n', g.AvailableMemory / 1e9);
    fprintf('  CUDA Driver:        %s\n', g.DriverVersion);
    fprintf('  CUDA Runtime:       %s\n', g.ToolkitVersion);
    fprintf('  Multiprocessor cnt: %d\n', g.MultiprocessorCount);
    fprintf('  Max threads/block:  %d\n', g.MaxThreadsPerBlock);

    % Try a tiny GPU op to confirm it actually works
    try
        a = gpuArray(single(rand(1024, 1024)));
        b = a * a;
        wait(g);
        fprintf('  Smoke test (1024x1024 single matmul): PASS\n');
        clear a b;
    catch ME
        fprintf(2, '  Smoke test FAILED: %s\n', ME.message);
        return;
    end

    if g.AvailableMemory / 1e9 < g.TotalMemory / 1e9 * 0.6
        fprintf(2, '\n[!] Warning: less than 60%% of VRAM is free (%.1f / %.1f GB).\n', ...
                g.AvailableMemory/1e9, g.TotalMemory/1e9);
        fprintf('    Other processes may be using your GPU. Close them or call:\n');
        fprintf('      >> reset(gpuDevice())\n');
    end

    fprintf('\nRecommended batch sizes (from auto_gpu_config):\n');
    cfg = auto_gpu_config(false);
    fprintf('  TA-Transformer train:  %d\n', cfg.train_batch);
    fprintf('  LSTM mini-batch:       %d\n', cfg.lstm_batch);
    fprintf('  Inference batch:       %d\n', cfg.predict_batch);

    fprintf('\nReady to train. Suggested commands:\n');
    fprintf('  >> main(''Stage'', 2)               %% UMass-33 main experiment\n');
    fprintf('  >> main(''Stage'', 3)               %% e-bike sensitivity\n');
    fprintf('  >> run_paper(''Stage'', [2 3])      %% multi-seed paper run\n\n');
end

function s = ternary(c, a, b)
    if c; s = a; else; s = b; end
end
