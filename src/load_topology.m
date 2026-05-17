function topo = load_topology(caseName, dataRoot)
%LOAD_TOPOLOGY  Parse a MATPOWER case .m file into a graph structure.
%
%   topo = LOAD_TOPOLOGY(CASENAME) parses CASENAME (default 'case33bw')
%   from <dataRoot>/07_IEEE_TestCase/ and returns:
%
%     topo.N            scalar     number of buses
%     topo.bus          double [N x ?] raw bus matrix (Pd in kW per case33bw)
%     topo.branch       double [Nb x ?] raw branch matrix
%     topo.adj_binary   double [N x N] symmetric 0/1 adjacency
%     topo.adj_weighted double [N x N] symmetric, weight = 1/|R+jX|
%     topo.adj_norm     double [N x N] symmetric normalized Laplacian-style
%                                       D^{-1/2} (A + I) D^{-1/2}
%     topo.bus_Pd_kW    double [N x 1] tabulated active load in kW
%     topo.bus_Qd_kVar  double [N x 1] tabulated reactive load in kVar
%     topo.from         double [Nb x 1] from-bus per branch
%     topo.to           double [Nb x 1] to-bus per branch
%     topo.R_ohm        double [Nb x 1]
%     topo.X_ohm        double [Nb x 1]
%     topo.case_name    char
%
%   The parser is robust to typical MATPOWER formatting (semicolons,
%   trailing comments) but does NOT require MATPOWER to be on the path; it
%   regex-extracts the mpc.bus and mpc.branch blocks directly.

    if nargin < 1 || isempty(caseName); caseName = 'case33bw';        end
    if nargin < 2 || isempty(dataRoot); dataRoot = locate_data_root(); end

    casePath = fullfile(dataRoot, '07_IEEE_TestCase', [caseName '.m']);
    assert(isfile(casePath), 'MATPOWER case file not found: %s', casePath);
    text     = fileread(casePath);

    % strip block comments and per-line comments to make regex robust
    text = regexprep(text, '%.*?$', '', 'lineanchors');

    bus    = extract_matrix(text, 'mpc\.bus');
    branch = extract_matrix(text, 'mpc\.branch');
    assert(~isempty(bus) && ~isempty(branch), ...
           'Failed to parse bus/branch from %s', casePath);

    N  = size(bus, 1);
    Nb = size(branch, 1);

    % MATPOWER columns: branch(:,1)=fbus, branch(:,2)=tbus, branch(:,3)=r, branch(:,4)=x
    f      = branch(:, 1);
    t      = branch(:, 2);
    R      = branch(:, 3);
    X      = branch(:, 4);
    Zmag   = abs(complex(R, X));
    w      = 1 ./ max(Zmag, 1e-6);   % impedance-based weight, guarded
    w      = w / median(w);          % normalize for numerical stability

    A_bin  = zeros(N);
    A_wt   = zeros(N);
    for k = 1:Nb
        i = f(k);  j = t(k);
        if i < 1 || j < 1 || i > N || j > N
            warning('load_topology:badBranch', ...
                'Branch row %d has out-of-range bus indices (%d -> %d).', k, i, j);
            continue;
        end
        A_bin(i, j) = 1;        A_bin(j, i) = 1;
        A_wt (i, j) = w(k);     A_wt (j, i) = w(k);
    end

    % normalized symmetric Laplacian-style A
    A_hat   = A_wt + eye(N);
    d_inv   = 1 ./ sqrt(max(sum(A_hat, 2), 1e-6));
    A_norm  = (d_inv .* A_hat) .* d_inv';     % element-wise scaling: faster than D^{-1/2} A D^{-1/2}

    topo.case_name     = caseName;
    topo.N             = N;
    topo.bus           = bus;
    topo.branch        = branch;
    topo.bus_Pd_kW     = bus(:, 3);
    topo.bus_Qd_kVar   = bus(:, 4);
    topo.from          = f;
    topo.to            = t;
    topo.R_ohm         = R;
    topo.X_ohm         = X;
    topo.adj_binary    = A_bin;
    topo.adj_weighted  = A_wt;
    topo.adj_norm      = A_norm;

    % connectivity sanity check
    G    = graph(A_bin);
    bins = conncomp(G);
    nCmp = max(bins);
    if nCmp > 1
        warning('load_topology:disconnected', ...
            '%s has %d connected components; check topology integrity.', ...
            caseName, nCmp);
    end

    fprintf('[load_topology] %s: %d buses, %d branches, %d component(s).\n', ...
            caseName, N, Nb, nCmp);
end

% --- helpers ----------------------------------------------------------------
function M = extract_matrix(text, name)
    % Find  <name> = [ ... ];   block, parse rows of doubles.
    pat = sprintf('%s\\s*=\\s*\\[(.*?)\\];', name);
    tok = regexp(text, pat, 'tokens', 'once');
    if isempty(tok)
        M = [];  return
    end
    body = tok{1};
    rows = regexp(body, '\n', 'split');
    M    = [];
    for r = rows
        line = strtrim(r{1});
        line = regexprep(line, ';\s*$', '');
        if isempty(line); continue; end
        nums = sscanf(line, '%f');
        if isempty(nums); continue; end
        if isempty(M)
            M = nums.';
        else
            % handle ragged rows defensively
            if numel(nums) ~= size(M, 2)
                if numel(nums) > size(M, 2)
                    nums = nums(1:size(M,2));
                else
                    nums(end+1:size(M,2)) = NaN;
                end
            end
            M(end+1, :) = nums.'; %#ok<AGROW>
        end
    end
end

function r = locate_data_root()
    here = fileparts(mfilename('fullpath'));
    r    = fullfile(here, '..', '..', 'data');
    r    = char(java.io.File(r).getCanonicalPath());
end
