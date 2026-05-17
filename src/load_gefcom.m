function dataOut = load_gefcom(taskRange, dataRoot)
%LOAD_GEFCOM  Load and concatenate GEFCom2014 Track-L load + temperature data.
%
%   dataOut = LOAD_GEFCOM(taskRange, dataRoot) reads the GEFCom2014-L_V2
%   data extracted to <dataRoot>/01_GEFCom/GEFCom2014 Data/Load_extracted/Load/
%   for the tasks listed in TASKRANGE (default: 1:12 for training) and
%   returns a structure with the following fields:
%
%       dataOut.timestamp   datetime [T x 1] in tz Asia/Shanghai
%       dataOut.load_MW     double   [T x 1] active power load (MW)
%       dataOut.temp_F      double   [T x 25] 25 weather stations (deg F)
%       dataOut.temp_C      double   [T x 25] converted to deg C
%       dataOut.task_id     double   [T x 1] which task the row came from
%       dataOut.meta        struct   diagnostics (n_rows, n_missing, ...)
%
%   Rows whose LOAD field is NaN (the held-out forecast window of each task)
%   are RETAINED in the output but flagged via dataOut.is_target.
%
%   Reference:
%     Hong, T. et al. "Probabilistic energy forecasting: GEFCom 2014 and
%     beyond". International Journal of Forecasting 32(3), 896-913, 2016.

    if nargin < 1 || isempty(taskRange);  taskRange = 1:12;                end
    if nargin < 2 || isempty(dataRoot);   dataRoot  = locate_data_root();  end

    baseDir = fullfile(dataRoot, '01_GEFCom', 'GEFCom2014 Data', ...
                       'Load_extracted', 'Load');
    assert(isfolder(baseDir), 'GEFCom2014 base folder not found: %s', baseDir);

    rowCells = cell(numel(taskRange), 1);
    for k = 1:numel(taskRange)
        t      = taskRange(k);
        csvF   = fullfile(baseDir, sprintf('Task %d', t), sprintf('L%d-train.csv', t));
        if ~isfile(csvF)
            warning('load_gefcom:missingTask','Task %d CSV not found: %s', t, csvF);
            continue;
        end
        T = readtable(csvF, 'TextType', 'string');

        % normalize column names: ZONEID, TIMESTAMP, LOAD, w1..w25
        T.Properties.VariableNames = lower(T.Properties.VariableNames);
        assert(all(ismember({'zoneid','timestamp','load'}, T.Properties.VariableNames)), ...
               'unexpected columns in %s', csvF);

        % parse timestamp -- GEFCom uses NO separator: e.g. "112001 1:00"
        % = 1/1/2001 1:00.  Format is M[M]d[d]YYYY H:MM where Y is exactly 4 digits.
        ts = parse_gefcom_timestamps(T.timestamp);
        ts.TimeZone = 'America/New_York';   % GEFCom is US-based
        ts          = datetime(ts, 'TimeZone', 'Asia/Shanghai'); % unify to project tz

        weatherCols = arrayfun(@(i) sprintf('w%d',i), 1:25, 'UniformOutput', false);
        weatherCols = weatherCols(ismember(weatherCols, T.Properties.VariableNames));
        tempF = T{:, weatherCols};

        rowCells{k} = struct( ...
            'timestamp', ts, ...
            'load_MW',   T.load, ...
            'temp_F',    tempF, ...
            'task_id',   repmat(t, height(T), 1));
    end
    rowCells = rowCells(~cellfun(@isempty, rowCells));
    assert(~isempty(rowCells), 'No Task CSVs successfully loaded.');

    % concatenate (preserve order)
    cat_ts   = vertcat(rowCells{:});
    dataOut.timestamp = vertcat(cat_ts.timestamp);
    dataOut.load_MW   = vertcat(cat_ts.load_MW);
    dataOut.temp_F    = vertcat(cat_ts.temp_F);
    dataOut.task_id   = vertcat(cat_ts.task_id);

    % sort by timestamp (Task ordering does not guarantee chronology)
    [dataOut.timestamp, idx] = sort(dataOut.timestamp);
    dataOut.load_MW = dataOut.load_MW(idx);
    dataOut.temp_F  = dataOut.temp_F(idx, :);
    dataOut.task_id = dataOut.task_id(idx);

    % de-duplicate exact timestamp collisions (keep first non-NaN load)
    [~, firstIdx, ~] = unique(dataOut.timestamp, 'first');
    dataOut.timestamp = dataOut.timestamp(firstIdx);
    dataOut.load_MW   = dataOut.load_MW(firstIdx);
    dataOut.temp_F    = dataOut.temp_F(firstIdx, :);
    dataOut.task_id   = dataOut.task_id(firstIdx);

    % convert F -> C
    dataOut.temp_C = (dataOut.temp_F - 32) * 5/9;

    % target rows: load is NaN
    dataOut.is_target = isnan(dataOut.load_MW);

    % diagnostics
    dataOut.meta.n_rows         = height(table(dataOut.timestamp));
    dataOut.meta.n_target_rows  = sum(dataOut.is_target);
    dataOut.meta.date_min       = min(dataOut.timestamp);
    dataOut.meta.date_max       = max(dataOut.timestamp);
    dataOut.meta.n_weather_cols = size(dataOut.temp_F, 2);

    fprintf('[load_gefcom] %d rows, %d target rows, %s ~ %s, %d weather stations\n', ...
            dataOut.meta.n_rows, dataOut.meta.n_target_rows, ...
            string(dataOut.meta.date_min), string(dataOut.meta.date_max), ...
            dataOut.meta.n_weather_cols);
end

% --- helpers ----------------------------------------------------------------
function r = locate_data_root()
    % default: ../../data relative to this file (../../matlab_workspace/src/x.m)
    here = fileparts(mfilename('fullpath'));
    r    = fullfile(here, '..', '..', 'data');
    r    = char(java.io.File(r).getCanonicalPath());
end

% ---------------------------------------------------------------------------
% GEFCom-specific timestamp parser.
%
% Input strings have NO separators, e.g.
%      "112001 1:00"   = 1/1/2001 01:00
%      "1102001 1:00"  = 1/10/2001 01:00
%      "1112001 1:00"  ambiguous: 1/11/2001 OR 11/1/2001
%      "10102001 1:00" = 10/10/2001 01:00
%
% Format spec:  M[M] d[d] YYYY [SPACE] H:MM
%
% Ambiguity in 7-char dates is resolved by preferring the candidate closest
% to (previous parsed timestamp + 1 hour).  This works because GEFCom2014-L
% data is dense hourly with no large gaps in the train set.
function dts = parse_gefcom_timestamps(strs)
    n   = numel(strs);
    dts = NaT(n, 1);
    prev = NaT;
    for i = 1:n
        cands = parse_candidates(char(strs(i)));
        if isempty(cands); continue; end
        if numel(cands) == 1 || isnat(prev)
            dts(i) = cands(1);
        else
            target = prev + hours(1);
            [~, k] = min(abs(cands - target));
            dts(i) = cands(k);
        end
        prev = dts(i);
    end
end

function dts = parse_candidates(s)
    dts = datetime.empty;
    sp = find(s == ' ', 1);
    if isempty(sp); return; end
    ds = s(1:sp-1);
    ts = s(sp+1:end);
    nd = length(ds);
    if nd < 6 || nd > 8; return; end
    yr   = str2double(ds(nd-3:nd));
    rest = ds(1:nd-4);

    tc = find(ts == ':', 1);
    if isempty(tc); return; end
    hr = str2double(ts(1:tc-1));
    mn = str2double(ts(tc+1:end));
    if hr == 24    % GEFCom uses 1..24 hour convention
        hr = 0; rollover = true; else; rollover = false;
    end

    pairs = {};
    switch length(rest)
        case 2; pairs = {[str2double(rest(1)),  str2double(rest(2))]};
        case 3; pairs = {[str2double(rest(1)),  str2double(rest(2:3))], ...
                         [str2double(rest(1:2)), str2double(rest(3))]};
        case 4; pairs = {[str2double(rest(1:2)), str2double(rest(3:4))]};
    end
    for k = 1:numel(pairs)
        mo = pairs{k}(1);
        dy = pairs{k}(2);
        if mo >= 1 && mo <= 12 && dy >= 1 && dy <= 31
            try
                d = datetime(yr, mo, dy, hr, mn, 0);
                if rollover; d = d + days(1); end
                dts(end+1) = d; %#ok<AGROW>
            catch
            end
        end
    end
end
