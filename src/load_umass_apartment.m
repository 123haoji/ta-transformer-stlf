function dataOut = load_umass_apartment(year, dataRoot, opts)
%LOAD_UMASS_APARTMENT  Load UMass Smart* Apartment data for a target year.
%
%   dataOut = LOAD_UMASS_APARTMENT(YEAR, DATAROOT, OPTS) reads all
%   Apt<id>_<year>.csv files under
%      <dataRoot>/03_UMass_SMART/extracted/apartment/<year>/
%   and returns:
%
%     dataOut.timestamp    datetime [T x 1] common 15-min grid, tz Asia/Shanghai
%     dataOut.load_kW      double   [T x N_apts] resampled (mean) to 15 min
%     dataOut.apartment_id string   [N_apts x 1] e.g. "Apt001"
%     dataOut.year         double
%     dataOut.weather      table    co-located hourly weather (3y file)
%     dataOut.meta         struct
%
%   OPTS (optional struct):
%     .resampleMinutes   default 15
%     .maxApts           default Inf (cap for quick testing)
%     .missingThresh     default 0.10 -- apartments with >this fraction of NaN
%                                       AFTER resampling are dropped from
%                                       load_kW but listed in meta.dropped.
%
%   This loader is deliberately tolerant: UMass apartment dates do not align
%   perfectly between units (some start in Feb, some end in Dec); the output
%   timestamp is the intersection of available data across all retained apts.
%
%   Reference:
%     Barker, S. et al. "Smart*: An Open Dataset and Tools for Enabling
%     Research in Sustainable Homes". SustKDD, 2012.

    if nargin < 1 || isempty(year);     year     = 2016;                end
    if nargin < 2 || isempty(dataRoot); dataRoot = locate_data_root();  end
    if nargin < 3 || isempty(opts);     opts     = struct();            end
    if ~isfield(opts,'resampleMinutes'); opts.resampleMinutes = 15;     end
    if ~isfield(opts,'maxApts');         opts.maxApts         = Inf;    end
    if ~isfield(opts,'missingThresh');   opts.missingThresh   = 0.10;   end

    yearDir = fullfile(dataRoot, '03_UMass_SMART', 'extracted', ...
                       'apartment', num2str(year));
    assert(isfolder(yearDir), 'UMass apartment year dir not found: %s', yearDir);

    csvList = dir(fullfile(yearDir, sprintf('Apt*_%d.csv', year)));
    assert(~isempty(csvList), 'No Apt*_%d.csv files in %s', year, yearDir);
    if numel(csvList) > opts.maxApts
        csvList = csvList(1:opts.maxApts);
    end
    nApts = numel(csvList);

    % ---- pass 1: read every apt as timetable and discover grid bounds ----
    apts     = cell(nApts, 1);
    aptIds   = strings(nApts, 1);
    % NaT() without TZ cannot accept TZ-aware datetime assignment;
    % we initialize TZ on the first iteration once it is known.
    tStart   = NaT(nApts, 1);
    tEnd     = NaT(nApts, 1);

    for k = 1:nApts
        f   = fullfile(csvList(k).folder, csvList(k).name);
        T   = readtable(f, 'ReadVariableNames', false, ...
                            'Format', '%{yyyy-MM-dd HH:mm:ss}D%f');
        T.Properties.VariableNames = {'timestamp','power_kW'};
        T.timestamp.TimeZone = 'America/New_York';
        TT  = table2timetable(T);
        TT  = retime(TT, 'regular', 'mean', ...
                     'TimeStep', minutes(opts.resampleMinutes));
        if k == 1 && isempty(tStart.TimeZone)
            tStart.TimeZone = TT.timestamp.TimeZone;
            tEnd.TimeZone   = TT.timestamp.TimeZone;
        end
        apts{k}   = TT;
        aptIds(k) = string(extractBefore(csvList(k).name, sprintf('_%d.csv', year)));
        tStart(k) = TT.timestamp(1);
        tEnd(k)   = TT.timestamp(end);
    end

    gridStart = max(tStart);
    gridEnd   = min(tEnd);
    assert(gridStart < gridEnd, ...
        ['No common time grid across apartments. Adjust opts.maxApts or ', ...
         'inspect data integrity.']);

    grid_ts            = (gridStart:minutes(opts.resampleMinutes):gridEnd).';
    % the colon operator can drop TimeZone in some MATLAB versions
    if isempty(grid_ts.TimeZone)
        grid_ts.TimeZone = tStart(1).TimeZone;
    end

    % ---- pass 2: align to common grid, drop high-missing apts ----
    loadMat = NaN(numel(grid_ts), nApts);
    for k = 1:nApts
        TT             = retime(apts{k}, grid_ts, 'mean');
        loadMat(:, k)  = TT.power_kW;
    end

    fracMissing = mean(isnan(loadMat), 1);
    keep        = fracMissing <= opts.missingThresh;
    dropped     = aptIds(~keep);

    % fill short gaps remaining in retained columns (linear, then nearest)
    loadMatKeep = loadMat(:, keep);
    for j = 1:size(loadMatKeep, 2)
        loadMatKeep(:, j) = fillmissing(loadMatKeep(:, j), 'linear', ...
                                       'EndValues', 'nearest');
    end

    % ---- weather (3-year file split out by year) ----
    wxFile = fullfile(dataRoot, '03_UMass_SMART', 'extracted', ...
                      'apartment-weather', sprintf('apartment%d.csv', year));
    if isfile(wxFile)
        Wx = readtable(wxFile);
        % unix seconds -> datetime (the file column is named 'time')
        wxTs = datetime(Wx.time, 'ConvertFrom', 'posixtime', ...
                        'TimeZone', 'America/New_York');
        Wx.timestamp = wxTs;
        % convert imperial -> SI
        Wx.temperature_C = (Wx.temperature - 32) * 5/9;
        Wx.dewPoint_C    = (Wx.dewPoint    - 32) * 5/9;
        Wx.windSpeed_mps = Wx.windSpeed * 0.44704;
        % ensure timezone metadata is consistent
        if isempty(Wx.timestamp.TimeZone)
            Wx.timestamp.TimeZone = 'America/New_York';
        end
    else
        warning('load_umass_apartment:noWeather','Weather file missing: %s', wxFile);
        Wx = table();
    end

    % convert timestamps to project tz for downstream consistency
    grid_ts = datetime(grid_ts, 'TimeZone', 'Asia/Shanghai');
    if ~isempty(Wx) && height(Wx) > 0
        Wx.timestamp = datetime(Wx.timestamp, 'TimeZone', 'Asia/Shanghai');
    end

    dataOut.timestamp    = grid_ts;
    dataOut.load_kW      = loadMatKeep;
    dataOut.apartment_id = aptIds(keep);
    dataOut.year         = year;
    dataOut.weather      = Wx;
    dataOut.meta.n_apartments_input  = nApts;
    dataOut.meta.n_apartments_kept   = sum(keep);
    dataOut.meta.dropped_apartments  = dropped;
    dataOut.meta.grid_min            = gridStart;
    dataOut.meta.grid_max            = gridEnd;
    dataOut.meta.resample_minutes    = opts.resampleMinutes;
    dataOut.meta.fraction_missing    = fracMissing(keep);

    fprintf(['[load_umass_apartment] year %d: %d/%d apartments kept, ', ...
             'grid %s to %s @ %d-min, weather rows = %d\n'], ...
            year, sum(keep), nApts, string(gridStart), string(gridEnd), ...
            opts.resampleMinutes, height(Wx));
end

% --- helpers ----------------------------------------------------------------
function r = locate_data_root()
    here = fileparts(mfilename('fullpath'));
    r    = fullfile(here, '..', '..', 'data');
    r    = char(java.io.File(r).getCanonicalPath());
end
