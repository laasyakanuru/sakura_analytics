#!/usr/bin/env python3
#%%
"""Resumable JMA daily weather collection. Python 3.10+, standard library only."""
import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://www.data.jma.go.jp/risk/obsdl/'
# Source-language labels are Unicode escapes; all code and messages are English.
METRICS = {
    'mean_temperature_c': ('201', '\u5e73\u5747\u6c17\u6e29'),
    'maximum_temperature_c': ('202', '\u6700\u9ad8\u6c17\u6e29'),
    'minimum_temperature_c': ('203', '\u6700\u4f4e\u6c17\u6e29'),
    'precipitation_mm': ('101', '\u964d\u6c34\u91cf'),
    'sunshine_duration_hours': ('401', '\u65e5\u7167\u6642\u9593'),
    'mean_relative_humidity_pct': ('605', '\u5e73\u5747\u6e7f\u5ea6'),
    'mean_wind_speed_ms': ('301', '\u5e73\u5747\u98a8\u901f'),
    'snowfall_cm': ('503', '\u964d\u96ea\u91cf'),
    'maximum_snow_depth_cm': ('501', '\u6700\u6df1\u7a4d\u96ea'),
    'solar_radiation_mj_m2': ('610', '\u5408\u8a08\u5168\u5929\u65e5\u5c04\u91cf'),
}
CORE = list(METRICS)[:5]
QUALITY = '\u54c1\u8cea\u60c5\u5831'
HOMOGENEITY = '\u5747\u8cea\u756a\u53f7'
NO_PHENOMENON = '\u73fe\u8c61\u306a\u3057\u60c5\u5831'
DATE_LABEL = '\u5e74\u6708\u65e5'
META = [('quality_code', QUALITY), ('homogeneity_number', HOMOGENEITY),
        ('no_phenomenon_flag', NO_PHENOMENON)]
KEYS = ['station_id', 'location', 'date']
STATION_COLS = ['station_id', 'location', 'jma_station_id', 'source_name',
                'prefecture_code', 'latitude', 'longitude', 'elevation_m',
                'observation_ended', 'mapping_status']


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=True, indent=2).encode('utf-8'))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class Client:
    def __init__(self, delay=3.0, retries=3, timeout=90):
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self.last = 0.0

    def post(self, route, data):
        for attempt in range(self.retries):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            request = Request(BASE + route, data=urlencode(data).encode('ascii'),
                              headers={'User-Agent': 'SakuraAcademicWeather/1.0',
                                       'Referer': BASE,
                                       'Content-Type': 'application/x-www-form-urlencoded'})
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read(), dict(response.headers.items())
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt + 1 == self.retries:
                    raise
                retry_after = error.headers.get('Retry-After', '')
                wait = float(retry_after) if retry_after.isdigit() else 5 * 2 ** attempt
                time.sleep(min(300, max(self.delay, wait)))
            except (URLError, TimeoutError, ConnectionError):
                if attempt + 1 == self.retries:
                    raise
                time.sleep(5 * 2 ** attempt)
        raise RuntimeError('Request failed.')


def targets_from_csv(path, selected=None):
    targets = {}
    for row in load_csv(path):
        sid, name = row['station_id'].strip(), row['location'].strip()
        if not re.fullmatch(r'\d{3}|47\d{3}', sid):
            raise ValueError(f'Unexpected station ID: {sid!r}')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9 _-]*', name):
            raise ValueError(f'Invalid English location: {name!r}')
        if sid in targets and targets[sid]['location'] != name:
            raise ValueError(f'Conflicting names for station {sid}')
        targets[sid] = {'station_id': sid, 'location': name,
                        'jma_station_id': 's' + (sid if len(sid) == 5 else '47' + sid)}
    if selected:
        missing = selected - targets.keys()
        if missing:
            raise ValueError(f'Station IDs not in input: {sorted(missing)}')
        targets = {k: v for k, v in targets.items() if k in selected}
    if not targets:
        raise ValueError('No target stations.')
    return sorted(targets.values(), key=lambda item: item['station_id'])


def cached_catalog(client, work, code):
    path = work / 'catalog' / f'prefecture_{code}.json'
    if path.exists():
        return json.loads(path.read_text())
    content, _ = client.post('top/station', {'pd': code, 'format': 'json'})
    result = json.loads(content)
    if result.get('view') not in {'prefecture', 'stationMap'}:
        raise ValueError('Unexpected JMA station catalog schema.')
    atomic_bytes(path, content)
    return result


def prefecture_codes(value):
    found = set()
    if isinstance(value, dict):
        if str(value.get('prid', '')).isdigit() and int(value['prid']) > 0:
            found.add(str(value['prid']).zfill(2))
        for child in value.values():
            found.update(prefecture_codes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(prefecture_codes(child))
    return found


def coordinates(title):
    def degrees(label):
        match = re.search(label + r'[:\uff1a]\s*([0-9.]+)\u5ea6([0-9.]+)\u5206', title)
        return round(float(match[1]) + float(match[2]) / 60, 6) if match else ''
    height = re.search(r'\u6a19\u9ad8[:\uff1a]\s*(-?[0-9.]+)m', title)
    return degrees('\u5317\u7def'), degrees('\u6771\u7d4c'), height[1] if height else ''


def prepare(args):
    selected = set(args.station_ids.split(',')) if args.station_ids else None
    targets = targets_from_csv(args.input, selected)
    client = Client(args.delay)
    codes = (args.prefectures.split(',') if args.prefectures else
             sorted(prefecture_codes(cached_catalog(client, args.work_dir, '00'))))
    if not codes:
        raise ValueError('No prefecture codes returned; JMA may have changed its catalog.')
    matched = {}
    wanted = {item['jma_station_id'] for item in targets}
    for code in codes:
        catalog = cached_catalog(client, args.work_dir, code.zfill(2))
        for marker in catalog.get('markers', []):
            if marker.get('kind') == 'station' and marker.get('stid') in wanted:
                matched[marker['stid']] = marker
        print(f'Catalog {code}: {len(matched)}/{len(wanted)} targets resolved.', flush=True)
        if wanted <= matched.keys():
            break
    rows = []
    for target in targets:
        row = dict.fromkeys(STATION_COLS, '')
        row.update(target)
        marker = matched.get(target['jma_station_id'])
        row['mapping_status'] = 'verified' if marker else 'unresolved'
        if marker:
            row.update(source_name=marker['stname'], prefecture_code=marker['prid'],
                       observation_ended=int(bool(marker.get('ended'))))
            row['latitude'], row['longitude'], row['elevation_m'] = coordinates(marker.get('title', ''))
        rows.append(row)
    write_csv(args.work_dir / 'stations.csv', STATION_COLS, rows)
    unresolved = sum(r['mapping_status'] != 'verified' for r in rows)
    print(f'Saved {len(rows)} stations; {unresolved} unresolved. Coordinates are current catalog metadata.')
    return 2 if unresolved else 0


def request_payload(station, start, end, metrics):
    # The current website requires strings inside ymdList, not JSON integers.
    return {'stationNumList': json.dumps([station['jma_station_id']]),
            'aggrgPeriod': '1', 'elementNumList': json.dumps([[METRICS[m][0], ''] for m in metrics]),
            'interAnnualType': '1',
            'ymdList': json.dumps([str(v) for v in (start.year, end.year, start.month,
                                                   end.month, start.day, end.day)]),
            'option': json.dumps({'list': ['obs'], 'nyear': 0}),
            'downloadFlag': 'true', 'rmkFlag': '1', 'disconnectFlag': '1',
            'csvFlag': '1', 'ymdLiteral': '1', 'kijiFlag': '0', 'youbiFlag': '0',
            'fukenFlag': '0', 'jikantaiFlag': '0', 'jikantaiList': '[]'}


def daily_dates(start, end):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def parse_response(content, station, start, end, metrics):
    if content.lstrip().startswith((b'<', b'{')):
        raise ValueError('JMA returned HTML/JSON instead of CSV; no observations imported.')
    for encoding in ('utf-8-sig', 'cp932'):
        try:
            rows = list(csv.reader(io.StringIO(content.decode(encoding))))
            break
        except UnicodeDecodeError:
            pass
    else:
        raise ValueError('CSV encoding must be UTF-8 or CP932.')
    header = next((i for i, r in enumerate(rows) if r and r[0].strip() == DATE_LABEL), None)
    if header is None or header == 0:
        raise ValueError('JMA date header not found.')
    start_row = next((i for i in range(header + 1, len(rows))
                      if rows[i] and re.fullmatch(r'\d{4}/\d{1,2}/\d{1,2}', rows[i][0].strip())), None)
    if start_row is None:
        raise ValueError('No dated rows; distinguish this from a valid all-missing CSV.')
    labels, places = rows[header], rows[header - 1]
    details = rows[header + 1:start_row]
    columns = {}
    for metric in metrics:
        prefix = METRICS[metric][1]
        candidates = [i for i, label in enumerate(labels) if i and label.startswith(prefix)
                      and not any(i < len(r) and r[i].strip() for r in details)]
        if len(candidates) != 1:
            raise ValueError(f'Missing or duplicate source column: {metric}')
        i = candidates[0]
        if places[i].strip() != station['source_name'].strip():
            raise ValueError(f'Wrong station in response for {station["station_id"]}')
        metadata = {}
        for j in range(i + 1, len(labels)):
            if labels[j] != labels[i] or places[j] != places[i]:
                break
            for detail in details:
                if j < len(detail) and detail[j].strip():
                    metadata[detail[j].strip()] = j
        if QUALITY not in metadata or HOMOGENEITY not in metadata:
            raise ValueError(f'Missing quality/homogeneity metadata: {metric}')
        columns[metric] = (i, metadata)
    output, seen = [], set()
    for source in rows[start_row:]:
        if not source or not any(c.strip() for c in source):
            continue
        if len(source) != len(labels):
            raise ValueError('Inconsistent CSV row width.')
        observed = datetime.strptime(source[0].strip(), '%Y/%m/%d').date()
        if not start <= observed <= end or observed in seen:
            raise ValueError('Duplicate or out-of-range date in response.')
        seen.add(observed)
        record = {'station_id': station['station_id'], 'location': station['location'],
                  'date': observed.isoformat()}
        for metric, (i, metadata) in columns.items():
            value = source[i].strip()
            if value and not math.isfinite(float(value)):
                raise ValueError(f'Non-finite value: {metric}')
            record[metric] = value
            for suffix, label in META:
                code = source[metadata[label]].strip() if label in metadata else ''
                if code and not code.isdigit():
                    raise ValueError(f'Unexpected source metadata: {metric}/{suffix}')
                record[metric + '_' + suffix] = code
        output.append(record)
    if seen != set(daily_dates(start, end)):
        raise ValueError('Response does not cover every requested date; nothing marked complete.')
    return sorted(output, key=lambda r: r['date'])


def year_chunks(start, end):
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def configuration(args):
    if args.start > args.end:
        raise ValueError('Start date must be on or before end date.')
    metrics = list(METRICS) if args.extended else CORE
    stations = load_csv(args.work_dir / 'stations.csv')
    if args.station_ids:
        selected = set(args.station_ids.split(','))
        if selected - {s['station_id'] for s in stations}:
            raise ValueError('Requested station IDs are absent from stations.csv.')
        stations = [s for s in stations if s['station_id'] in selected]
    if not stations or any(s['mapping_status'] != 'verified' for s in stations):
        raise ValueError('Resolve station mappings with prepare before downloading.')
    return stations, metrics


def download(args):
    stations, metrics = configuration(args)
    config = {'start': args.start.isoformat(), 'end': args.end.isoformat(), 'metrics': metrics,
              'stations': [{k: s[k] for k in ['station_id', 'location', 'jma_station_id', 'source_name']}
                           for s in stations]}
    token = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    run = args.work_dir / 'runs' / token
    write_json(run / 'config.json', config)
    jobs = len(stations) * len(list(year_chunks(args.start, args.end)))
    print(f'{len(stations)} stations; {jobs} station-year requests; run {token}', flush=True)
    if args.dry_run:
        return 0
    client = Client(args.delay)
    failures = []
    completed = 0
    fields = KEYS + metrics + [m + '_' + suffix for m in metrics for suffix, _ in META]
    for station in stations:
        for start, end in year_chunks(args.start, args.end):
            stem = f'{station["station_id"]}_{start.isoformat()}_{end.isoformat()}'
            raw, meta, clean = (run / folder / (stem + suffix)
                                for folder, suffix in [('raw', '.csv'), ('raw', '.json'), ('daily', '.csv')])
            payload = request_payload(station, start, end, metrics)
            try:
                if raw.exists() and meta.exists():
                    info = json.loads(meta.read_text())
                    content = raw.read_bytes()
                    if info['payload'] != payload or info['sha256'] != hashlib.sha256(content).hexdigest():
                        raise ValueError('Raw cache integrity mismatch; inspect before continuing.')
                else:
                    if args.max_requests is not None and completed >= args.max_requests:
                        continue
                    completed += 1
                    content, headers = client.post('show/table', payload)
                    normalized = {k.lower(): v for k, v in headers.items()}
                    if not normalized.get('content-disposition', '').lower().startswith('attachment'):
                        raise ValueError('JMA did not return a downloadable CSV attachment.')
                    # Preserve raw bytes even if parsing later fails; sidecar explains request.
                    atomic_bytes(raw, content)
                    write_json(meta, {'url': BASE + 'show/table', 'payload': payload,
                                     'retrieved_at': utcnow(), 'headers': headers,
                                     'sha256': hashlib.sha256(content).hexdigest()})
                records = parse_response(content, station, start, end, metrics)
                write_csv(clean, fields, records)
                print(f'OK {stem}: {len(records)} days', flush=True)
            except (ValueError, OSError, KeyError) as error:
                clean.unlink(missing_ok=True)
                failures.append({'station_id': station['station_id'], 'start': start.isoformat(),
                                 'end': end.isoformat(), 'error': str(error)})
                print(f'FAILED {stem}: {error}', file=sys.stderr, flush=True)
    write_json(run / 'failures.json', failures)
    return export_run(run, stations, metrics, args.start, args.end, failures)


def export_run(run, stations, metrics, start, end, failures):
    expected_dates = {d.isoformat() for d in daily_dates(start, end)}
    summaries, incomplete = [], False
    for station in stations:
        records, seen = [], set()
        for path in sorted((run / 'daily').glob(station['station_id'] + '_*.csv')):
            for record in load_csv(path):
                if record['date'] not in expected_dates or record['date'] in seen:
                    raise ValueError('Duplicate or unexpected date in normalized cache.')
                seen.add(record['date'])
                records.append(record)
        records.sort(key=lambda r: r['date'])
        complete = seen == expected_dates
        incomplete |= not complete
        folder = run / ('exports' if complete else 'partial_exports')
        slug = station['location'].replace(' ', '_')
        filename = f'weather_{slug}_{station["station_id"]}_daily.csv'
        # Keep only the matching completeness category for both output variants.
        obsolete = run / ('partial_exports' if complete else 'exports')
        (obsolete / filename).unlink(missing_ok=True)
        (obsolete / filename.replace('_daily.csv', '_daily_quality.csv')).unlink(missing_ok=True)
        fields = KEYS + metrics
        write_csv(folder / filename, fields, [{k: r[k] for k in fields} for r in records])
        quality_fields = fields + [m + '_' + suffix for m in metrics for suffix, _ in META]
        write_csv(folder / filename.replace('_daily.csv', '_daily_quality.csv'), quality_fields, records)
        for m in metrics:
            missing = sum(r[m] == '' for r in records)
            summaries.append({'station_id': station['station_id'], 'location': station['location'],
                              'measurement': m, 'requested_days': len(expected_dates),
                              'downloaded_days': len(records), 'unretrieved_days': len(expected_dates - seen),
                              'missing_values': missing, 'valid_values': len(records) - missing,
                              'quality_not_8': sum(r[m + '_quality_code'] != '8' for r in records),
                              'coverage_complete': int(complete)})
    write_csv(run / 'coverage.csv', list(summaries[0]), summaries)
    write_json(run / 'status.json', {'completed_at': utcnow(), 'coverage_complete': not incomplete,
                                    'request_failures': len(failures), 'new_requests_only_limit_supported': True})
    print(f'Results: {run.resolve()}\nCoverage complete: {not incomplete}; failures: {len(failures)}')
    return 2 if failures or incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Extract targets and verify them against the live JMA station catalog.')
    p.add_argument('--input', type=Path, default=Path('input/01_sakura_flowering.csv'))
    p.add_argument('--prefectures', help='Optional comma-separated prefecture codes, for a small pilot.')
    d = sub.add_parser('download', help='Download, resume, normalize and export daily weather.')
    d.add_argument('--start', type=date.fromisoformat, default=date(1952, 1, 1))
    d.add_argument('--end', type=date.fromisoformat, default=date(2025, 12, 31))
    d.add_argument('--extended', action='store_true', help='Include humidity, wind, snow and solar radiation.')
    d.add_argument('--dry-run', action='store_true', help='Print request count without downloading weather.')
    d.add_argument('--max-requests', type=int, help='Limit new weather requests; partial output is labelled.')
    for command in (p, d):
        command.add_argument('--work-dir', type=Path, default=Path('work'))
        command.add_argument('--station-ids', help='Comma-separated IDs from the input, e.g. 662,936.')
        command.add_argument('--delay', type=float, default=3.0, help='Minimum seconds between requests; at least 1.')
    args = parser.parse_args()
    if args.delay < 1:
        parser.error('--delay must be at least 1 second.')
    if getattr(args, 'max_requests', None) is not None and args.max_requests < 1:
        parser.error('--max-requests must be positive.')
    try:
        return prepare(args) if args.command == 'prepare' else download(args)
    except (ValueError, OSError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

# %%
