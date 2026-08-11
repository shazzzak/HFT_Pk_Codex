# os for filesystem walking, re for pattern generalization, duckdb for schema reads
import os, re, duckdb
# defaultdict to group paths by their generalized layout pattern
from collections import defaultdict

# candidate roots: home dir and any mounted external/data drives
roots = [os.path.expanduser('~'), '/Volumes']

# regex that matches a YYYY-MM-DD date anywhere in a path segment
date_re = re.compile(r'\d{4}-\d{2}-\d{2}')

# turn a concrete path into a layout PATTERN so many files collapse into one group
def normalize(path):
    # split the path into its segments
    parts = path.split(os.sep)
    # rebuilt, generalized segments go here
    norm = []
    # inspect each segment
    for p in parts:
        # hive-style partition segment like symbol=UBL -> symbol=*
        if '=' in p:
            # keep the key, wildcard the value
            norm.append(p.split('=', 1)[0] + '=*')
        # a dated folder or filename like daily_stats_2025-09-12 -> daily_stats_*
        elif date_re.search(p):
            # replace the date token with a wildcard
            norm.append(date_re.sub('*', p))
        # otherwise keep the segment as-is
        else:
            norm.append(p)
    # reassemble into a pattern string
    return os.sep.join(norm)

# map: pattern -> list of real matching paths
groups = defaultdict(list)

# walk each root
for root in roots:
    # skip roots that do not exist (e.g. no external drives mounted)
    if not os.path.isdir(root):
        continue
    # recurse the tree
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden + heavy system dirs in place for speed
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('Library', 'Applications')]
        # look at every file in this directory
        for fn in filenames:
            # keep only parquet files
            if fn.endswith('.parquet'):
                # build the full path
                full = os.path.join(dirpath, fn)
                # add it under its generalized pattern
                groups[normalize(full)].append(full)

# report how many distinct layout patterns exist
print(f'distinct parquet path patterns: {len(groups)}\n')

# reuse a single duckdb connection for all schema reads
con = duckdb.connect()

# show biggest datasets first (most files = most likely your raw source)
for pattern, paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    # separator
    print('=' * 100)
    # file count and the generalized pattern
    print(f'{len(paths):>6} file(s)   pattern: {pattern}')
    # a concrete example path from this group
    print(f'         example: {paths[0]}')
    # try to describe the representative file's schema
    try:
        # DESCRIBE reads footer metadata only -> fast, low memory; single-quote the path (handles spaces)
        schema = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{paths[0]}')"
        ).df()[['column_name', 'column_type']]
        # print just column names and types
        print(schema.to_string(index=False))
    # if a file cannot be read, report why and continue
    except Exception as e:
        # surface the error without aborting the whole scan
        print('  (could not read schema:', e, ')')
