# os for filesystem walking, duckdb for reading parquet schema
import os, duckdb

# candidate roots: your home dir and any mounted external/data drives
roots = [os.path.expanduser('~'), '/Volumes']

# accumulator for discovered parquet paths
found = []

# walk each root looking for parquet files
for root in roots:
    # recursively descend the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden + heavy system dirs in-place so the walk stays fast
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('Library', 'Applications')]
        # collect any parquet files in this directory
        for fn in filenames:
            # match the extension
            if fn.endswith('.parquet'):
                # record the full path
                found.append(os.path.join(dirpath, fn))
        # stop early once we have a small sample
        if len(found) >= 8:
            break
    # stop the outer loop too
    if len(found) >= 8:
        break

# report how many we found
print('found', len(found), 'parquet file(s):')

# print the first few example paths (these reveal your partition layout)
for f in found[:8]:
    print('  ', f)

# if we found anything, describe the schema of the first file
if found:
    # header
    print('\nSCHEMA of', found[0], ':')
    # DESCRIBE reads only footer metadata, not the row data -> fast, low memory
    print(duckdb.connect().execute(f"DESCRIBE SELECT * FROM read_parquet('{found[0]}')").df().to_string())
else:
    # nothing found: the data lives on a root we did not search
    print('\nNo parquet under', roots, '- tell me the path (e.g. an external SSD elsewhere).')