# adjust the tar path to your real raw capture for 2026-06-30
tar -xzOf /path/to/2026-06-30.tar.gz --wildcards '*.txt' \
  | grep '35=W' | grep 'UBL' | awk '{print length, $0}' \
  | sort -rn | head -1 | cut -d' ' -f2- \
  | tr '\001' '|' | head -c 4000