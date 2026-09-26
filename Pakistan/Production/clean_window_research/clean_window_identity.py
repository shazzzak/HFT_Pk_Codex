# Resolve reused exchange IDs against immutable application-reference generations.
from bisect import bisect_right

# Preserve the existing sequence lookup while indexing identities by causal time.
class AddIndex(dict):
    # Build the secondary index only after loading one complete symbol-day.
    def seal(self):
        # Group every observed generation under its exchange order ID.
        self.by_oid={}
        # Retain sequence as the tie-breaker for same-timestamp amendments.
        for sequence,source in self.items():
            # Preserve the immutable source dictionary and its exact reference.
            self.by_oid.setdefault(source['oid'],[]).append((source['ts'],sequence,source))
        # Cache searchable time/sequence pairs without reconstructing them per snapshot.
        self.keys={}
        # Sort each identity's generations independently.
        for oid,rows in self.by_oid.items():
            # Use application order when two generations share one timestamp.
            rows.sort(key=lambda row:(row[0],row[1]))
            # Preserve only primitive pairs for fast causal searches.
            self.keys[oid]=[(ts,sequence) for ts,sequence,source in rows]
        # Permit explicit chaining after loading.
        return self

    # Return only the latest generation already available at the supplied cutoff.
    def latest(self,oid,asof,sequence=float('inf')):
        # Missing identities have no invented source mapping.
        if oid not in self.by_oid:
            # Their snapshot quantity must remain anonymous at its reported price.
            return None
        # Find the last causal time/sequence pair without using later amendments.
        index=bisect_right(self.keys[oid],(asof,sequence))-1
        # Return the immutable add metadata only when an earlier generation exists.
        return self.by_oid[oid][index][2] if index>=0 else None

# Keep production generation keys explicit while supporting legacy synthetic fixtures.
def order_key(source):
    # Source-loaded rows always provide a sequence-qualified key.
    return source.get('key',source['oid'])
