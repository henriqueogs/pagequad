"""Drop sources that are research-only (MIDV-500, SmartDoc) and all identity-card / passport collections (personal data) from a prepared .npy.
Usage: python filter_sources.py in.npy out.npy"""
import sys, numpy as np
BAD = ('midv', 'smartdoc', 'id_detections', 'dccc_vn', 'card_', 'ktp_sim', 'idcard', 'business_cards', 'passport')
items = [x for x in np.load(sys.argv[1], allow_pickle=True) if not any(b in x[2] for b in BAD)]
arr = np.empty(len(items), dtype=object)
for i, x in enumerate(items): arr[i] = x
np.save(sys.argv[2], arr, allow_pickle=True); print(len(items), 'items,', sum(1 for x in items if x[1] is None), 'negatives')
