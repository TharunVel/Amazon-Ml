"""
Stage 3 (v2): Blocking using MinHash LSH

Uses Locality Sensitive Hashing on character 3-grams for fast,
high-recall approximate string matching. ~50x faster than TF-IDF
matrix multiplication while maintaining ~91%+ recall.

Checkpoints each (country, source) pair to disk so crashes are
recoverable without losing completed work.
"""

import os
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128       # Hash functions: higher = more accurate, slower
LSH_THRESHOLD = 0.3  # Jaccard threshold: lower = higher recall
TOP_N = 30           # Max candidates per S1 entity (caps feature computation)

def get_shingles(text, k=3):
    text = str(text).strip()
    if not text:
        return set()
    return set(text[i:i+k] for i in range(len(text) - k + 1))

def compute_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    for shingle in get_shingles(text):
        m.update(shingle.encode('utf8'))
    return m

def build_lsh_index(df, name_col='name_clean'):
    """Build MinHash LSH index from a DataFrame. Returns (lsh, minhashes)."""
    print(f"    Building LSH index for {len(df):,} records...", flush=True)
    t0 = time.time()
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}
    for _, row in df.iterrows():
        eid = row['entity_id']
        mh = compute_minhash(row[name_col])
        try:
            lsh.insert(eid, mh)
        except ValueError:
            pass  # Duplicate key, skip
        minhashes[eid] = mh
    print(f"    Done in {time.time()-t0:.1f}s", flush=True)
    return lsh, minhashes

def run_blocking_for_country(s1_df, s2_df, s3_df, country, checkpoint_dir):
    """
    Run all blocking channels for a specific country with checkpointing.
    Returns dict: s1_entity_id -> set of candidate entity_ids.
    """
    print(f"\n--- Blocking for country: '{country}' ---", flush=True)
    print(f"  S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}", flush=True)
    
    candidates = defaultdict(set)
    
    for target_df, target_name in [(s2_df, "S2"), (s3_df, "S3")]:
        ckpt_file = os.path.join(checkpoint_dir, f"{country}_{target_name}.pkl")
        
        # Load checkpoint if exists
        if os.path.exists(ckpt_file):
            print(f"  [{target_name}] Checkpoint found, loading...", flush=True)
            with open(ckpt_file, 'rb') as f:
                ckpt_cands = pickle.load(f)
            for k, v in ckpt_cands.items():
                candidates[k].update(v)
            print(f"  [{target_name}] Loaded {sum(len(v) for v in ckpt_cands.values()):,} candidate pairs from checkpoint", flush=True)
            continue
        
        if len(target_df) == 0 or len(s1_df) == 0:
            continue
        
        t0 = time.time()
        this_cands = defaultdict(set)
        c1, c2, c3 = 0, 0, 0
        
        # 1. Exact Name Match
        s1_name = s1_df[s1_df['name_clean'] != ''][['entity_id', 'name_clean']]
        tgt_name = target_df[target_df['name_clean'] != ''][['entity_id', 'name_clean']]
        merged = s1_name.merge(tgt_name, on='name_clean')
        for s1_id, tgt_id in zip(merged['entity_id_x'], merged['entity_id_y']):
            this_cands[s1_id].add(tgt_id)
            c1 += 1

        # 2. Exact Sorted Tokens Match
        s1_tok = s1_df[s1_df['name_tokens_sorted'] != ''][['entity_id', 'name_tokens_sorted']]
        tgt_tok = target_df[target_df['name_tokens_sorted'] != ''][['entity_id', 'name_tokens_sorted']]
        merged = s1_tok.merge(tgt_tok, on='name_tokens_sorted')
        for s1_id, tgt_id in zip(merged['entity_id_x'], merged['entity_id_y']):
            this_cands[s1_id].add(tgt_id)
            c2 += 1

        # 3. MinHash LSH on character 3-grams
        print(f"  [{target_name}] Building MinHash LSH index...", flush=True)
        lsh, tgt_minhashes = build_lsh_index(target_df)
        
        print(f"  [{target_name}] Querying {len(s1_df):,} S1 records...", flush=True)
        t_q = time.time()
        for idx, row in enumerate(s1_df.itertuples()):
            if not row.name_clean:
                continue
            s1_mh = compute_minhash(row.name_clean)
            results = lsh.query(s1_mh)
            
            # Cap to TOP_N most similar
            if len(results) > TOP_N:
                # Sort by estimated Jaccard and take top N
                scored = [(r, s1_mh.jaccard(tgt_minhashes[r])) for r in results]
                scored.sort(key=lambda x: -x[1])
                results = [r for r, _ in scored[:TOP_N]]
            
            for tgt_id in results:
                this_cands[row.entity_id].add(tgt_id)
                c3 += 1
            
            if (idx + 1) % 100000 == 0:
                elapsed = time.time() - t_q
                rate = (idx + 1) / elapsed
                eta = (len(s1_df) - idx - 1) / rate / 60
                print(f"    Queried {idx+1:,}/{len(s1_df):,} | ETA: {eta:.1f}min", flush=True)
        
        print(f"  [{target_name}] Done in {time.time()-t0:.1f}s", flush=True)
        print(f"    -> Exact Name: {c1:,}", flush=True)
        print(f"    -> Exact Tokens: {c2:,}", flush=True)
        print(f"    -> MinHash LSH: {c3:,}", flush=True)
        
        # Save checkpoint
        with open(ckpt_file, 'wb') as f:
            pickle.dump(dict(this_cands), f)
        print(f"  [{target_name}] Checkpoint saved.", flush=True)
        
        for k, v in this_cands.items():
            candidates[k].update(v)
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Blocking (MinHash LSH)")
    parser.add_argument("--split", choices=["val", "test", "train_full"], default="val")
    parser.add_argument("--s1-path", default="preprocessed/train/train_source1.tsv")
    parser.add_argument("--s2-path", default="preprocessed/train/train_source2.tsv")
    parser.add_argument("--s3-path", default="preprocessed/train/train_source3.tsv")
    parser.add_argument("--val-ids-path", default="validation/val_s1_ids.txt")
    parser.add_argument("--output-path", default="output/val_candidate_pairs.tsv")
    parser.add_argument("--checkpoint-dir", default="output/checkpoints")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading preprocessed data...", flush=True)
    t0 = time.time()
    s1 = pd.read_csv(args.s1_path, sep='\t').fillna('')
    s2 = pd.read_csv(args.s2_path, sep='\t').fillna('')
    s3 = pd.read_csv(args.s3_path, sep='\t').fillna('')

    if args.split == "val":
        print("Filtering S1 to validation set...", flush=True)
        with open(args.val_ids_path, 'r') as f:
            val_ids = set(line.strip() for line in f if line.strip())
        s1 = s1[s1['entity_id'].isin(val_ids)].copy()

    print(f"Loaded in {time.time()-t0:.1f}s. S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}", flush=True)

    all_candidates = defaultdict(set)
    countries = s1['country_clean'].unique()
    total_t0 = time.time()

    for country in countries:
        if not country:
            continue
        s1_c = s1[s1['country_clean'] == country]
        s2_c = s2[s2['country_clean'] == country]
        s3_c = s3[s3['country_clean'] == country]
        country_cands = run_blocking_for_country(s1_c, s2_c, s3_c, country, args.checkpoint_dir)
        for k, v in country_cands.items():
            all_candidates[k].update(v)

    print("\nFormatting output...", flush=True)
    output_rows = []
    for s1_id in s1['entity_id'].values:
        cands = all_candidates.get(s1_id, set())
        output_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(sorted(cands))
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(args.output_path, sep='\t', index=False)

    total_time = time.time() - total_t0
    total_cands = sum(len(c) for c in all_candidates.values())
    print(f"\nBlocking completed in {total_time/60:.1f} minutes!")
    print(f"Saved candidates to {args.output_path}")
    print(f"Total candidates: {total_cands:,}")
    print(f"Average per S1: {total_cands / max(len(s1),1):.1f}")

if __name__ == "__main__":
    main()
