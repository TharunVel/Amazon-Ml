"""
Validation framework for Entity Resolution pipeline.

Provides:
1. Stratified train/val split of ground truth (preserves singleton ratio + country distribution)
2. Macro-averaged F0.5 scorer (exact competition metric)
3. Blocking recall evaluator
4. Detailed per-entity and aggregate diagnostics

Usage:
    python validation.py --action split     # Create train/val split
    python validation.py --action score     # Score a predictions file against val ground truth
    python validation.py --action blocking  # Evaluate blocking recall against val ground truth
"""

import os
import sys
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# F0.5 scoring (exact competition metric)
# ---------------------------------------------------------------------------

def f_beta_per_entity(predicted_set: set, true_set: set, beta: float = 0.5) -> float:
    """
    Compute F_beta score for a single S1 entity.

    Special cases (matching competition rules):
    - Singleton (true_set empty) + empty prediction -> 1.0
    - Singleton + non-empty prediction -> 0.0
    - Non-singleton + empty prediction -> 0.0
    """
    if len(true_set) == 0 and len(predicted_set) == 0:
        return 1.0
    if len(true_set) == 0 and len(predicted_set) > 0:
        return 0.0
    if len(predicted_set) == 0 and len(true_set) > 0:
        return 0.0

    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f_score


def compute_macro_f05(predictions: dict, ground_truth: dict) -> dict:
    """
    Compute macro-averaged F0.5 score.

    Args:
        predictions: {s1_entity_id: set(matched_ids)}
        ground_truth: {s1_entity_id: set(matched_ids)}

    Returns:
        Dictionary with overall and breakdown metrics.
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    missing_entities = []
    per_entity = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = f_beta_per_entity(pred_set, true_set, beta=0.5)
        scores.append(score)
        per_entity[s1_id] = {
            'f05': score,
            'n_true': len(true_set),
            'n_pred': len(pred_set),
            'tp': len(pred_set & true_set),
            'fp': len(pred_set - true_set),
            'fn': len(true_set - pred_set),
            'is_singleton': len(true_set) == 0,
        }

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

        if s1_id not in predictions:
            missing_entities.append(s1_id)

    macro_f05 = np.mean(scores) if scores else 0.0

    # Compute aggregate precision/recall
    total_tp = sum(d['tp'] for d in per_entity.values())
    total_fp = sum(d['fp'] for d in per_entity.values())
    total_fn = sum(d['fn'] for d in per_entity.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        'macro_f05': macro_f05,
        'n_entities': len(scores),
        'n_singletons': len(singleton_scores),
        'n_non_singletons': len(non_singleton_scores),
        'singleton_f05': np.mean(singleton_scores) if singleton_scores else 0.0,
        'non_singleton_f05': np.mean(non_singleton_scores) if non_singleton_scores else 0.0,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'n_missing_entities': len(missing_entities),
        'per_entity': per_entity,
    }


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> dict:
    """
    Load ground truth TSV into {s1_id: set(matched_ids)}.
    """
    gt = pd.read_csv(path, sep='\t')
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')

    result = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        ids_str = row['matched_entity_ids'].strip()
        if ids_str:
            result[s1_id] = set(ids_str.split(','))
        else:
            result[s1_id] = set()

    return result


def load_predictions(path: str) -> dict:
    """
    Load predictions TSV into {s1_id: set(matched_ids)}.
    """
    df = pd.read_csv(path, sep='\t')
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')

    result = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        ids_str = str(row['matched_entity_ids']).strip()
        if ids_str:
            result[s1_id] = set(ids_str.split(','))
        else:
            result[s1_id] = set()

    return result


# ---------------------------------------------------------------------------
# Train / Validation split
# ---------------------------------------------------------------------------

def create_train_val_split(gt_path: str, s1_path: str, output_dir: str,
                           val_fraction: float = 0.2, seed: int = 42):
    """
    Create a stratified train/val split of the ground truth.

    Stratifies by:
    - country (US / India)
    - singleton vs non-singleton
    - number of matches bucket (0, 1, 2, 3, 4-5, 6+)

    This ensures the val set has the same distribution of difficulty
    as the full training set.
    """
    print("Creating train/val split...")
    print(f"  Val fraction: {val_fraction}")
    print(f"  Seed: {seed}")

    # Load ground truth
    gt = pd.read_csv(gt_path, sep='\t')
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt['n_matches'] = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)

    # Load S1 for country info
    s1 = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'country_clean'])
    gt = gt.merge(s1, left_on='source1_entity_id', right_on='entity_id', how='left')

    # Create stratification key
    def match_bucket(n):
        if n == 0: return '0_singleton'
        elif n == 1: return '1_one'
        elif n == 2: return '2_two'
        elif n == 3: return '3_three'
        elif n <= 5: return '4_four_five'
        else: return '5_six_plus'

    gt['strat_key'] = gt['country_clean'].astype(str) + '_' + gt['n_matches'].apply(match_bucket)

    # Stratified split
    rng = np.random.RandomState(seed)
    val_mask = np.zeros(len(gt), dtype=bool)

    for key, group_idx in gt.groupby('strat_key').groups.items():
        n_val = max(1, int(len(group_idx) * val_fraction))
        chosen = rng.choice(group_idx, size=n_val, replace=False)
        val_mask[chosen] = True

    gt_train = gt[~val_mask][['source1_entity_id', 'matched_entity_ids']].copy()
    gt_val = gt[val_mask][['source1_entity_id', 'matched_entity_ids']].copy()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'gt_train.tsv')
    val_path = os.path.join(output_dir, 'gt_val.tsv')

    gt_train.to_csv(train_path, sep='\t', index=False)
    gt_val.to_csv(val_path, sep='\t', index=False)

    # Also save the val S1 entity IDs for quick lookup
    val_s1_ids = set(gt_val['source1_entity_id'].values)
    val_ids_path = os.path.join(output_dir, 'val_s1_ids.txt')
    with open(val_ids_path, 'w') as f:
        for s1_id in sorted(val_s1_ids):
            f.write(s1_id + '\n')

    # Print stats
    gt_val_parsed = load_ground_truth(val_path)
    n_val_singletons = sum(1 for v in gt_val_parsed.values() if len(v) == 0)
    n_val_matches = sum(len(v) for v in gt_val_parsed.values())

    print(f"\n  Train: {len(gt_train):,} S1 entities")
    print(f"  Val:   {len(gt_val):,} S1 entities")
    print(f"  Val singletons: {n_val_singletons:,} ({100*n_val_singletons/len(gt_val):.1f}%)")
    print(f"  Val total match pairs: {n_val_matches:,}")

    # Verify stratification
    print("\n  Stratification check:")
    for split_name, split_df in [('Full', gt), ('Val', gt[val_mask])]:
        n = len(split_df)
        sing_pct = 100 * (split_df['n_matches'] == 0).sum() / n
        us_pct = 100 * (split_df['country_clean'] == 'us').sum() / n
        mean_m = split_df['n_matches'].mean()
        print(f"    {split_name:5s}: singletons={sing_pct:.1f}%, US={us_pct:.1f}%, mean_matches={mean_m:.2f}")

    print(f"\n  Saved:")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"    {val_ids_path}")

    return train_path, val_path


# ---------------------------------------------------------------------------
# Blocking recall evaluator
# ---------------------------------------------------------------------------

def evaluate_blocking(candidate_pairs: dict, ground_truth: dict,
                      verbose: bool = True) -> dict:
    """
    Evaluate blocking quality.

    Args:
        candidate_pairs: {s1_id: set(candidate_ids)}
        ground_truth: {s1_id: set(true_match_ids)}

    Returns:
        Dictionary with blocking metrics.
    """
    total_true_pairs = 0
    recovered_pairs = 0
    lost_pairs = 0
    total_candidates = 0
    entities_with_lost_matches = 0
    lost_details = []

    for s1_id, true_set in ground_truth.items():
        if len(true_set) == 0:
            continue  # singletons don't affect blocking recall

        candidates = candidate_pairs.get(s1_id, set())
        total_candidates += len(candidates)

        recovered = true_set & candidates
        lost = true_set - candidates

        total_true_pairs += len(true_set)
        recovered_pairs += len(recovered)
        lost_pairs += len(lost)

        if lost:
            entities_with_lost_matches += 1
            if len(lost_details) < 50:  # keep first 50 for inspection
                lost_details.append({
                    's1_id': s1_id,
                    'n_true': len(true_set),
                    'n_candidates': len(candidates),
                    'n_recovered': len(recovered),
                    'n_lost': len(lost),
                    'lost_ids': lost,
                })

    n_non_singletons = sum(1 for v in ground_truth.values() if len(v) > 0)
    blocking_recall = recovered_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_candidates = total_candidates / n_non_singletons if n_non_singletons > 0 else 0

    results = {
        'blocking_recall': blocking_recall,
        'total_true_pairs': total_true_pairs,
        'recovered_pairs': recovered_pairs,
        'lost_pairs': lost_pairs,
        'total_candidates': total_candidates,
        'avg_candidates_per_entity': avg_candidates,
        'n_non_singleton_entities': n_non_singletons,
        'entities_with_lost_matches': entities_with_lost_matches,
        'lost_details': lost_details,
    }

    if verbose:
        print(f"\n  Blocking Evaluation:")
        print(f"    Recall: {blocking_recall:.4f} ({recovered_pairs:,}/{total_true_pairs:,})")
        print(f"    Lost pairs: {lost_pairs:,}")
        print(f"    Entities with lost matches: {entities_with_lost_matches:,}/{n_non_singletons:,}")
        print(f"    Total candidates: {total_candidates:,}")
        print(f"    Avg candidates/entity: {avg_candidates:.1f}")
        print(f"    Reduction ratio: {total_candidates/total_true_pairs:.2f}x true pairs"
              if total_true_pairs > 0 else "")

        if lost_details:
            print(f"\n    Sample lost matches (first {len(lost_details)}):")
            for d in lost_details[:10]:
                print(f"      {d['s1_id']}: {d['n_lost']} lost of {d['n_true']} "
                      f"(had {d['n_candidates']} candidates)")

    return results


# ---------------------------------------------------------------------------
# Score predictions against ground truth
# ---------------------------------------------------------------------------

def score_predictions(pred_path: str, gt_path: str, verbose: bool = True) -> dict:
    """
    Score a predictions file against ground truth.
    """
    print(f"Scoring predictions...")
    print(f"  Predictions: {pred_path}")
    print(f"  Ground truth: {gt_path}")

    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)

    results = compute_macro_f05(preds, gt)

    if verbose:
        print(f"\n  === Results ===")
        print(f"  Macro F0.5:           {results['macro_f05']:.6f}")
        print(f"  Singleton F0.5:       {results['singleton_f05']:.6f}")
        print(f"  Non-singleton F0.5:   {results['non_singleton_f05']:.6f}")
        print(f"  Micro Precision:      {results['micro_precision']:.6f}")
        print(f"  Micro Recall:         {results['micro_recall']:.6f}")
        print(f"  Total TP:             {results['total_tp']:,}")
        print(f"  Total FP:             {results['total_fp']:,}")
        print(f"  Total FN:             {results['total_fn']:,}")
        print(f"  Entities evaluated:   {results['n_entities']:,}")
        print(f"  Missing entities:     {results['n_missing_entities']}")

        # Breakdown by match count
        pe = results['per_entity']
        by_bucket = defaultdict(list)
        for s1_id, d in pe.items():
            n = d['n_true']
            if n == 0: bucket = 'singleton'
            elif n == 1: bucket = '1 match'
            elif n == 2: bucket = '2 matches'
            elif n == 3: bucket = '3 matches'
            elif n <= 5: bucket = '4-5 matches'
            else: bucket = '6+ matches'
            by_bucket[bucket].append(d['f05'])

        print(f"\n  F0.5 by match count:")
        for bucket in ['singleton', '1 match', '2 matches', '3 matches', '4-5 matches', '6+ matches']:
            if bucket in by_bucket:
                scores = by_bucket[bucket]
                print(f"    {bucket:14s}: {np.mean(scores):.4f} (n={len(scores):,})")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validation framework for Entity Resolution")
    parser.add_argument("--action", required=True,
                        choices=["split", "score", "blocking"],
                        help="Action to perform")
    # Split args
    parser.add_argument("--gt-path",
                        default="6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv",
                        help="Path to ground truth TSV")
    parser.add_argument("--s1-path",
                        default="preprocessed/train/train_source1.tsv",
                        help="Path to preprocessed S1 (for country info in split)")
    parser.add_argument("--output-dir", default="validation",
                        help="Where to save split files")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split")
    # Score args
    parser.add_argument("--pred-path", default=None,
                        help="Path to predictions TSV (for scoring)")
    parser.add_argument("--val-gt-path", default="validation/gt_val.tsv",
                        help="Path to val ground truth (for scoring)")
    # Blocking args
    parser.add_argument("--candidate-path", default=None,
                        help="Path to candidate pairs TSV (for blocking eval)")

    args = parser.parse_args()

    if args.action == "split":
        create_train_val_split(
            gt_path=args.gt_path,
            s1_path=args.s1_path,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

    elif args.action == "score":
        if args.pred_path is None:
            print("ERROR: --pred-path required for scoring")
            sys.exit(1)
        score_predictions(args.pred_path, args.val_gt_path)

    elif args.action == "blocking":
        if args.candidate_path is None:
            print("ERROR: --candidate-path required for blocking eval")
            sys.exit(1)
        gt = load_ground_truth(args.val_gt_path)
        # Load candidate pairs
        cand_df = pd.read_csv(args.candidate_path, sep='\t')
        cand_df['candidate_entity_ids'] = cand_df['candidate_entity_ids'].fillna('')
        candidates = {}
        for _, row in cand_df.iterrows():
            s1_id = row['source1_entity_id']
            ids_str = str(row['candidate_entity_ids']).strip()
            if ids_str:
                candidates[s1_id] = set(ids_str.split(','))
            else:
                candidates[s1_id] = set()
        evaluate_blocking(candidates, gt)


if __name__ == "__main__":
    main()
