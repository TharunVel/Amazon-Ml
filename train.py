"""
Stage 5: LightGBM Classifier & Thresholding

Trains a LightGBM model on the pairwise features to classify true matches.
Optimizes for the F0.5 score by sweeping probability thresholds.
"""

import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="output/val_10k_features.csv")
    parser.add_argument("--gt-path", default="validation/gt_val_10k.tsv")
    parser.add_argument("--model-out", default="output/lgb_model.txt")
    args = parser.parse_args()

    print("Loading features...")
    t0 = time.time()
    feats_df = pd.read_csv(args.features_path)
    
    print("Loading Ground Truth...")
    gt = pd.read_csv(args.gt_path, sep='\t')
    
    # GT maps source1_entity_id to a comma-separated list of matches (which includes the source1_entity_id itself if it's a singleton, but here we only care about S2/S3 matches)
    print("Creating labels...")
    # Explode GT
    gt['match_ids'] = gt['matched_entity_ids'].astype(str).str.split(',')
    gt_exploded = gt.explode('match_ids')
    gt_exploded['is_match'] = 1
    
    # Merge to create labels
    merged = feats_df.merge(
        gt_exploded[['source1_entity_id', 'match_ids', 'is_match']], 
        left_on=['source1_entity_id', 'candidate_entity_id'],
        right_on=['source1_entity_id', 'match_ids'],
        how='left'
    )
    merged['is_match'] = merged['is_match'].fillna(0).astype(int)
    
    # Define feature columns
    feature_cols = [
        'name_exact', 'name_jaccard_word', 'name_jaccard_char3',
        'name_jaro_winkler', 'name_levenshtein',
        'addr_num_exact', 'addr_jaccard_word', 'addr_jaro_winkler',
        'country_exact'
    ]
    
    X = merged[feature_cols]
    y = merged['is_match']
    
    print(f"Data ready in {time.time()-t0:.1f}s. Shape: {X.shape}. Positives: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    train_data = lgb.Dataset(X, label=y)
    
    # Basic params
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 5,
        'feature_fraction': 0.9,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    model.save_model(args.model_out)
    
    print("Evaluating and Threshold Sweeping...")
    y_pred_prob = model.predict(X)
    
    best_f05 = 0
    best_thresh = 0
    
    print(f"\n{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F0.5':<10}")
    print("-" * 45)
    for thresh in np.arange(0.1, 0.95, 0.1):
        y_pred = (y_pred_prob >= thresh).astype(int)
        
        # Calculate standard pairwise metrics
        p = precision_score(y, y_pred, zero_division=0)
        r = recall_score(y, y_pred, zero_division=0)
        f05 = fbeta_score(y, y_pred, beta=0.5, zero_division=0)
        
        print(f"{thresh:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f05:<10.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"\nBest Pairwise F0.5 Score: {best_f05:.4f} at Threshold {best_thresh:.1f}")
    
    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (Gain):")
    print(feat_imp.to_string(index=False))

if __name__ == "__main__":
    main()
