# Pipeline Instructions for New Machine

## 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Trained Model
The trained LightGBM model is already saved in the repository at:
`output/lgb_model.txt`

## 3. Running Blocking on Test Set
To run the blocking stage on the test set:
```bash
python blocking.py \
  --split test \
  --s1-path preprocessed/test/test_source1.tsv \
  --s2-path preprocessed/test/test_source2.tsv \
  --s3-path preprocessed/test/test_source3.tsv \
  --output-path output/test_candidate_pairs.tsv \
  --checkpoint-dir output/checkpoints_test
```
*Note: Checkpoints are saved per (country, source) in `output/checkpoints_test/` so progress is preserved across runs.*

## 4. Feature Extraction & Pairwise Scoring
Run feature computation on candidate pairs:
```bash
python features.py \
  --candidate-path output/test_candidate_pairs.tsv \
  --s1-path preprocessed/test/test_source1.tsv \
  --s2-path preprocessed/test/test_source2.tsv \
  --s3-path preprocessed/test/test_source3.tsv \
  --output-path output/test_features.csv
```

## 5. Model Inference & Generating Submission
Generate the final predictions using the trained model:
```bash
python predict.py \
  --features-path output/test_features.csv \
  --model-path output/lgb_model.txt \
  --s1-path preprocessed/test/test_source1.tsv \
  --output-path output/matching_results.tsv \
  --threshold 0.5
```

## 6. Validate Submission
Run the official competition validator before submitting:
```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/test_candidate_pairs.tsv \
  --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
