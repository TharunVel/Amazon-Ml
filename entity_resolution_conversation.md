# Conversation: Candidate Generation, Search, and Entity Resolution

---

## 1. Candidate generation for billions of records — algorithms

Candidate generation is the first stage of a typical multi-stage retrieval pipeline (candidate generation → ranking → re-ranking), and its job is to narrow billions of items down to a few hundred/thousand that are plausibly relevant, fast enough to serve online.

### The core idea: two-stage funnel

You never score billions of items with a heavy ranking model. Instead:

1. **Candidate generation** — cheap, high-recall, approximate. Billions → thousands.
2. **Ranking** — expensive, precise model (e.g., deep neural net with many features). Thousands → hundreds.
3. **Re-ranking/blending** — business logic, diversity, freshness. Hundreds → tens shown to the user.

### Main algorithmic families

**1. Embedding-based retrieval (most common today)**
Represent users/queries and items as vectors in the same space, then find nearest neighbors.
- **Two-tower neural networks**: one tower encodes the user/query, the other encodes the item; trained so relevant pairs have high dot-product/cosine similarity. Used at YouTube, Pinterest, Twitter, etc.
- Once you have item embeddings, the actual "billions of records" problem becomes **Approximate Nearest Neighbor (ANN) search**:
  - **HNSW** (Hierarchical Navigable Small World graphs) — graph-based, very fast, good recall, popular default (used in FAISS, Milvus, Weaviate, Qdrant).
  - **IVF + Product Quantization (IVF-PQ)** — partitions space into clusters (inverted file), compresses vectors (PQ) so billions fit in memory. FAISS's workhorse for huge datasets.
  - **LSH (Locality Sensitive Hashing)** — hash similar vectors into same buckets; older, less recall-efficient than HNSW/IVF-PQ but simple and used in streaming settings.
  - **ScaNN** (Google) — anisotropic quantization, often better recall/latency tradeoff than FAISS for certain workloads.
  - **DiskANN / SPANN** — designed for billion-scale vectors that don't fit in RAM, using SSD-resident graphs.

**2. Collaborative filtering / co-occurrence based**
- Matrix factorization (ALS, SVD++) to get latent user/item vectors — still an embedding problem, same ANN infra applies.
- Item-item co-occurrence ("users who did X also did Y") precomputed offline into a lookup table — cheap, no ANN needed, good for "similar items" candidates.

**3. Term/lexical retrieval (for text/search)**
- **Inverted indices** + **BM25/TF-IDF** — classic search engine approach (Lucene, Elasticsearch, OpenSearch). Still extremely effective, and often combined with embeddings ("hybrid search").

**4. Graph-based**
- Personalized PageRank / random walks on user-item bipartite graphs (Pinterest's PixieRank/PinSage) — good for "related pins" style candidates and cold-start.

**5. Rule/heuristic based**
- Popularity, trending, geo/time filters, recency — cheap, high-precision candidates that complement the above; almost always blended in.

### Scaling to billions of records — system-level tricks

- **Sharding**: partition the index across machines (by hash, cluster, or geography); query shards in parallel, merge top-k.
- **Quantization/compression**: PQ, scalar quantization, or binary embeddings shrink billion-scale vector sets to fit in RAM.
- **Offline precomputation**: embeddings and indices are built in batch (Spark/Flink jobs), refreshed periodically; online path only does lookup, not training.
- **Multi-source blending**: production systems rarely rely on one method — they run several candidate generators in parallel (ANN embedding retrieval + co-occurrence + trending + graph walk) and merge/de-dupe the results before ranking.
- **Caching**: popular queries/users get cached candidate sets.
- **Approximate over exact**: recall@k of 90-95% with 10-100x speedup beats exact kNN, since the ranking stage cleans up ordering anyway.

### Common libraries/systems

- **FAISS** (Meta) — the most widely used ANN library, supports IVF-PQ, HNSW, GPU indices.
- **ScaNN** (Google)
- **Milvus, Weaviate, Qdrant, Vespa, Pinecone** — production vector databases wrapping these algorithms with sharding/replication/filtering built in.
- **Elasticsearch/OpenSearch** with `dense_vector` fields — combines lexical + vector (hybrid) retrieval.
- **pgvector** — if you want ANN inside Postgres (fine up to tens/hundreds of millions, less ideal at true billion-scale without extra sharding).

---

## 2. Candidate generation for search, simplified

### The problem
You type a query. There are billions of documents/pages. You can't compare your query to every single one in real time. The system needs a fast first pass that throws away 99.99% of documents and keeps only a few thousand "maybe relevant" ones. Then a slower, smarter model ranks just those few thousand. That first, fast pass = **candidate generation**.

### The two main tricks

**1. Inverted index (the classic way)**

Think of it like the index at the back of a book, but for every word across every document.

- For every word, keep a list of which documents contain it: `"pizza" → [doc5, doc19, doc3021, ...]`
- When you search "pizza recipe", the system looks up both lists and finds documents in both (or scores them with **BM25**, which rewards documents where your words appear often but aren't just common filler words).
- Extremely fast because it's just list lookups, not reading every document.

**2. Embeddings + nearest neighbor search (the modern AI way)**

- Turn your query into a list of numbers (a "vector") that captures its *meaning*, not just its exact words. Turn every document into a vector too, the same way.
- Similar meanings → vectors that are close together in space.
- "Candidate generation" becomes: *find the documents whose vector is closest to my query's vector.*
- Doing this exactly against billions of vectors is slow, so we use **approximate nearest neighbor (ANN)** search (common ones: **HNSW**, **FAISS**, **ScaNN**) — finds "close enough" matches almost instantly.
- This is why you can search "cheap food that's not too filling" and get pizza results even without the word "pizza" — it matches meaning, not just keywords.

### What real search engines do
Almost all serious search systems run **both** in parallel — keyword lookup (inverted index) AND meaning-based lookup (embeddings + ANN) — then merge the two candidate lists and hand them to a smarter ranking model that picks the final order shown to the user.

**One-line summary:** keyword search = fast dictionary lookup by exact words; embedding search = fast lookup by meaning, using compressed math shortcuts (ANN) so it still works at billion-scale. Real systems combine both, then rank the combined shortlist properly.

---

## 3. Entity resolution task — problem statement

Source 1 is the deduplicated reference source. The task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many records from Source 2 and Source 3. There is no common identifier, and some of the data is noisy or partial.

**Schema** (all three source files, `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`):
1. `entity_id` — unique identifier, prefixed `S1-`, `S2-`, or `S3-` (source is inferred from prefix/file, no separate source column)
2. `business_name` — may contain abbreviations, legal suffixes, typos, transliterations
3. `business_address` — may contain partial addresses, format variations, missing components, landmark-based references
4. `country` — training covers US and India; the test set additionally contains France, which does not appear in training. Country must be treated as an **open set** of string labels — no hard-coding, filtering, or one-hot encoding to a fixed set, and every test entity (France included) must appear in the submission.

**Ground truth** (`train_ground_truth.tsv`):
1. `source1_entity_id`
2. `matched_entity_ids` — comma-separated list of matching Source 2/3 IDs (empty when no matches)

**Noise patterns to expect:**
- Name variations: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs "and"), word-order transpositions, typos
- Address variations: abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN code, no state), landmark-based references (Near SBI ATM), municipal numbering formats, component reordering

---

## 4. Discussion: normalization vs. blocking + pairwise scoring

The user raised the concern that field normalization won't work given the huge number of variations, and asked about relying on blocking + pairwise scoring instead.

**Response:** Blocking and pairwise scoring are the right core combo, but normalization isn't all-or-nothing — cutting it out entirely tends to hurt blocking more than it helps, because blocking's job is to cheaply guess which records might match, and dirty blocking keys fracture blocks so the true match never even reaches scoring. The recommendation was to keep a **light, cheap normalization layer** (lowercase, strip punctuation/whitespace, remove accents) while avoiding heavy normalization (full address parsing, aggressive rule-based rewriting), which is what actually breaks under large variation.

To make blocking robust to variation without heavy normalization, use **multiple fuzzy/redundant blocking keys in parallel**, unioning candidates across all of them:
- Phonetic keys (Soundex, Double Metaphone) on names
- N-gram/token blocking for typos and reordering
- Sorted neighborhood blocking
- Multiple independent blocking passes (name-phonetic, zip+year, email-domain, etc.), unioned together

For pairwise scoring: Jaro-Winkler/Levenshtein for names, token-based Jaccard/cosine on n-grams for longer noisy fields, combined into a match probability via Fellegi-Sunter or a trained classifier (logistic regression / gradient boosting) so the model learns which fields are reliable rather than hand-weighting them.

Libraries suggested: `splink`, `recordlinkage`, `dedupe`.

---

## 5. Full pipeline for this specific dataset

### Blocking strategy (multi-pass, union of candidates)
- **Name n-gram / MinHash-LSH blocking**: character 3/4-grams, MinHash, LSH buckets — catches typos, transliterations, reordering
- **Phonetic blocking on name tokens**: Double Metaphone per token
- **First-token / sorted-token blocking**: sort name tokens, block on first 1-2 — catches word-order transpositions
- **Address-derived blocking**: extract PIN/ZIP via regex, block on exact PIN; fall back to city/locality token overlap when missing
- **Country as a coarse pre-filter**: exact-string equality used to partition comparisons (S1×S2, S1×S3) — safe for the open-set requirement since it's equality-based partitioning, not a learned representation of country identity

Union of all candidate pairs across passes is the blocking output.

### Pairwise similarity features

**`business_name`:**
- Jaro-Winkler (typo-tolerant, prefix-weighted)
- Token sort ratio (word-order transpositions)
- Character n-gram Jaccard/Dice (transliteration drift)
- Legal-suffix-normalized token overlap using a small finite dictionary (Corp/Corporation, Ltd/Limited, Pvt/Private, Inc/Incorporated, & / and) — a bounded lookup table, not full normalization

**`business_address`:**
- Token Jaccard/Dice on address tokens
- PIN/ZIP exact-match flag, with missing treated as its own "unknown" state, not a mismatch
- Numeric-token overlap (building/street numbers) — strong signal when present
- Fuzzy substring containment (handles landmark-based vs. formal address pairs)

**`country`:**
- Single binary equality feature (match / no-match / one-missing) — never one-hot the raw string, since that would break on unseen labels like France

### Scoring/classification
- Train a classifier (logistic regression or gradient boosting — XGBoost/LightGBM) on the similarity features using `train_ground_truth.tsv`
- Positive pairs = ground-truth matches; negative pairs = same-block pairs *not* in ground truth (hard negatives, not random negatives)
- Output a match probability per pair; threshold it. Do not force one-to-one matching — keep every pair above threshold per S1 entity, since matches can be zero, one, or many
- Tune the threshold on a held-out slice of training ground truth using pair-level or set-level F-score

### Why this generalizes to France with zero training examples
Every feature is either a string-similarity function or an equality check — none encode *which* country's data is being examined. The classifier's decision boundary is learned purely from name/address similarity patterns, which are country-agnostic by construction; country only ever acts as a partition key at blocking time and an equality feature at scoring time.

---

## 6. Step-by-step explanation with alternatives at each stage

### Stage 0: Problem framing
This is entity resolution / record linkage, specifically "many-to-many with a designated reference source." Source 1 is deduplicated (one row = one entity); Source 2/3 may have duplicates and zero/multiple matches per S1 entity. This rules out one-to-one matching approaches (e.g., Hungarian algorithm / bipartite max-matching) and confirms a classify-each-pair-then-group approach is needed.

### Stage 1: Preprocessing
| Level | What it does | Risk |
|---|---|---|
| Minimal | lowercase, strip whitespace/punctuation, remove accents | Very low risk, almost always worth doing |
| Tokenization | split into words, drop empty tokens | Low risk |
| Closed-vocabulary substitution | replace known legal suffixes via a small lookup table | Low risk if the table stays small/curated |
| Full normalization/parsing | address parsers (libpostal), name-part splitting, transliteration engines | High effort, breaks under high variation — skip |

By field: `business_name` gets minimal + closed-vocab substitution; `business_address` gets minimal only (full parsing breaks on landmarks/missing components); `country` gets exact-string trim/case-fold only, never mapped to a fixed category set.

### Stage 2: Blocking (candidate generation)
Exists because full S1×(S2+S3) comparison is infeasible at scale. The core tradeoff is recall vs. candidate-set size, handled by running multiple blocking passes and unioning results rather than relying on one perfect key.

Options:
1. **Exact-key blocking** — fast but fails hard on missing/noisy keys; use only as one of several passes
2. **Phonetic blocking** (Soundex, Double Metaphone) — good for typos/some transliteration, weak for word-order and abbreviation differences
3. **N-gram / MinHash-LSH blocking** — robust to typos, partial matches, reordering; best general-purpose option under high variation
4. **Sorted-neighborhood blocking** — simple, tolerant to small key errors, sensitive to window size
5. **Canopy clustering / TF-IDF cosine blocking** — more compute, sometimes better blocking-stage precision

Recommended combination for this dataset: n-gram/MinHash-LSH + sorted-token blocking on name, exact PIN/ZIP falling back to city/locality token overlap, country as an exact-match filter within each pass.

Scales differently by data size: smaller datasets can afford looser blocking (let scoring do more work); larger datasets need blocking precision tuned more carefully (e.g., tuned LSH band/row parameters), adding blocking passes narrowly rather than widening broadly.

### Stage 3: Pairwise similarity scoring
**Name similarity methods and their strengths/weaknesses:**
| Method | Good for | Weak for |
|---|---|---|
| Levenshtein / edit distance | typos, small char-level noise | word reordering, length differences |
| Jaro-Winkler | typos, especially near string start | large structural differences |
| Token sort ratio | word-order transpositions | typos within tokens |
| Token set ratio / Jaccard on tokens | partial overlap, extra/missing words | doesn't catch typos |
| Character n-gram Jaccard/Dice | transliteration drift, typos | can be noisy on short names |
| Cosine similarity on TF-IDF/n-gram vectors | general robustness, weights rare tokens higher | needs a corpus for IDF |

Compute several as separate features rather than picking one, letting the downstream classifier learn which signal to trust.

**Address similarity:** token Jaccard/Dice, numeric token overlap (high-precision when present), PIN/ZIP exact match flag with explicit "missing" state, fuzzy substring containment.

**Country:** one equality feature only (match / mismatch / one-missing).

If address data proves unreliable in practice, let the classifier downweight it automatically rather than manually reweighting.

### Stage 4: Classification
Options, increasing sophistication:
1. **Rule-based thresholding** — interpretable, no training data needed, but brittle; useful only as a baseline/fallback
2. **Fellegi-Sunter probabilistic model** — classic record-linkage statistical model; doesn't require labels (EM-fittable), but a supervised version outperforms it when labels exist (implemented in `splink`)
3. **Supervised ML classifier** (logistic regression, random forest, XGBoost/LightGBM) — best fit here given labeled ground truth; negative sampling matters (use same-block non-matches as hard negatives, not random negatives); gradient boosting handles nonlinear feature interactions well
4. **Deep/embedding-based matching** (Siamese networks, fine-tuned sentence embeddings) — overkill unless transliteration/multilingual variation is severe enough that string metrics genuinely fail; worth testing empirically before committing

Recommended: gradient boosting classifier on the Stage 3 features.

### Stage 5: Threshold selection and one-to-many assembly
- Classifier outputs a probability per pair, not a hard decision
- Do not force top-1 matching (no Hungarian algorithm) — keep every pair above threshold, per S1 entity independently, since matches can be zero/one/many
- Tune threshold via F-score (F1 or F₀.₅ depending on the actual scoring criteria) on a held-out slice of `train_ground_truth.tsv`, sweeping threshold values and picking based on which error type is costlier
- Consider per-source thresholds if S2 and S3 have different noise characteristics

### Handling the open-set country requirement (France)
Country only ever functions as (a) a blocking partition and (b) an equality feature — both operate on any string pair, seen or unseen, so France requires zero pipeline changes. Sanity check: temporarily rename "US" to something else in a copy of the code and confirm nothing breaks. Every French S1 entity must appear in the submission even with an empty match set, so the assembly step must iterate over *all* S1 entity_ids from the source file, not just ones that survived blocking.

### Evaluation strategy
- Split `train_ground_truth.tsv` into train/validation (e.g., 80/20)
- Compute matched-set F-score per S1 entity, then average (macro)
- Break down performance by source (S2 vs S3) and by whether the entity has zero/one/many true matches, to reveal whether one-to-many logic specifically is underperforming

### Implementation options
| Approach | When to use |
|---|---|
| **Splink** (Python, DuckDB/Spark backend) | Best fit here — built-in blocking rules, Fellegi-Sunter and supervised scoring, scales well, handles missing data natively |
| **Dedupe.io** (Python) | Good for smaller datasets, active-learning-based labeling |
| **recordlinkage** (Python) | More manual/flexible, full control over each stage |
| **Custom (pandas + datasketch + XGBoost)** | Most control, best if Splink's built-in rules don't fit exact blocking needs |

Leaning toward Splink for the blocking+scoring core, with a custom feature layer for closed-vocabulary suffix substitution and numeric-token address features.

---

## 7. Refinements from a follow-up summary document

The user later uploaded a summary document (`entity_resolution_chat_summary.md`) covering the same pipeline, with two additions worth calling out:

**F₀.₅ as the target metric** — gives more weight to precision than recall, appropriate here because a false merge (claiming two different businesses are the same entity) is a costlier error than missing a true match. This should be checked against whatever scoring rubric the actual task specifies — if the real grading uses F1 or per-entity Jaccard instead, threshold tuning should optimize for that metric instead; the mechanics (sweep threshold, plot precision/recall, pick the point maximizing the target metric) stay the same regardless.

**Blocking-recall diagnostic table** — build a dataframe with one row per true match pair and one boolean column per blocking method (PIN, n-gram, token, locality, etc.), then compute each method's marginal contribution: how many true matches it finds, how many *new* true matches it uniquely contributes, and how many extra candidates it creates. This turns "is this blocking pass worth its candidate-volume cost" into a measurable decision rather than a guess, and avoids the anti-pattern of blindly OR-ing together many blocking rules until candidate volume balloons.

Key principle reiterated throughout: **optimize blocking for high recall first** (don't lose true matches — the classifier can never recover a match blocking throws away), **then optimize the ML classifier + threshold for high precision** (avoid false merges). This recall-then-precision separation across the two stages is the central design principle of the whole pipeline.

---

## 8. Next step offered

Build a runnable pipeline scaffolded around:
- a preprocessing module (lowercase/strip + closed-vocab suffix substitution)
- a blocking module (n-gram/LSH via `datasketch`, phonetic via `jellyfish`, PIN/token/locality blocks, country partition) with a recall-diagnostic report
- a feature module (name/address/country similarity features listed above)
- a training + threshold-tuning script against `train_ground_truth.tsv`, optimizing F₀.₅ (swappable for the actual target metric)
- an inference script producing the final `source1_entity_id → matched_entity_ids` submission, guaranteed to include every S1 entity

Pending: the actual `.tsv` files, to either build the scaffold now for later use or wire it directly to the real data once uploaded.
