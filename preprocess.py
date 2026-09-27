"""
Preprocessing module for Entity Resolution pipeline.

Design philosophy: LIGHT preprocessing only.
- We clean what's safe to clean (accents, case, whitespace, punctuation noise)
- We normalize a small, curated set of legal suffixes
- We do NOT attempt: full address parsing, transliteration, aggressive rewriting
- The ML classifier handles the rest via similarity features

This module produces cleaned DataFrames that downstream blocking and
similarity stages consume.

Uses multiprocessing to parallelize across CPU cores for large datasets.
"""

import re
import os
import time
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
from unidecode import unidecode


# ---------------------------------------------------------------------------
# Legal suffix normalization table (small, curated, low-risk)
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    # Corporations
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "incorp": "incorporated",
    # Limited
    "ltd": "limited",
    "ltd.": "limited",
    "ltda": "limited",
    "ltda.": "limited",
    # Private
    "pvt": "private",
    "pvt.": "private",
    # LLC variants
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "l l c": "llc",
    # LLP variants
    "l.l.p.": "llp",
    "l.l.p": "llp",
    # Partnership
    "ptnrs": "partners",
    "ptnr": "partner",
    # Company
    "co": "company",
    "co.": "company",
    # And
    "&": "and",
    "+": "and",
}

# Common address abbreviations (small, safe set)
ADDRESS_ABBREV_MAP = {
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "rd": "road",
    "rd.": "road",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "cir": "circle",
    "cir.": "circle",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "hwy": "highway",
    "hwy.": "highway",
    "ter": "terrace",
    "terr": "terrace",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
}


# ---------------------------------------------------------------------------
# Null / missing value markers found in the data
# ---------------------------------------------------------------------------
NULL_MARKERS = {"nan", "null", "<null>", "none", "n/a", "na", "", "-", "--"}


def _is_null(val) -> bool:
    """Check if a value is effectively null/missing."""
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in NULL_MARKERS:
        return True
    return False


# ---------------------------------------------------------------------------
# Core text cleaning functions
# ---------------------------------------------------------------------------

def clean_text_basic(text: str) -> str:
    """
    Basic text cleaning applied to ALL fields:
    - Strip whitespace
    - Lowercase
    - Remove accents/diacritics (é → e, ó → o)
      NOTE: unidecode also transliterates non-Latin scripts (Hindi, Tamil)
      to ASCII — this is intentional as it helps matching across scripts
    - Collapse multiple spaces
    """
    if _is_null(text):
        return ""

    text = str(text).strip()

    # Remove accents and transliterate non-Latin scripts
    text = unidecode(text)

    # Lowercase
    text = text.lower()

    # Remove the common [[...]] and [...] wrappers seen in S3 data
    text = re.sub(r'\[\[(.+?)\]\]', r'\1', text)
    text = re.sub(r'\[(.+?)\]', r'\1', text)

    # Collapse multiple spaces/tabs
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_business_name(name) -> str:
    """
    Clean business name with light normalization:
    1. Basic cleaning (lowercase, accents, whitespace)
    2. Remove D.B.A. prefixes
    3. Remove phone numbers embedded in names
    4. Remove .com/.org/.net domain suffixes (keep the name part)
    5. Normalize legal suffixes via lookup table
    6. Remove stray punctuation noise
    """
    if _is_null(name):
        return ""

    text = clean_text_basic(name)
    if not text:
        return ""

    # Strip leading dashes/hyphens (e.g., "-- Holloway Peak")
    text = re.sub(r'^[\-\s]+', '', text)

    # Remove D.B.A. / DBA prefixes and the trade name before them
    # "korbrixx d.b.a. obsidian, llc" → "obsidian, llc"
    text = re.sub(r'^.*?\bd\.?b\.?a\.?\b\s*', '', text, flags=re.IGNORECASE)

    # Remove phone numbers (sequences of 7+ digits, possibly with hyphens)
    text = re.sub(r'[\s\-]*\b\d[\d\-]{6,}\b', '', text)

    # Remove .com / .org / .net / .io domain endings
    text = re.sub(r'\.(?:com|org|net|io|co)\b', '', text)

    # Normalize legal suffixes: split into tokens, replace known ones
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        clean_token = token.strip(".,;:-'\"()")
        lookup = LEGAL_SUFFIX_MAP.get(clean_token, None)
        if lookup is not None:
            normalized_tokens.append(lookup)
        else:
            normalized_tokens.append(token)

    text = ' '.join(normalized_tokens)

    # Remove stray punctuation (keep alphanumeric, spaces, hyphens, apostrophes)
    text = re.sub(r"[^\w\s\-']", ' ', text)

    # Collapse spaces again after removals
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_business_address(address) -> str:
    """
    Clean business address with light normalization:
    1. Basic cleaning (lowercase, accents, whitespace)
    2. Remove <NULL>, null markers within address
    3. Normalize common street abbreviations
    4. Remove leading zeros from building numbers
    5. Keep everything else as-is (no address parsing!)
    """
    if _is_null(address):
        return ""

    text = clean_text_basic(address)
    if not text:
        return ""

    # Remove embedded null markers
    text = re.sub(r'\b(?:null|<null>|none|n/a|nan)\b', '', text, flags=re.IGNORECASE)

    # Remove leading zeros in building/unit numbers (AF-0684 → AF-684)
    text = re.sub(r'\b0+(\d+)', r'\1', text)

    # Normalize address abbreviations: tokenize, replace, rejoin
    tokens = text.split()
    normalized = []
    for token in tokens:
        clean_tok = token.strip(".,;:-")
        lookup = ADDRESS_ABBREV_MAP.get(clean_tok, None)
        if lookup is not None:
            normalized.append(lookup)
        else:
            normalized.append(token)
    text = ' '.join(normalized)

    # Remove stray punctuation but keep # (building), / (unit), - (ranges)
    text = re.sub(r"[^\w\s\-#/.,']", ' ', text)

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove trailing/leading commas and spaces
    text = text.strip(' ,')

    return text


def clean_country(country) -> str:
    """
    Clean country: just lowercase and strip.
    NEVER map to a fixed set — must work for unseen countries (France).
    """
    if _is_null(country):
        return ""
    return str(country).strip().lower()


def extract_numeric_tokens(text: str) -> str:
    """
    Extract all numeric tokens from a string.
    Useful for address matching (building numbers, PINs, etc.)
    Returns space-separated numbers.
    """
    if not text:
        return ""
    nums = re.findall(r'\b\d+\b', text)
    return ' '.join(nums)


def extract_pin_zip(address: str) -> str:
    """
    Try to extract PIN/ZIP code from an address.
    - India PIN: 6 digits (standalone)
    - US ZIP: 5 digits appearing after a comma or near end of string
      (to avoid treating street numbers like '17560' as ZIPs)
    Returns the code as a string, or empty string if not found.
    """
    if not address:
        return ""

    # India PIN: 6 consecutive digits (unambiguous — street numbers are rarely 6 digits)
    pin_match = re.findall(r'\b\d{6}\b', address)
    if pin_match:
        return pin_match[0]

    # US ZIP: 5 digits that appear after a comma/space near the end,
    # or as the last token — avoids grabbing street numbers at the start
    # Look for patterns like ", 12345" or state abbreviation followed by 5 digits
    zip_match = re.search(r'(?:,\s*|\b[a-z]{2}\s+)(\d{5})(?:-\d{4})?\s*$', address)
    if zip_match:
        return zip_match.group(1)

    # Fallback: 5-digit number after a comma anywhere in the string
    zip_match2 = re.search(r',\s*(\d{5})(?:-\d{4})?\b', address)
    if zip_match2:
        return zip_match2.group(1)

    return ""


def extract_name_tokens_sorted(name: str) -> str:
    """
    Extract alphabetic tokens from name, sort them.
    Useful for order-independent blocking.
    """
    if not name:
        return ""
    # Remove common legal suffixes before sorting (they add noise)
    skip = {"llc", "inc", "incorporated", "limited", "private", "corporation",
            "company", "llp", "partners", "partner", "corp", "ltd", "pvt",
            "center", "services", "group", "enterprises", "solutions"}
    tokens = [t for t in name.split() if t.isalpha() and t not in skip and len(t) > 1]
    return ' '.join(sorted(tokens))


# ---------------------------------------------------------------------------
# Row-level processing function (for multiprocessing)
# ---------------------------------------------------------------------------

def _process_row(row_tuple):
    """
    Process a single row. Takes (entity_id, business_name, business_address, country)
    and returns (entity_id, name_clean, addr_clean, country_clean,
                 name_tokens_sorted, addr_numbers, pin_zip).

    Designed to be called via multiprocessing Pool.map().
    """
    entity_id, bname, baddr, country = row_tuple

    nc = clean_business_name(bname)
    ac = clean_business_address(baddr)
    cc = clean_country(country)
    nts = extract_name_tokens_sorted(nc)
    an = extract_numeric_tokens(ac)
    pz = extract_pin_zip(ac)

    return (entity_id, bname, baddr, country, nc, ac, cc, nts, an, pz)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame, source_label: str = "",
                         n_workers: int = None) -> pd.DataFrame:
    """
    Preprocess a source DataFrame using multiprocessing.

    Input columns: entity_id, business_name, business_address, country
    Added columns:
        - name_clean: cleaned business name
        - addr_clean: cleaned business address
        - country_clean: cleaned country
        - name_tokens_sorted: sorted alpha tokens from name (for blocking)
        - addr_numbers: numeric tokens from address (for blocking/features)
        - pin_zip: extracted PIN/ZIP code (for blocking/features)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # leave one core free

    print(f"  Preprocessing {source_label}: {len(df):,} records using {n_workers} workers...")

    # Prepare row tuples for multiprocessing
    rows = list(zip(
        df['entity_id'].values,
        df['business_name'].fillna('').values,
        df['business_address'].fillna('').values,
        df['country'].fillna('').values,
    ))

    # Process in parallel
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_process_row, rows, chunksize=5000)
    elapsed = time.time() - t0

    # Rebuild DataFrame from results
    out = pd.DataFrame(results, columns=[
        'entity_id', 'business_name', 'business_address', 'country',
        'name_clean', 'addr_clean', 'country_clean',
        'name_tokens_sorted', 'addr_numbers', 'pin_zip'
    ])

    # Stats
    n_empty_name = (out['name_clean'] == '').sum()
    n_empty_addr = (out['addr_clean'] == '').sum()
    n_empty_pin = (out['pin_zip'] == '').sum()
    rate = len(df) / elapsed if elapsed > 0 else 0
    print(f"    Done in {elapsed:.1f}s ({rate:,.0f} records/sec)")
    print(f"    Empty names: {n_empty_name:,} ({100*n_empty_name/len(out):.1f}%)")
    print(f"    Empty addresses: {n_empty_addr:,} ({100*n_empty_addr/len(out):.1f}%)")
    print(f"    Missing PIN/ZIP: {n_empty_pin:,} ({100*n_empty_pin/len(out):.1f}%)")

    return out


# ---------------------------------------------------------------------------
# CLI: preprocess all source files and save
# ---------------------------------------------------------------------------

def main():
    import shutil

    parser = argparse.ArgumentParser(description="Preprocess entity resolution source files")
    parser.add_argument("--data-dir", default="6ab10eb3b23ba_student_resource/student_resource/dataset",
                        help="Root dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"],
                        help="Which split to preprocess")
    parser.add_argument("--output-dir", default="preprocessed",
                        help="Where to save preprocessed .tsv files")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count - 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    total_t0 = time.time()

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        split_dir = os.path.join(args.data_dir, split)
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        for src_num in [1, 2, 3]:
            fname = f"{split}_source{src_num}.tsv"
            fpath = os.path.join(split_dir, fname)
            if not os.path.isfile(fpath):
                print(f"  Skipping {fname} (not found)")
                continue

            t0 = time.time()
            df = pd.read_csv(fpath, sep='\t')
            print(f"  Loaded {fname}: {len(df):,} rows in {time.time()-t0:.1f}s")

            df_clean = preprocess_dataframe(df, source_label=f"{split}/source{src_num}",
                                            n_workers=args.workers)

            t0 = time.time()
            out_path = os.path.join(out_dir, fname)
            df_clean.to_csv(out_path, sep='\t', index=False)
            print(f"    Saved -> {out_path} in {time.time()-t0:.1f}s")

        # Copy ground truth as-is for train split
        if split == "train":
            gt_src = os.path.join(split_dir, "train_ground_truth.tsv")
            gt_dst = os.path.join(out_dir, "train_ground_truth.tsv")
            if os.path.isfile(gt_src):
                shutil.copy2(gt_src, gt_dst)
                print(f"  Copied ground truth -> {gt_dst}")

    total_elapsed = time.time() - total_t0
    print(f"\nAll done in {total_elapsed/60:.1f} minutes!")
    print(f"Preprocessed files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
