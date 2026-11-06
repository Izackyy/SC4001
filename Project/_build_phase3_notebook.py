"""Create the inline Phase 3 data notebook; removed after generation."""

from pathlib import Path
import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell("""# Phase 3: external human kinases

Build an unlabelled protein pool from the already downloaded UniProt FASTA.
All functions are inline; assertions check each filter and the exported tokens.
No download is attempted. The query was
`(organism_id:9606) AND (reviewed:true) AND (xref:pfam-PF00069 OR xref:pfam-PF07714)`.
Most entries already occur in DAVIS or KIBA, which are human kinase panels.
Test sequences are used only to exclude overlap here, never as domain inputs.
"""),
    nbf.v4.new_code_cell('''from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
FASTA = ROOT / "data/raw/uniprot/human_kinases.fasta"
PROCESSED = ROOT / "data/processed"
if not FASTA.is_file():
    raise FileNotFoundError(
        f"Required local FASTA is missing: {FASTA}. Raw data is gitignored; "
        "place the downloaded UniProt human kinases file there before running."
    )
vocab = json.loads((PROCESSED / "vocab.json").read_text(encoding="utf-8"))
GZIP = {"method": "gzip", "mtime": 0}
'''),
    nbf.v4.new_markdown_cell("## 1. Parse the 474 reviewed proteins"),
    nbf.v4.new_code_cell('''def parse_fasta(path):
    """Parse FASTA records in file order, keyed by the first header token.

    Args:
        path: Local FASTA path; headers may contain a GN= gene name.

    Returns:
        DataFrame with kinase_id, gene, sequence and full sequence length.

    Raises:
        AssertionError: A header ID repeats or sequence precedes a header.
    """
    records = {}
    record = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                ident = line[1:].split()[0]
                assert ident not in records, ident
                gene = re.search(r"\\bGN=(\\S+)", line)
                record = {"kinase_id": ident, "gene": gene.group(1) if gene else "",
                          "sequence": ""}
                records[ident] = record
            else:
                assert record is not None, "Sequence precedes a FASTA header"
                record["sequence"] += line
    table = pd.DataFrame(records.values())
    table["length"] = table.sequence.str.len()
    return table


pool = parse_fasta(FASTA)
assert len(pool) == 474
assert pool.kinase_id.is_unique and (pool.length > 0).all()
funnel = [{"stage": "Parsed", "removed": 0, "remaining": len(pool)}]
print(f"Parsed {len(pool)} records")
'''),
    nbf.v4.new_markdown_cell("## 2. Check the existing protein alphabet"),
    nbf.v4.new_code_cell('''alphabet = set(vocab["protein"])
invalid = pd.Series([bool(set(sequence) - alphabet) for sequence in pool.sequence],
                    index=pool.index)
assert invalid.sum() == 0
pool = pool.loc[~invalid].copy()
funnel.append({"stage": "Protein alphabet", "removed": int(invalid.sum()), "remaining": len(pool)})
'''),
    nbf.v4.new_markdown_cell("## 3. Exclude exact DAVIS and KIBA sequences"),
    nbf.v4.new_code_cell('''targets, pairs = {}, {}
for dataset in ("davis", "kiba"):
    targets[dataset] = pd.read_csv(PROCESSED / f"{dataset}_targets.csv.gz",
                                   dtype={"target_id": str, "merged_ids": str})
    pairs[dataset] = pd.read_csv(PROCESSED / f"{dataset}_pairs.csv.gz",
                                 usecols=["target_id", "split_cold_target"],
                                 dtype={"target_id": str})
benchmark_sequences = set().union(*(set(table.sequence) for table in targets.values()))
exact = pool.sequence.isin(benchmark_sequences)
assert exact.sum() == 353
pool = pool.loc[~exact].copy()
assert len(pool) == 121
funnel.append({"stage": "Exact benchmark overlap", "removed": int(exact.sum()), "remaining": len(pool)})
'''),
    nbf.v4.new_markdown_cell("""## 4. Exclude near-duplicates of held-out test proteins

Use sets of all overlapping 5-mers from the full sequences. Jaccard similarity
is intersection size divided by union size. The maximum spans both datasets'
cold-target test sets; a value of 0.90 or above is excluded.
"""),
    nbf.v4.new_code_cell('''def kmers(sequence, k=5):
    """Return the distinct overlapping substrings of length k.

    Args:
        sequence: Full protein sequence.
        k: Substring width, default five.

    Returns:
        A set of sequence substrings.
    """
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def max_jaccard(sequence, reference_kmers):
    """Measure the greatest 5-mer Jaccard similarity to held-out references.

    Args:
        sequence: Full candidate protein sequence.
        reference_kmers: Nonempty list of 5-mer sets of test proteins.

    Returns:
        Maximum intersection-over-union similarity, between zero and one.
    """
    candidate = kmers(sequence)
    return max(len(candidate & reference) / len(candidate | reference)
               for reference in reference_kmers)


test_sequences = set()
for dataset in targets:
    test_ids = set(pairs[dataset].loc[pairs[dataset].split_cold_target == "test", "target_id"])
    test_sequences.update(targets[dataset].loc[targets[dataset].target_id.isin(test_ids), "sequence"])
test_kmers = [kmers(sequence) for sequence in sorted(test_sequences)]
pool["max_test_jaccard"] = [max_jaccard(sequence, test_kmers) for sequence in pool.sequence]
near = pool.max_test_jaccard >= 0.90
assert near.sum() == 3
assert set(pool.loc[near, "kinase_id"].str.split("|").str[1]) == {"P19525", "P29320", "Q9BZL6"}
display(pool.loc[near, ["kinase_id", "gene", "length", "max_test_jaccard"]])
pool = pool.loc[~near].copy()
assert len(pool) == 118
funnel.append({"stage": "Test 5-mer Jaccard >= 0.90", "removed": int(near.sum()), "remaining": len(pool)})
'''),
    nbf.v4.new_markdown_cell("""## 5. Remove giant multidomain proteins

Sequences longer than 3000 residues are excluded. Titin, obscurin, SPEG and
TRIO place their kinase domains beyond the retained prefix, making these
truncated fragments unrepresentative kinase inputs.
"""),
    nbf.v4.new_code_cell('''giant = pool.length > 3000
assert giant.sum() == 4
assert set(pool.loc[giant, "gene"]) == {"TTN", "OBSCN", "SPEG", "TRIO"}
display(pool.loc[giant, ["kinase_id", "gene", "length", "max_test_jaccard"]])
pool = pool.loc[~giant].reset_index(drop=True)
assert len(pool) == 114
assert pool.sequence.is_unique
funnel.append({"stage": "Length > 3000", "removed": int(giant.sum()), "remaining": len(pool)})
'''),
    nbf.v4.new_markdown_cell("## 6. Tokenise with Phase 1's vocabulary and lengths"),
    nbf.v4.new_code_cell('''def tokenize(strings, characters, max_len):
    """Encode prefixes as int16 rows, pad with zero and verify decoding.

    Args:
        strings: Protein sequences in CSV row order.
        characters: Existing vocabulary; token ID is list index plus one.
        max_len: Dataset-specific prefix length.

    Returns:
        Zero-padded int16 array with one row per sequence.

    Raises:
        AssertionError: A decoded prefix or its padding differs.
        KeyError: A sequence contains a residue outside the alphabet.
    """
    lookup = {char: index + 1 for index, char in enumerate(characters)}
    tokens = np.zeros((len(strings), max_len), dtype=np.int16)
    for index, sequence in enumerate(strings):
        prefix = sequence[:max_len]
        tokens[index, :len(prefix)] = [lookup[char] for char in prefix]
        assert "".join(characters[token - 1] for token in tokens[index] if token) == prefix
        assert not tokens[index, len(prefix):].any()
    return tokens


assert vocab["max_len"]["davis"]["protein"] == 1200
assert vocab["max_len"]["kiba"]["protein"] == 1000
tokens = {dataset: tokenize(pool.sequence, vocab["protein"], vocab["max_len"][dataset]["protein"])
          for dataset in ("davis", "kiba")}
'''),
    nbf.v4.new_markdown_cell("## 7. Deterministic exports and reload checks"),
    nbf.v4.new_code_cell('''columns = ["kinase_id", "gene", "sequence", "length", "max_test_jaccard"]
pool = pool[columns]
path = PROCESSED / "external_kinases.csv.gz"
pool.to_csv(path, index=False, compression=GZIP)
reloaded = pd.read_csv(path, keep_default_na=False, dtype={"kinase_id": str, "gene": str})
pd.testing.assert_frame_equal(reloaded, pool, check_exact=False, rtol=1e-12, atol=1e-12)
for dataset, array in tokens.items():
    path = PROCESSED / f"external_tokens_{dataset}.npz"
    # NumPy writes fixed ZIP member timestamps, so repeated saves are byte-identical.
    np.savez_compressed(path, prot_tokens=array)
    with np.load(path, allow_pickle=False) as saved:
        assert saved.files == ["prot_tokens"]
        assert saved["prot_tokens"].dtype == np.int16
        np.testing.assert_array_equal(saved["prot_tokens"], array)
        for sequence, row in zip(reloaded.sequence, saved["prot_tokens"]):
            decoded = "".join(vocab["protein"][token - 1] for token in row if token)
            assert decoded == sequence[:vocab["max_len"][dataset]["protein"]]
print("Reload verified: CSV values, token dtype, row alignment and decoded prefixes")
'''),
    nbf.v4.new_markdown_cell("## 8. Pool audit"),
    nbf.v4.new_code_cell('''print(pd.DataFrame(funnel).to_string(index=False))
print("\\nFinal pool length summary:")
print(pool.length.describe().to_string())
print(f"Maximum retained test Jaccard: {pool.max_test_jaccard.max():.6f}")
for dataset in tokens:
    limit = vocab["max_len"][dataset]["protein"]
    print(f"{dataset}: {len(pool)} sequences, {(pool.length > limit).sum()} truncated at {limit}")
'''),
    nbf.v4.new_markdown_cell("""## What Phase 3 reads

`data/processed/external_kinases.csv.gz` records the 114 new sequences, gene
names, full lengths and maximum test 5-mer Jaccard similarities.
`external_tokens_davis.npz` and `external_tokens_kiba.npz` each contain only
`prot_tokens` (int16), with rows aligned to that CSV and zero padding exactly
as in Phase 1. `src/train_dann.py` combines these unlabelled proteins with the
validation proteins for its target-domain stream. No test protein is a domain
training input; test references above serve only the exclusion audit.
"""),
]
book = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.write(book, Path("notebooks/03_external_kinases.ipynb"))
