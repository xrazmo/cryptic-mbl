"""Reproducible adapter for the published PLM-ARG checkpoints.

The upstream repository has no explicit software license as of the pinned
revision used by this project, so neither its source nor checkpoints are
vendored here.  This adapter accepts a user-supplied checkout/model paths,
records their hashes, and independently reproduces the published inference:
ESM-1b layer 32, the upstream 200-token truncation, mean pooling, then the
released binary ARG and multi-label resistance-category classifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    current_id = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(chunks)))
            current_id, chunks = line[1:].split()[0], []
        elif current_id is not None:
            chunks.append(line.strip().upper())
    if current_id is not None:
        records.append((current_id, "".join(chunks)))
    if not records or len({sid for sid, _seq in records}) != len(records):
        raise ValueError("FASTA must contain unique, non-empty records")
    return records


def embed(records: list[tuple[str, str]], model_path: Path, batch_size: int) -> np.ndarray:
    import torch
    from esm.pretrained import load_model_and_alphabet_local

    # fair-esm 2.0 predates PyTorch's weights_only=True default. The official
    # ESM checkpoint contains an argparse.Namespace in addition to tensors.
    # This adapter is explicitly for a trusted, user-supplied upstream model;
    # restore the old loader semantics only for this call.
    original_torch_load = torch.load
    def legacy_checkpoint_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)
    torch.load = legacy_checkpoint_load
    try:
        model, alphabet = load_model_and_alphabet_local(str(model_path))
    finally:
        torch.load = original_torch_load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    converter = alphabet.get_batch_converter()
    output = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            normalized = [(sid, "".join(aa if aa in "GAVLIPFYW RSTCMNQDEKH".replace(" ", "") else "X"
                                         for aa in sequence)) for sid, sequence in batch]
            _labels, strings, tokens = converter(normalized)
            # Exact upstream behavior: 200 tokens including BOS, hence at most
            # 199 residues for long proteins (not 200 residues plus BOS/EOS).
            tokens = tokens[:, :200].to(device)
            representations = model(tokens, repr_layers=[32], return_contacts=False)["representations"][32]
            for i, sequence in enumerate(strings):
                end = min(len(sequence) + 1, representations.shape[1])
                output.append(representations[i, 1:end].mean(0).cpu().numpy())
    return np.asarray(output, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--esm1b-model", required=True, type=Path)
    parser.add_argument("--arg-model", required=True, type=Path)
    parser.add_argument("--category-model", required=True, type=Path)
    parser.add_argument("--category-index", required=True, type=Path)
    parser.add_argument("--upstream-checkout", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--arg-threshold", type=float, default=0.5)
    parser.add_argument("--category-threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import joblib
    import sklearn
    import torch
    import xgboost

    records = read_fasta(args.fasta)
    embeddings = embed(records, args.esm1b_model, args.batch_size)
    arg_model = joblib.load(args.arg_model)
    category_model = joblib.load(args.category_model)
    categories = [line.strip() for line in args.category_index.read_text().splitlines() if line.strip()]
    if "beta-lactam" not in categories:
        raise ValueError("PLM-ARG category index has no beta-lactam category")

    arg_probability = arg_model.predict_proba(embeddings)[:, 1]
    category_probabilities = np.zeros((len(records), len(categories)), dtype=float)
    arg_indices = np.flatnonzero(arg_probability > args.arg_threshold)
    if len(arg_indices):
        predictions = category_model.predict_proba(embeddings[arg_indices])
        if len(predictions) > len(categories):
            raise ValueError(
                f"category model returned {len(predictions)} outputs for {len(categories)} categories"
            )
        # The released checkpoint has 20 estimators while Category_Index.csv
        # lists 21 rows; the upstream code likewise fills the first 20 and
        # leaves the trailing "others" column at zero.
        for category_index, matrix in enumerate(predictions):
            category_probabilities[arg_indices, category_index] = matrix[:, 1]

    beta_index = categories.index("beta-lactam")
    per_example = {}
    for i, (sid, sequence) in enumerate(records):
        predicted_arg = bool(arg_probability[i] > args.arg_threshold)
        beta_probability = float(category_probabilities[i, beta_index])
        per_example[sid] = {
            "sequence_length": len(sequence),
            "embedded_residues": min(len(sequence), 199),
            "arg_probability": float(arg_probability[i]),
            "predicted_arg": predicted_arg,
            "beta_lactam_probability": beta_probability,
            "predicted_beta_lactam": predicted_arg and beta_probability >= args.category_threshold,
        }

    output = {
        "schema_version": 1,
        "method": "PLM-ARG released ESM-1b/XGBoost checkpoints",
        "upstream_url": "https://github.com/Junwu302/PLM-ARG",
        "upstream_revision": git_revision(args.upstream_checkout),
        "license_note": (
            "No explicit upstream license was found at the pinned revision; source and "
            "checkpoints are therefore external inputs and are not redistributed here."
        ),
        "runtime_compatibility_note": (
            "The released pickle checkpoints were loaded in the current project runtime, "
            "not the upstream Python 3.7 / scikit-learn 1.0.2 / XGBoost 1.6.1 environment. "
            "Probabilities are recorded for comparison but should be treated as a "
            "compatibility reproduction unless independently matched in that legacy environment."
        ),
        "runtime_versions": {"torch": torch.__version__, "scikit_learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "joblib": joblib.__version__,
                             "numpy": np.__version__},
        "model_hashes": {
            "esm1b": sha256(args.esm1b_model), "arg": sha256(args.arg_model),
            "category": sha256(args.category_model), "category_index": sha256(args.category_index),
        },
        "fasta_sha256": sha256(args.fasta),
        "embedding": {"model": "esm1b_t33_650M_UR50S", "layer": 32,
                      "maximum_total_tokens": 200, "maximum_residues": 199,
                      "pooling": "mean"},
        "thresholds": {"arg_probability": args.arg_threshold,
                       "category_probability": args.category_threshold},
        "n_sequences": len(records),
        "n_predicted_arg": sum(row["predicted_arg"] for row in per_example.values()),
        "n_predicted_beta_lactam": sum(row["predicted_beta_lactam"] for row in per_example.values()),
        "per_example": per_example,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
