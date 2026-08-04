"""
triplet_sampling.py — Task 5 (spec §6)

Builds (anchor, positive, negative) triplets from a fold's train split.

Sampling rules (spec §6):
  - Anchors: reference-bank members + all positives in the fold.
  - Positives: any other positive instance, including cross-subclass pairs
    (B1 vs B3) — intentional, this is the generalization signal.
  - Negatives: majority hard negatives (Pfam CL0381 look-alikes), minority
    easy negatives (unrelated folds), with semi-hard mining once embeddings
    exist.

Two entry points:
  - `random_triplets(...)` — used for the first epoch / warm start, before
    any embeddings exist to mine against.
  - `semi_hard_triplets(...)` — used every epoch after that; requires the
    current embedding of every candidate instance (computed by the training
    loop each epoch, or every N steps for efficiency — see train.py).

Semi-hard definition (standard FaceNet-style): for anchor a and positive p,
select negatives n from the label!=positive pool such that
    d(a, p) < d(a, n) < d(a, p) + margin
i.e. violates the margin but isn't the *easiest* negative to push away
(that would waste gradient on already-well-separated pairs) nor the
hardest (which can destabilize training early on).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Triplet:
    anchor_id: str
    positive_id: str
    negative_id: str
    negative_kind: str  # "hard" | "easy"


def random_triplets(
    positive_ids: list[str],
    hard_negative_ids: list[str],
    easy_negative_ids: list[str],
    n_triplets: int,
    hard_negative_fraction: float = 0.8,
    seed: int = 0,
) -> list[Triplet]:
    rng = random.Random(seed)
    triplets = []
    for _ in range(n_triplets):
        anchor, positive = rng.sample(positive_ids, 2)
        use_hard = rng.random() < hard_negative_fraction and len(hard_negative_ids) > 0
        neg_pool = hard_negative_ids if use_hard else easy_negative_ids
        if not neg_pool:
            neg_pool = hard_negative_ids or easy_negative_ids
        negative = rng.choice(neg_pool)
        triplets.append(Triplet(anchor, positive, negative, "hard" if use_hard else "easy"))
    return triplets


def semi_hard_triplets(
    positive_ids: list[str],
    negative_ids: list[str],  # hard + easy pooled together; kind tracked via negative_kind_lookup
    negative_kind_lookup: dict[str, str],
    embeddings: dict[str, np.ndarray],
    margin: float,
    n_triplets: int,
    hard_negative_fraction: float = 0.8,
    seed: int = 0,
) -> list[Triplet]:
    """
    embeddings: {instance_id: embedding vector}, must cover every id in
    positive_ids + negative_ids. Distance metric = Euclidean (matches
    model.py's default; swap consistently if cosine is used instead).
    """
    rng = random.Random(seed)
    triplets = []
    hard_pool = [n for n in negative_ids if negative_kind_lookup.get(n) == "hard"]
    easy_pool = [n for n in negative_ids if negative_kind_lookup.get(n) == "easy"]

    attempts_per_triplet = 20  # cap re-tries when no semi-hard negative exists in the sampled subset

    for _ in range(n_triplets):
        anchor, positive = rng.sample(positive_ids, 2)
        d_ap = float(np.linalg.norm(embeddings[anchor] - embeddings[positive]))

        prefer_hard = rng.random() < hard_negative_fraction and hard_pool
        pool = hard_pool if prefer_hard else (easy_pool or hard_pool)

        chosen_negative = None
        for _ in range(attempts_per_triplet):
            candidate = rng.choice(pool)
            d_an = float(np.linalg.norm(embeddings[anchor] - embeddings[candidate]))
            if d_ap < d_an < d_ap + margin:
                chosen_negative = candidate
                break
        if chosen_negative is None:
            # fall back to hardest available negative in the sampled pool
            # (small subset, not full corpus — cheap) rather than dropping the triplet
            chosen_negative = min(pool, key=lambda c: np.linalg.norm(embeddings[anchor] - embeddings[c]))

        kind = negative_kind_lookup.get(chosen_negative, "hard")
        triplets.append(Triplet(anchor, positive, chosen_negative, kind))

    return triplets
