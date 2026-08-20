# Coordination fingerprint findings (Structural V2, Part 1)

## Scientific statement

Corrected first-shell coordination features strongly identify canonical
DCH-containing B1/B2 MBLs, but contain no useful information for
separating B3 MBLs from the tested metallo-hydrolase hard negatives. B3
discrimination must therefore use structural context beyond the
coordination shell. This is a valuable negative result, not a failure.

## B1/B2: operationally strong, not "solved"

Rule: `donor_s_count >= 1` (a coordinating cysteine present).

| panel | sensitivity | specificity | balanced accuracy |
|---|---|---|---|
| B1_B2_transfer (116 test positives) | 0.948 (110/116) | 0.983 | 0.966 |
| B3_transfer | 0.000 | 0.983 | 0.491 |
| remote_outlier (n=4) | 0.000 | 0.880 | 0.440 |

By-subclass S-donor prevalence (all labeled positives): B1 92.9%, B2
100% (n=3), B3 0.0% (0/27), unclassified 75%. Matches known MBL
active-site chemistry: B1/B2 canonically use a Cys ligand (the DCH
site), B3 canonically doesn't.

**Qualifications, load-bearing, not footnotes:**
- The rule was selected by inspecting the complete labeled corpus
  (by-subclass S-donor fractions), not validated on a held-out split
  the way a trained classifier would be. There is no free parameter
  (`>=1` is the only sensible threshold for a presence/absence count),
  but it has not been prospectively validated on genuinely new data.
- Misses both nominal B1 positives in the remote_outlier panel
  (`AAF94716.1`, `MBS5055441.1`, both donor_s_count=0.0).
- Specificity falls to 0.880 on the phosphodiesterase-heavy
  remote_outlier panel (28 false positives / 233 negatives).
- Six B1_B2_transfer B1 positives are missed:
  `ACB54703.1, AEX08599.1, AYF56302.1, IMP-1, QTG68658.1, VIM-2`.

Correct framing: canonical B1/B2 DCH-site detection is operationally
strong on the dominant component. Prospective (metagenome atlas)
validation is still required.

## B3: a real ceiling, not unfinished feature engineering

Every one of the 24 fingerprint features is statistically
indistinguishable between B3 positives and each hard-negative family
(glyoxalase-II, RNase Z, phosphodiesterase, lactonase) -- medians
overlap on coordination number, donor N/O counts, all donor-metal
distances, all angle statistics, bond valence, H-bond count, SASA, and
metal-metal distance. B3 and these hard negatives share a common
metallo-hydrolase-fold zinc-binding scaffold; their functional
differences live in substrate-pocket shape and second-shell residues,
which a first-shell coordination fingerprint does not measure by
construction.

One methodological caveat on the negative result's precision: angle and
template-deviation features pool donors from either accepted metal site
by nearest-distance, but always compute angles at the PRIMARY site's
vertex (coordination_fingerprint.py's compute_fingerprint) -- a
dinuclear site's second-shell geometry is never separately resolved.
This establishes a ceiling for the present *global* fingerprint, not a
proof that no possible site-resolved angular feature could help. The
biological overlap on every other feature is convincing enough that
fixing this specific gap just to continue first-shell feature hunting
for B3 is not judged worthwhile -- B3 discrimination needs a
structurally different kind of input (pocket shape, second-shell
residues), not a repaired version of this one.

## Decision

Stop first-shell fingerprint work for B3. Proceed with one bounded
outer-pocket-encoder experiment (frozen pretrained geometric encoder +
small calibrated head, B3-only) per the agreed go/no-go plan; freeze or
abandon based on its result rather than iterating indefinitely. Keep
B1/B2 (DCH rule), B3 (encoder if it passes, else ESM2), and
sequence (mean ESM2) as separate, un-fused scores -- do not force B1/B2
and B3 through one universal classifier, which is what caused the
subclass-specific signal to disappear in the earlier flat/branched GNN
results.
