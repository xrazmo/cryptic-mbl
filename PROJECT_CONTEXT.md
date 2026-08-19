# Cryptic MBL Discovery Project Context

Last updated: 2026-08-19  
Repository: `cryptic-mbl`  
Current branch at the time of writing: `feature/b1-structural-detector`

## Purpose of this document

This is a scientific and technical handoff for a future Codex thread. It records the biological objective, the reasoning behind the modeling strategy, the major audit findings, approaches that were tried or rejected, the current production state, and the questions that remain open. It is deliberately broader than a code-change log.

The most important high-level conclusion is:

> The broad mixed-subclass structural models did not beat sequence methods, but a new full-chain, metal-coordinate-independent detector now establishes a strong structural result for canonical B1 specifically: 109/110 sequence-remote B1 positives and 0/410 panel false positives, including all ten B1 examples missed by mean-ESM2 5-NN. It uses the complete six-donor 3D architecture rather than a labeled reference panel or cysteine presence alone. fARGene still detects every available known B1 in the principal panels, so structure-beyond-HMM discovery remains a prospective biological question rather than a completed claim.

Any future work must preserve the version boundary. The existing production
model is a frozen sequence-dominant retrieval/prioritization system. The new B1
pharmacophore is a separate structural evidence channel and has not been used
for Atlas screening or silently substituted into production.

## 1. Biological problem and intended use

The project aims to discover cryptic metallo-β-lactamases (MBLs), especially antibiotic-resistance MBLs (ARG MBLs), in very large environmental and metagenomic protein collections. The intended next search space is the ESM Metagenomic Atlas, on the order of tens of terabytes. The practical output is a ranked shortlist for deeper computational review and, ultimately, biochemical validation—not a definitive resistance annotation from a model score alone.

The scientific motivation is to go beyond conventional homology searches. HMMs, BLAST-like tools, and nearest-neighbor sequence retrieval are effective for finding relatives of known MBL families, but may preferentially rediscover more B1 homologues. A meaningful role for three-dimensional structure would be to recognize conserved catalytic organization despite extensive primary-sequence divergence, or to distinguish true MBL chemistry from proteins that are sequence- or fold-adjacent but catalytically different.

The desired discovery claim is therefore stronger than “detects proteins below a BLAST/MMseqs identity threshold.” A protein language model such as ESM-2 is also sequence-derived and can recognize remote evolutionary relationships with no conventional high-coverage alignment. The relevant scientific question is:

> Does explicit pocket structure or metal-site chemistry add reproducible information beyond a strong pretrained sequence representation, particularly on remote positives and confusing metallo-hydrolase negatives?

This is not equivalent to asking whether the model beats an older B1-focused HMM.

### MBL subclass considerations

The labeled positive set is strongly imbalanced by subclass:

- B1: 112 examples
- B3: 27 examples
- B2: 3 examples
- Unclassified positives: 4 examples

B1 dominates both the dataset and much of the prior literature. Berglund et al. (2017), for example, primarily pursued B1 MBLs and faced a related positive-diversity limitation. B1/B2 and B3 are biologically distinct enough that transfer direction matters. A model trained mostly on B1/B2 and tested on B3 is not the same problem as training on the small B3 group and testing on the large B1/B2 group. Performance asymmetry cannot automatically be attributed to a general “B3 biological problem”; it is confounded by training-positive count, component composition, metal-site organization, and family diversity.

With only three B2 examples, this project cannot make a reliable B2-specific generalization claim.

## 2. Dataset and resources

### Labeled corpus

The frozen corpus contains 1,077 structures:

- 146 positives
- 931 negatives
- 1,051 pockets originally centered from Metal3D output
- 26 pockets centered using a cavity fallback

Positive sources include CARD and published MBL collections, including Berglund et al. and Gudeta et al. Negatives were assembled primarily from UniProt-derived examples and include both structurally unrelated easy decoys and chemically/fold-related hard negatives.

Important hard-negative families include:

- glyoxalase II
- RNase Z
- lactonase
- phosphodiesterase

Other negative folds/families include alpha/beta hydrolases, Rossmann SDR proteins, thioredoxin-like folds, TIM barrels, lysozyme-like proteins, and globins.

The seven canonical named reference proteins are:

- NDM-1
- VIM-2
- IMP-1
- CphA
- Sfh-I
- FEZ-1
- L1

Five of these fall in the large B1/B2 sequence component; FEZ-1 and L1 fall in the B3 component. They are useful operational anchors, but they are not an independent external test set.

### Provenance confounding

The confidence tier is a perfect proxy for class in the current corpus:

- positives from CARD/Berglund/Gudeta are assigned tier 3, and canonical references tier 1
- negatives from UniProt are assigned tier 4

Tier was created from source provenance in `catalog_to_manifest.py`, not from an independently measured confidence variable. It is not included directly as a node feature, but it affects triplet-loss weighting. Even if tier itself is removed, source-correlated artifacts may remain in sequences, structures, prediction quality, annotations, pocket success, or preprocessing.

Therefore, performance may partly reflect CARD/published-positive versus UniProt-negative provenance rather than biology. This is a real limitation and must appear in manuscripts and model cards.

### Relevant repository artifacts

- `configs/manifest.csv`: labeled manifest
- `full_structure_catalog.csv`: full source catalog used for production freeze
- `data/domain_pdbs/`: cleaned dominant-chain/full-chain structures used for structural comparison
- `data/pockets/`: current pocket archives; these are affected by the Metal3D-site averaging and chain-identity issues described below
- `data/esm2_embeddings/`: frozen per-residue ESM-2 embeddings
- `data/split_graph.json`: audited sequence/structure similarity graph
- `data/challenge_splits.json`: component challenge panels and LONO configurations
- `reports/split_graph_audit.json`: exhaustive-search and graph audit
- `reports/challenge_split_audit.json`: panel composition, hashes, and reference policy
- `reports/production_model_manifest.json`: frozen production model design and hashes
- `models/production/`: eight final production checkpoints
- `Production_Model_Design_and_Evaluation.docx`: publication-oriented description of V1, written before the later structural-input root-cause finding

## 3. Original AI/deep-learning strategy

### Intended representation

Each putative MBL pocket is represented as a residue graph. Current graph features include:

- amino-acid identity
- residue physicochemical descriptors
- residue-centroid distance to the predicted metal
- canonical donor-atom distance and a coarse coordination flag
- backbone dihedral features where available
- pocket SASA
- radial shell relative to the proposed metal site
- frozen per-residue ESM-2 vectors (1,280 dimensions)

Edges combine spatial proximity with explicit chain-sequence adjacency. The adjacency flag was added so the model can distinguish residues connected along the same loop/backbone from unrelated spatial contacts.

### Current encoder

The production architecture, `PocketEncoder` / `new_graph_flat`, uses four distance-aware message-passing layers and global mean pooling to produce a unit-normalized 128-dimensional embedding. It is invariant to rotation and translation because it uses scalar pairwise distances, but it is not a true SE(3)-equivariant architecture and does not preserve directional/angular information through the network.

The code itself calls this a placeholder for a more capable structural backbone. This qualification is scientifically important: the present GNN is not expected to resolve coordination orientation as effectively as GVP, GearNet-like geometric pretraining, e3nn, or the proposed EZSpecificity-style equivariant encoder.

### Training and decision rule

The model uses Siamese/triplet metric learning. Positives are pulled together and negatives pushed away, with hard-negative sampling and periodic semi-hard mining. Cross-subclass positive pairing encourages a common MBL embedding region rather than separate subclass classifiers.

The frozen production ensemble contains eight independently initialized models, each trained for 60 epochs on all 1,077 labeled structures (146 positives, 931 negatives). There was intentionally no validation carve and no validation-based checkpoint selection; `final.pt` is used for every seed. This avoids unstable model selection with too few independent positive components.

For production inference:

1. Each seed embeds the candidate in that seed's own latent space.
2. Five nearest neighbors are retrieved from the labeled embedding bank made by that same seed.
3. Each seed gives a label and a positive-neighbor fraction.
4. Seed labels are majority-voted; positive-neighbor fractions are averaged as a support score.
5. Latent coordinates are never averaged across seeds.
6. A separate five-nearest-neighbor score from mean-pooled ESM-2 is reported independently.

The last point is intentional. Fusing ESM-2 and GNN results into one opaque number would hide whether a candidate is supported by pretrained sequence resemblance, task-specific pocket learning, or both.

### Three different meanings of “reference panel/bank”

Future discussions must distinguish:

1. **Seven canonical named references:** biologically recognizable anchors such as NDM-1 and L1. These are not an external validation set because they belong to the major training components.
2. **Panel training partition:** the labeled examples available to a model or baseline in a particular held-out challenge panel.
3. **Production nearest-neighbor bank:** embeddings of the full labeled corpus used to assign a production candidate's k-NN label.

The production decision rule is a closed-set, prototype/retrieval-style rule. Even if the encoder learns useful structure, k-NN against the labeled bank asks whether a candidate lands near known labeled examples. This can bottleneck extrapolation: a genuinely novel structural solution may be far from all positive prototypes and be rejected. It also means that “model score” is not a direct learned probability of MBL catalysis.

This does not make the reference comparison useless—retrieval is transparent and practical—but it limits the generalizability claim. Future structural V2 evaluation should compare k-NN with a direct discriminative head, class prototypes, distance-to-positive versus distance-to-hard-negative calibration, and an explicit out-of-distribution/abstention rule. Any such alternative must be evaluated on the same frozen challenge panels, not selected after seeing Atlas candidates.

## 4. Why the original ESM-2 improvement was reinterpreted

The first ESM-2-enhanced results appeared dramatically better than the earlier chemistry-only `reachfeat` model. An audit then showed:

- a training-free five-nearest-neighbor classifier using mean ESM-2 embeddings exactly reproduced the trained model's confusion matrix
- BLASTP/MMseqs analysis showed heavy train/test sequence overlap in the old split
- 95% of old test positives had at least 40% identity to a training example
- named reference proteins such as NDM-1, VIM-2, and IMP-1 had 98–100% identity matches in training

The correct interpretation was therefore not “dramatic structural improvement.” It was excellent in-distribution family recognition driven mainly by pretrained sequence/evolutionary representation. At that stage, task-specific GNN training added no measurable value over raw ESM-2 neighbor lookup.

This audit changed the project from ordinary performance optimization to honest remote-generalization testing.

## 5. Similarity splitting: what failed and what replaced it

### Unimplemented documented filter

The original `clustering_split.py` documentation claimed an RMSD/identity post-filter with CLI options and a `filter_clusters()` function, but none of these existed. The script consumed raw Foldseek clustering output. Overnight runs therefore did not implement the stated leakage-control rule.

### Pocket-fragment RMSD is unsuitable as a grouping edge

When the documented pocket rule was implemented literally and transitively closed, it generated a giant structural component containing 926 of 1,077 structures, spanning all labels and biologically unrelated negative folds such as globins, TIM barrels, Rossmann SDR, and thioredoxin-like proteins.

This was not credible whole-fold homology. On approximately 40-residue pocket fragments, alignment tools can find short sub-alignments with deceptively low RMSD. OR-combining such noisy edges and applying connected-component closure creates dataset-spanning chains.

Consequences:

- raw pocket-fragment RMSD must not be used as a hard grouping criterion
- pocket similarity may remain a diagnostic
- structural redundancy should be measured on cleaned full chains/domains with length-normalized TM-score

### Audited split graph

The replacement pipeline exports:

- the exact ESM-visible dominant-chain sequence, truncated consistently with ESM-2 at 1,022 residues
- a genuinely untruncated cleaned dominant-chain/full-chain structure for structural comparison

All-vs-all searches are exhaustive:

- MMseqs2 uses exhaustive search
- Foldseek uses exhaustive search and exact TM-score
- result caps were raised and audited
- every full-chain query returned all 1,077 targets before thresholding

The current sequence grouping threshold is identity ≥0.30 with bidirectional coverage ≥0.80. The graph has 156 sequence components, with a largest component of 286. The positive examples occupy only six sequence components, of sizes 116, 26, 1, 1, 1, and 1.

Full-chain structural grouping showed that novel-fold cross-validation is not supported by this dataset:

- at TM-score ≥0.5, 144 of 146 positives lie in one structural component
- at TM-score ≥0.65, 142 of 146 lie in one component

This is biologically plausible because B1/B2/B3 MBLs share an overall fold/topology. Forcing “structure-remote” k-fold CV would either be impossible or scientifically misleading. This is a dataset limitation, not a splitting bug.

### Identity-threshold sensitivity

At fixed 80% coverage, positive component structure changes sharply between 40% and 50% identity. At 20–30%, most positives chain into only six positive-containing components; at 40% there are about ten; at 50% many more components appear. This threshold dependence must be reported rather than choosing a convenient cutoff after observing performance.

## 6. Honest validation design

Balanced random or ordinary stratified k-fold CV is not meaningful with only six positive sequence components and two dominant components. The project therefore uses component-level challenge panels.

### Main panels

1. **B1_B2_transfer**
   - Test: 116 positives, mostly B1 with three B2 and a few unclassified examples
   - Training contains only about 30 positives, dominated by B3 plus singleton positives
   - This is the hardest reliable positive-transfer panel and exposes small-positive-pool instability

2. **B3_transfer**
   - Test: 26 B3 positives
   - Training contains about 120 positives, mostly B1/B2
   - This asks whether a large B1/B2 training pool transfers to B3

3. **remote_outlier**
   - Test: four singleton positive components
   - Too small for a population sensitivity estimate
   - Must be reported as four case studies

All test examples have no train edge meeting identity ≥0.30 and bidirectional coverage ≥0.80. The similarity audit reports no leaking edges and retains input hashes.

### Leave-one-negative-family-out tests

Each LONO test holds out every component containing any member of the target negative family, with zero target-family examples left in training. The four named families are RNase Z, glyoxalase II, lactonase, and phosphodiesterase. These panels contain no positives and therefore measure specificity/false-positive rate only, not balanced accuracy.

Several negative families are themselves monolithic sequence components—for example, all 286 glyoxalase II negatives can move only as one block. This constrains panel balancing and means panel-level balanced accuracy should not be compared as if every panel had identical negative difficulty.

### Reference-anchored retrieval protocol

A separate operational protocol retains canonical references and both large positive components as anchors, then evaluates only the four reference-free singleton positives. This approximates candidate retrieval but is not a mechanism-generalization test and cannot yield a stable sensitivity estimate.

### Ensembling correction

An earlier evaluation averaged latent coordinates across independently trained seeds. That is invalid because independently learned latent axes are not aligned. Correct evaluation performs k-NN separately within each seed's own space and then majority-votes labels.

Correcting this bug revealed substantial seed instability, especially on B1/B2 transfer. The correction preserved the qualitative graph-feature improvement but reduced its apparent magnitude and removed false precision.

### The “zero identity” label was incorrect

`analyze_identity_stratified_sensitivity.py` called examples with `max_identity_at_80cov == 0` “no detectable hit.” This actually means no train alignment met the **80% bidirectional coverage** requirement. It does not mean no local sequence alignment or literally zero identity.

Every member of those bins had a local MMseqs hit at some coverage:

- B1/B2 bin: 110/110 had local hits; best local identities ranged approximately 0.538–1.0, median about 0.615
- B3 bin: 16/16 had local hits; best local identities ranged approximately 0.50–0.75, median about 0.666

Therefore, results in this bin may be described as **no qualifying high-coverage train alignment**, not “zero alignable sequence identity.” Any manuscript or plot using the latter phrase must be corrected.

## 7. Main empirical findings

### Training-free mean-ESM-2 baseline

On the audited challenge panels, mean-pooled ESM-2 five-nearest-neighbor retrieval achieved:

| Panel | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|
| B1/B2 transfer | 0.897 | 0.998 | 0.947 |
| B3 transfer | 0.692 | 0.990 | 0.841 |
| Remote outliers | 0.750 (3/4) | 0.961 | 0.856 |

This baseline uses no task-specific training but benefits from massive protein-language-model pretraining. It is stronger than conventional alignment and must not be described as equivalent to an HMM.

The baseline also exposed a severe specificity blind spot: when lactonases were absent from training/reference examples, specificity fell to 0.416 (59 false positives among 101 lactonases). RNase Z, glyoxalase II, and phosphodiesterase LONO specificity was much better. Lactonases require an explicit counter-screen or orthogonal evidence in Atlas triage.

### Original flat, enhanced graph, branched, and structure-only models

Correct per-seed vote-ensemble results were:

| Configuration | B1/B2 sens/spec/BA | B3 sens/spec/BA | Remote sens/spec/BA |
|---|---|---|---|
| Original flat | .034 / 1.000 / .517 | .962 / .993 / .977 | .500 / .983 / .741 |
| Enhanced graph flat (`new_graph_flat`) | .276 / 1.000 / .638 | .731 / 1.000 / .865 | .500 / .983 / .741 |
| Branched fused | .310 / 1.000 / .655 | .692 / 1.000 / .846 | .250 / .983 / .616 |
| Branched structure-only | .121 / 1.000 / .560 | .462 / .965 / .713 | .500 / .906 / .703 |

The explicit sequence-adjacency edges and radial-shell feature moved B1/B2 transfer off a near-total floor, so better graph context mattered. However, individual enhanced-flat seeds ranged from about 3% to 66% B1/B2 sensitivity, showing severe initialization/data scarcity instability.

The structure-only B3 result was reproducible across seeds, with sensitivity tightly around 0.42–0.50 and aggregate specificity 0.965. This is evidence that non-ESM structural/chemical features carry real predictive signal on B3. It is **not** evidence that this signal is competitive with ESM-2, that it generalizes to B1/B2, or that the production fused model actually relies on it.

The enhanced flat model modestly beat mean ESM-2 on B3 transfer but was dramatically worse on B1/B2 transfer. Thus task-specific learning helps in one transfer direction with many training positives, but is not a safe replacement for the sequence baseline in the main cryptic-discovery scenario.

### No-high-coverage-alignment subgroup

Using the corrected interpretation of the bin:

| Panel | n positives | Raw ESM-2 | Enhanced flat | Branched fused | Structure-only |
|---|---:|---:|---:|---:|---:|
| B1/B2 transfer | 110 | .91 | .29 | .33 | .13 |
| B3 transfer | 16 | .75 | .75 | .75 | .50 |

On B1/B2, all 14 structure-only true positives were also detected by ESM-2; it added no unique positive recovery. On all B3 positives, structure-only recovered three positives missed by ESM-2 but added ten false positives. A simple union improved sensitivity at a large specificity cost.

The correct next comparison is not unconstrained sensitivity. Structural complementarity must be assessed at matched specificity or within a fixed candidate-review budget. Useful endpoints include:

- incremental true-positive recovery among ESM-2 misses at ≥99% specificity
- precision/recall among the top N Atlas candidates
- enrichment over ESM-2 alone at equal screening cost
- unique recovery confirmed by biochemical assay

## 8. Architecture and feature approaches considered

### Flat ESM-2 + structure concatenation

The flat model concatenates 1,280 ESM-2 dimensions with roughly 40 identity/structural dimensions. It is simple and was selected for V1 because it had the most balanced proxy behavior among trained models. Its weakness is that ESM-2 can dominate the first layer by dimensionality and predictive ease. A good fused score does not prove structural use.

### Branched structure/ESM-2 fusion

A branched model was built with:

- separate structural and ESM-2 encoders
- independent normalization before fusion
- random ESM-2 modality dropout
- an auxiliary structure-only triplet loss

It slightly improved B1/B2 sensitivity over the enhanced flat model but regressed on the remote cases and did not consistently outperform across seeds. It was not selected as the production architecture. It remains a useful experimental design, especially after structural-input repair.

### Multiple discovery models

The project does not currently require an uncontrolled collection of models. The operational recommendation was one frozen GNN ensemble plus a separately reported raw-ESM-2 signal, with disagreement used as a triage feature. Adding many models would complicate calibration and invite post hoc selection.

A future structural V2 may justifiably become a third orthogonal score only if it demonstrates incremental recovery at matched specificity. Until then, “several models” is not a substitute for showing that structure adds information.

### Explicit coordination fingerprint

A 22-feature metal-centered descriptor was implemented, including:

- coordination number and donor element counts
- predicted metal–donor distances
- donor–metal–donor angles
- deviations from ideal geometry templates
- approximate bond-valence features
- second-shell polar-network density
- donor-residue SASA

Balanced gradient boosting and a prototype classifier gave weak results and recovered few ESM-2 misses. Initial donor coverage was also poor: only 113/145 positives had a canonical donor within 5 Å of the stored predicted metal, and only 59/145 within 2.8 Å.

These results are **not valid evidence that coordination geometry lacks discriminative value**, because the stored metal coordinate is often corrupted upstream. The fingerprint experiment must be repeated after the preprocessing repair.

Also note: the R0 constants 1.70/1.77/2.01 Å in the fingerprint are bond-valence parameters, not universal physical Zn–donor bond lengths. Actual Zn–N/O distances are commonly around 2.0–2.2 Å and Zn–S can be longer; 2.8 Å is a permissive search cutoff, not a textbook single bond length.

### Larger/pretrained geometric encoders

GearNet, GVP, e3nn, or an EZSpecificity-style equivariant backbone were considered because the current scalar-distance GNN discards orientation. They are deferred, not rejected. A larger encoder cannot rescue a physically wrong pocket center; it would likely learn to ignore geometry or memorize easier sequence/provenance cues.

The correct order is:

1. repair and validate metal-site/pocket preprocessing
2. repeat cheap interpretable structural baselines
3. establish incremental value at matched specificity
4. only then invest in a pretrained/equivariant encoder

## 9. Decisive structural-input root cause

The weak geometry input is not explained merely by noisy predicted apo structures. Two deterministic implementation problems were found.

### Metal3D candidate sites were averaged

`pocket_extraction.py` runs Metal3D with `--writeprobes` and `--maxp`, reads the probe PDB, and computes the mean coordinate of all returned probe atoms.

The bundled Metal3D behavior is different from what the wrapper assumed:

- `--maxp` prints/writes maximum-probability information separately
- it does not restrict `--writeprobes` to one top site
- `--writeprobes` can contain multiple clustered candidate sites
- the probe occupancy contains prediction probability

Averaging all sites creates a phantom point that may lie far from every real candidate metal site. It also destroys real mono- versus dinuclear geometry.

### Direct NDM-1 verification

On the exact original NDM-1 input used for extraction, Metal3D returned 30 candidate probes. The nearest predictions were approximately 0.457 Å and 0.730 Å from the two experimental Zn coordinates. Thus Metal3D itself localized the known site well.

The mean of all 30 probes was approximately 23.6–24.6 Å from the experimental Zn ions. The metal coordinate stored in `data/pockets/NDM-1.npz` was similarly about 23.6–24.6 Å away.

This proves that, at least for NDM-1, the wrapper—not Metal3D—destroyed a good prediction. It invalidates the interpretation that the 3.5–3.8 Å donor distances primarily reveal an unavoidable Metal3D or apo-structure ceiling.

### Chain/residue identity collision

Residues were keyed by integer `res_id` alone. Chain ID and insertion code were ignored in centroid construction and pocket selection. In multi-chain structures, repeated residue numbers caused atoms from different chains to be merged or selected together.

The stored NDM-1 pocket contains 8,724 atoms, confirming gross contamination rather than a compact active-site pocket.

This problem affects residue-level graph construction as well as explicit coordination features wherever residue IDs collide.

### Metal confidence was not real confidence

The current wrapper assigns a constant `metal_confidence = 0.7` to Metal3D-derived results instead of preserving the predicted probe probability. Confidence-based filtering or calibration built on this value is therefore meaningless.

### Scientific consequence

The current structural features, structure-only results, and fused production results were computed from pockets that can be centered on phantom coordinates and contaminated across chains. This does not necessarily make every one of the 1,051 Metal3D-centered examples wrong, but it affects the whole preprocessing path and prevents a trustworthy claim about structural learning.

The ESM-2 baseline and sequence-component audits remain informative because they do not depend on this metal coordinate. The production V1 model remains operational as a sequence-dominant ranking system, but its apparent structural component must be treated as unvalidated.

## 10. Required bounded structural V2 repair

Do not overwrite the existing V1 pockets or production artifacts. Generate versioned structural V2 data so the frozen V1 remains reproducible.

The repair should:

1. Run Metal3D on the cleaned dominant-chain/full-chain structure, not the original multi-chain assembly used by the old pocket path.
2. Preserve residue identity as `(chain_id, residue_id, insertion_code)` throughout extraction, graph construction, donor finding, SASA lookup, and sequence adjacency.
3. Parse every Metal3D probe as a discrete candidate with its probability/occupancy.
4. Never average spatially separate candidate sites.
5. Select the highest-probability plausible primary site as the pocket center.
6. Retain nearby high-probability probes (for example, within roughly 5 Å) as members of the same metal complex so B1/B3 dinuclear geometry is not collapsed.
7. Store `metal_coords` and probabilities, with a backward-compatible primary `metal_coord` only if needed by old code.
8. Build the pocket around the union of residues near the retained one- or two-site complex.
9. Replace the constant 0.7 confidence with actual Metal3D probability-derived metadata.
10. Record site-selection provenance and failure modes in every pocket archive.

Site selection may use label-independent physical plausibility, such as nearby canonical donors, but it must not use the positive/negative label or subclass.

### Validation before corpus regeneration

First validate the repaired parser/extractor on the seven named experimental references:

- nearest predicted-to-observed metal distance
- correct mono-/dinuclear site count where known
- donor coverage at 2.8, 3.2, and 3.5 Å
- chain purity
- plausible pocket atom/residue counts
- preservation of Metal3D probabilities

Suggested go/no-go targets, to be treated as engineering gates rather than universal biological laws:

- median reference-site localization error around or below 1 Å
- no cross-chain residue contamination
- approximately 90% donor coverage among known positives by 3.5 Å after site selection
- sensible mono-/dinuclear recovery on canonical references

Then audit a label-balanced sample of predicted positives and negatives visually and quantitatively. Experimental reference structures validate coordinate localization; predicted structures test whether side-chain and pocket quality remain sufficient in the actual deployment domain.

Only after these checks should the full V2 pocket corpus be regenerated.

## 11. Structural V2 evaluation plan

The immediate objective is not to maximize a generic accuracy number. It is to test whether corrected structure supplies orthogonal discovery value.

### Stage A: cheap diagnostics

On corrected pockets, rerun:

- explicit coordination fingerprint with balanced GBM
- positive/negative structural prototypes
- enhanced graph structure-only model
- simple hard-negative discrimination, especially MBL versus lactonase

Use the existing frozen component challenge splits without changing membership.

### Stage B: matched comparisons

Report:

- B1/B2, B3, and remote panel sensitivity at matched specificity
- recovery of ESM-2 misses at ≥99% specificity
- overlap and disagreement sets between ESM-2 and structure
- per-seed uncertainty and bootstrap intervals clustered by sequence component, not naïve per-example bootstrap
- LONO specificity, with special emphasis on lactonase
- performance stratified by metal localization confidence and predicted-structure quality

Avoid claiming independent evidence merely because two scores come from different network branches if they share the same upstream sequence-derived structure prediction.

### Stage C: decision gate for an equivariant encoder

Proceed to GVP/GearNet/e3nn/EZSpecificity-style modeling only if corrected interpretable geometry shows either:

- meaningful unique positive recovery at matched specificity, or
- clear hard-negative suppression without losing remote positives.

If corrected geometry remains weak, the honest conclusion may be that predicted apo structures and this dataset do not contain enough metal-site information for a structure-led classifier. In that case, a larger encoder should not be built merely to avoid a negative result.

### Stage D: inference-rule comparison

Because production k-NN can itself limit novelty, compare on frozen panels:

- seed-specific k-NN
- direct classification head trained on structural embedding
- class/prototype distance or energy score
- calibrated positive-versus-hard-negative likelihood
- OOD/abstention score

Select a rule before inspecting Atlas outcomes. Retain nearest-neighbor provenance even if a direct classifier is used, because it is valuable for interpretation.

## 12. Atlas deployment strategy

The Atlas is too large to use as an unconstrained training/debugging loop. Model and thresholds must be frozen before large-scale candidate inspection, and candidate outputs must preserve enough metadata to avoid rerunning tens of terabytes.

For V1 screening, retain at minimum:

- GNN vote count and per-seed neighbor fractions
- raw mean-ESM-2 k-NN score
- identities and labels/families of nearest neighbors
- distance to nearest positive and nearest hard negative
- MMseqs/HMM evidence and alignment coverage
- predicted subclass/family context, if assigned
- Metal3D/site confidence and structure confidence
- lactonase and other metallo-hydrolase counter-screen annotations
- OOD/distance-from-bank indicator

Candidate prioritization should not simply require both ESM-2 and GNN positivity. That would suppress genuinely complementary discoveries. A better tiering scheme is:

- agreement high: strong retrieval candidates, likely closer to known families
- ESM-2 high / structure low: sequence-supported candidates; inspect structure quality and pocket extraction
- ESM-2 low / corrected structure high: highest-interest structural novelty candidates, but demand strict hard-negative and geometry review
- both low: deprioritize unless supported by independent catalytic motifs or contextual evidence

Because the current V1 structure input is compromised, V1 disagreement cannot yet be interpreted this strongly. Until V2 is validated, the ESM-2 score is the more trustworthy signal and V1 GNN should be treated as an auxiliary prioritizer.

Prospective biochemical validation is the only convincing final test of cryptic activity. High-priority candidates should be assayed for β-lactam hydrolysis, metal dependence, substrate profile, and expression/solubility. Structural novelty without hydrolysis is not an ARG MBL discovery.

## 13. Approaches rejected or explicitly limited

### Rejected as invalid

- **Random/stratified folds with homologous overlap:** measure family recognition, not remote generalization.
- **Pocket-fragment RMSD OR sequence edges with transitive closure:** creates giant components through short-fragment alignment artifacts.
- **Calling the no-80%-coverage bin “zero alignable identity”:** contradicted by substantial local MMseqs hits.
- **Averaging latent coordinates across seeds:** latent axes are not aligned.
- **Averaging all Metal3D probes into one point:** creates physically meaningless coordinates.
- **Using tier as an independent confidence feature:** tier is class/source-redundant.
- **Treating the seven canonical references as external validation:** they sit within the major positive training components.

### Not supported by this dataset

- **Novel-fold/structure-remote cross-validation:** nearly all positives share one structural component because they share the MBL fold.
- **Reliable B2-specific conclusions:** only three B2 positives exist.
- **Population claims from remote outliers:** n=4.
- **A single certified accuracy for the final production checkpoints:** they were trained on the full labeled pool and have no held-out test set.

### Tried but not selected for V1

- **Branched fusion:** modest B1/B2 improvement but remote regression and no consistent overall advantage.
- **Structure-only current GNN:** reproducible signal on B3, weak and non-complementary on B1/B2.
- **Current coordination fingerprint classifiers:** weak, but the experiment is inconclusive because the input coordinate was corrupted.
- **Many-model ensemble:** not justified without proven orthogonal value and calibration.

### Deferred

- **Equivariant/pretrained structural encoder:** wait for validated V2 inputs and cheap structural evidence.
- **Retrospective Atlas-driven retraining:** avoid until a prospectively defined discovery round is complete.

## 14. Current production status

The frozen V1 is `new_graph_flat_production`:

- eight seeds: 0–7
- 60 epochs each
- trained on all 1,077 labeled structures
- 1,320-dimensional flat node input
- seed-specific five-neighbor retrieval and majority vote
- separate mean-ESM-2 five-neighbor score

Best available proxy results for this architecture are:

| Panel | GNN sens/spec/BA | Mean-ESM-2 sens/spec/BA |
|---|---|---|
| B1/B2 transfer | .276 / 1.000 / .638 | .897 / .998 / .947 |
| B3 transfer | .731 / 1.000 / .865 | .692 / .990 / .841 |
| Remote outliers | .500 / .983 / .741 | .750 / .961 / .856 |

These are panel-trained proxies, not direct accuracy measurements of the full-data production checkpoints.

The V1 artifact should be described as moderate-quality and sequence-dominant. It adds evidence on B3-like transfer but is weaker than raw ESM-2 on the hardest B1/B2 transfer. Its structure-aware claim is further weakened by the newly identified pocket preprocessing defects.

Do not silently replace V1. Preserve its manifest, hashes, checkpoints, and existing results for reproducibility. A repaired model should be named/versioned as structural V2 and compared prospectively against V1.

## 15. Unresolved scientific and design questions

1. After corrected site extraction, does explicit coordination geometry recover B1/B2 positives missed by ESM-2 at matched high specificity?
2. Can structure reduce the severe lactonase false-positive problem without sacrificing remote-MBL sensitivity?
3. How accurately does Metal3D localize one or two sites on predicted, apo-like structures across the whole corpus after correct parsing?
4. Is one primary site plus nearby probes the right complex-selection rule for all B1/B2/B3 cases, or is a label-independent clustering/refinement method needed?
5. How much of the B1/B3 performance asymmetry is biological subclass difference versus positive-component size and diversity?
6. Does the metric-learning objective over-collapse distinct MBL subclasses into one region?
7. Does production k-NN reject truly novel structural solutions simply because they lie far from the labeled bank?
8. Would a direct structural classifier or energy/OOD model generalize better than reference-bank retrieval?
9. Can provenance confounding be reduced without acquiring new labels—for example, by source-balanced negatives, reweighting, or adversarial source invariance?
10. Are the negative labels experimentally secure, especially for metallo-hydrolase-family proteins that may have unannotated promiscuous β-lactamase activity?
11. What candidate-review budget and minimum specificity are operationally acceptable at Atlas scale?
12. What prospective wet-lab validation set will provide the first unbiased estimate of discovery precision?

The lack of additional known positive instances is a hard constraint, not a prompt for endless repartitioning. Progress must come from correcting invalid inputs, using component-aware uncertainty, exploiting hard negatives, and designing a prospective Atlas/wet-lab evaluation—not repeatedly tuning on the same six positive components.

## 16. Recommended immediate continuation in a fresh thread

1. Inspect `pocket_extraction.py`, `utils.py`, `graph_construction.py`, and the bundled Metal3D output format.
2. Implement a versioned, chain-aware multi-site pocket schema without changing V1 data.
3. Validate the extraction on the seven canonical experimental references, beginning with the already demonstrated NDM-1 case.
4. Audit a balanced predicted-structure subset.
5. Regenerate V2 pockets only after passing explicit geometry/chain-purity gates.
6. Repeat the coordination fingerprint and structure-only diagnostics on unchanged challenge splits.
7. Decide whether an equivariant encoder is justified using matched-specificity incremental value.
8. Freeze either V1-only or V1+validated-V2 Atlas ranking rules before inspecting large-scale candidates.

The guiding principle is not perfectionism. It is to fix the single known upstream defect that makes the structural experiment uninterpretable, run a bounded decision experiment, and then either proceed with a justified structural V2 or accept an honest sequence-led V1.

## 17. Superseding update: corrected V2 and candidate-local Structural V3

The recommendations above were subsequently executed. Corrected multi-site,
single-chain V2 pockets established that canonical DCH cysteine coordination
is a strong B1/B2-specific structural signal, while first-shell chemistry does
not discriminate B3 from related metallohydrolases. Mean-pooled ESM-IF1 outer-
pocket classification was also tested and rejected because it recovered none
of the B3 positives missed by ESM2 and worsened several negative-family false-
positive rates. The current production regime remains the separately versioned
V2 asymmetric scorer; legacy V1 is retained only for reproducibility.

The concern that positive/reference-panel comparison itself might prevent a
generic structural solution was then tested directly in Structural V3. V3
uses no sequence, ESM embeddings, labels, learned weights, positive-protein
centroids, or nearest neighbours. It transfers experimental beta-lactam
reaction states from 4EYL (B1), 1X8I (B2), 2AIO (B3), and 6U0Z (B3) into each
candidate's corrected metal/donor coordinate frame, then measures local
pharmacophore RMSD, severe steric clashes, and pocket contact.

The acceptance criteria were frozen before scoring the 1,077-structure corpus.
V3 recovered 8/20 positives missed by mean-ESM2 and donor-direction scrambling
eliminated all 115 native positive calls, confirming genuine three-dimensional
dependence. Nevertheless, V3 failed decisively: specificity among evaluable
negatives was 0.338, only 50.9% of structures were evaluable, and false-positive rates reached
0.608 for glyoxalase-II, 0.451 for RNase Z, and 0.260 for phosphodiesterases.

This result changes the causal interpretation. Reference-panel comparison was
not the main bottleneck. The central difficulty is functional degeneracy:
related metallohydrolases genuinely share the Zn/donor scaffold and can accept
a rigidly transferred hydrolyzed-product pose. Distinguishing beta-lactam
turnover requires evidence not present in this representation, such as
substrate-conditioned flexibility, water/proton networks, transition-state
energetics, or experimental activity labels.

Structural V3 is therefore a documented NO-GO and is not integrated into
production. Do not tune its thresholds on the same panels and do not begin an
Atlas-scale screen with it. Full methods, hashes, metrics, and controls are in
`reports/catalytic_template_audit.json`,
`reports/catalytic_feasibility_evaluation.json`, and
`reports/catalytic_feasibility_no_go.md`. No Atlas access or processing was
performed during this experiment.

## 18. Superseding update: canonical-B1 six-donor pharmacophore

The broad Structural V3 failure did not imply that every subclass-specific
structural question was dead. B3 first-shell geometry is genuinely degenerate
with related metallohydrolases, but canonical B1 contains a more distinctive
two-site donor arrangement. Work therefore narrowed to B1 rather than asking
one structural model to detect every MBL subclass.

The first B1 model was anchored to corrected Metal3D sites and already showed
real angular dependence, but its apparent specificity included 241/931
negative structures that were unevaluable because no acceptable dinuclear
metal prediction existed. This was fixed by removing predicted metal
coordinates from the detector entirely.

`scripts/metal_independent_b1.py` now searches a complete single-chain
structure for two donor triads: three distinct His N atoms and an Asp O, Cys S,
His N set. It fits all six donor coordinates to the 4EYL hydrolyzed-meropenem
reaction-state template. It reads no sequence order, motif, ESM embedding,
HMM score, label, reference protein, class centroid, or predicted metal site.
The primary output is the six-donor pharmacophore; transferred product-pose
clash/contact is retained as secondary evidence because that gate is more
NDM-pocket-specific.

Frozen internal results on 110 B1 positives and 410 negatives from the audited
B1/B2 transfer panel are:

- donor inventory only: sensitivity 1.000, specificity 0.059;
- within-site geometry: sensitivity 0.991, specificity 0.990;
- six-donor pharmacophore: sensitivity 0.991, specificity 1.000;
- pharmacophore plus transferred product pose: sensitivity 0.982,
  specificity 1.000;
- mean-ESM2 5-NN comparator: sensitivity 0.909, specificity 0.998;
- fARGene B1-specific HMM: sensitivity 1.000, specificity 1.000.

The six-donor model called 0/931 labeled negatives and recovered all ten B1
panel positives missed by mean-ESM2. It detected 104/105 B1 positives with no
MMseqs hit at the audited 80%-coverage criterion. Donor-coordinate permutation
reduced detection to 2/110 and generated 15/410 false positives, while donor
inventory alone called 386/410 negatives. The useful signal is therefore the
role-specific three-dimensional architecture, not cysteine presence or donor
composition alone.

The method also recovered 15 known B1 examples missed by the Metal3D-anchored
version, including VIM-2 and IMP-1. Every full-chain input is now evaluable;
missing predicted sites are no longer converted into operational negative
calls.

An external PDB panel was declared from MBL structure reviews before scoring:
15 canonical B1 enzymes, SPS-1 as a noncanonical boundary case, two B2
controls, and eight B3 controls. At the frozen 1.50 A donor-pair enumeration
tolerance, the six-donor architecture detected 14/15 canonical B1 structures
and rejected 10/10 B2/B3 controls. SPM-1 and FIM-1 contain oxidized modified
cysteines in the crystallographic files; the parser now preserves those SG
coordinates but reports modified-ligand architecture separately from native
thiolate support. SPS-1 was correctly rejected under the declared canonical-B1
scope. The sole architecture miss was a distorted NDM-1 3S0Z conformation.
This is a portability panel rather than an independent novel-family set:
several canonical families overlap the project's biological reference space,
and the physical template is itself NDM-derived.

A 1.00-2.00 A enumeration-tolerance sweep left the internal result exactly
109/110 and 0/410, with 0/931 negative calls, at every setting. The external
panel progressed from 13/15 at 1.00-1.25 A to 14/15 at the frozen 1.50 A and
15/15 at 1.75-2.00 A, always with 0/10 B2/B3 calls. Because the external miss
motivated inspection of the margin, the more permissive value is sensitivity
analysis, not a post-hoc replacement of the primary threshold.

The correct current claim is:

> A reference-independent, sequence-blind six-donor pharmacophore recognizes
> canonical B1 catalytic architecture and outperforms mean-ESM2 nearest-neighbor
> retrieval on the audited B1 panel. It is not merely a cysteine rule and does
> not depend on Metal3D coordinates. It has not yet demonstrated prospective
> discovery of a biochemically verified fARGene-negative B1 enzyme.

This is a useful structural tool even though it does not solve B3. It provides
an orthogonal way to confirm or prioritize canonical B1 architecture and has a
mechanistic route to detecting HMM-negative sequences. The remaining evidence
gap cannot be closed by repartitioning the same known positives: it requires a
real structure that falls below the frozen fARGene threshold and ultimately a
functional assay. No Atlas search was run in this branch.

Authoritative artifacts for this update are:

- `scripts/metal_independent_b1.py`;
- `scripts/evaluate_metal_independent_b1.py`;
- `scripts/evaluate_external_mbl_panel.py`;
- `scripts/evaluate_b1_threshold_sensitivity.py`;
- `configs/external_experimental_mbl_panel.json`;
- `reports/metal_independent_b1_evaluation.json`;
- `reports/external_experimental_mbl_panel.json`;
- `reports/metal_independent_b1_threshold_sensitivity.json`;
- `docs/B1_STRUCTURAL_DETECTOR.md`.
