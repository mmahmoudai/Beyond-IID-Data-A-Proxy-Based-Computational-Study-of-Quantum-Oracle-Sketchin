# Revision Audit — MDPI Sensors variant (branch `mdpi-revision-r1`)

Date: 2026-08-11. Scope: `paper/mdpi_sensors/` only; the other four venue variants are untouched on `main`.

**Provenance note (updated 2026-08-11).** The verbatim reviewer reports were received after the first revision pass (R1: 15 recommendations, review dated 07 Aug 2026, "must be improved" on Methods; R2: long-form "extensive revision"). `response_to_reviewers.md` is now the final point-by-point letter with verbatim quotes. The reports confirmed the brief's digest almost exactly; four points required *new* text beyond the first pass and were added in the third commit: R1.7 (baseline justification, §5.2), R1.9 (regime realism, §4.6), R1.11 (τ/r estimator cost and scalability, §5.1 — the one item entirely absent before), and R2.7 (full-series percentile threshold = look-ahead in the target definition — disclosed in §6.8 + Threats(v); placement is target-free so only baseline reference numbers are touched; calibration-window threshold adopted as future work).

---

## Phase 1 — Review-audit table

| # | Reviewer concern (theme) | Severity | Affected sections | Required action | No new experiments? | Applied revision |
|---|---|---|---|---|---|---|
| T1 | Manuscript overstates quantum implications of a heuristic proxy framework | **MAJOR** | Title, abstract, §1, §3, captions, §Conclusions | Reposition as diagnostic + hypothesis-generation framework | Yes | Title changed; abstract rewritten (293→~220 words); positioning sentence in §1; "hypothesized/proxy-favourable" throughout; conclusion rewritten |
| T2 | Wilcoxon pairs stochastic baselines with deterministic proxy values; p-value-led interpretation unjustified | **MAJOR** | §Exp 5 (Table 5), abstract, §Discussion | Reinterpret via effect sizes, differences, CIs; state tests are not algorithm-vs-algorithm | Yes (re-analysis of existing seeds) | HL estimates + exact signed-rank CIs + rank-biserial r added; Holm non-survival disclosed; "What these tests do and do not establish" paragraph |
| T3 | Analytic vs empirical τ/r divergence (esp. long-range dependence) unexplained | MODERATE | §Exp 4, §Discussion | Dedicated discussion: explanation, interpretation, implications | Yes | New §"Why Analytic and Empirical Correlation Estimates Diverge" (3 labelled paragraphs) |
| T4 | Reader could infer experimentally demonstrated exponential memory advantage | **MAJOR** | §3, §Exp 2, §Exp 6, Fig 6, §Discussion | State all memory bounds are theoretical; no empirical floor for SGD/avg-SGD/Count-Min | Yes | §3.3 "Memory references"; Exp 6 opening disclaimer; Fig 6 caption; provenance table row |
| T5 | Only two real datasets; generalisation unverified | MODERATE | §Exp 8, §Limitations | Mark illustrative; add domains as future work | Yes | "Illustrative rather than statistically representative" sentence; Threats (External); Future Work names 4 domains |
| T6 | Proxy cannot predict downstream classifier performance (target dependence) | MODERATE | §Exp 9, §Discussion | State explicitly; characterises correlation structure, not task learning | Yes | Scope-boundary text in Exp 9; new §"Scope of the Diagnostic"; Fig 9 caption |
| T7 | Sensitivity to constants C and δ requested | MODERATE | §3.2, §6.3, §Exp 9 | Expand analysis if existing data permit | Yes (deterministic re-evaluation) | New C–δ paragraph + Table 8 (ρ* across 4×3 grid); §3.2/§6.3 updated |
| T8 | Figure crowding / labelling / framing | MINOR | Figs 1–9 | Figure-specific fixes; captions matched to framing | Yes | fig5 legend "advantage"→"Proxy-estimated separation"; 5 titles "vs.\\"→"vs."; figs 4/5/6 regenerated; captions 3/5/6/8/9 rewritten; 4 overfull table boxes fixed |
| T9 | No systematic validity treatment | MODERATE | §Limitations | Dedicated Threats to Validity section | Yes | New §Threats to Validity (construct/internal/external/statistical-conclusion), absorbing the former 6-item list |
| T10 | Conclusion frames results as quantum advantage evidence | **MAJOR** | §Conclusions | Rewrite: primary = landscape; secondary = hypothesis generation | Yes | Full rewrite incl. practical contribution, limitations pointer, required validation steps |

**Every required action was achievable without new stochastic experiments.** T2 and T7 used deterministic re-analysis of existing per-seed data (reproduced bit-exactly; see below).

## Reviewer consensus matrix

| Theme | Reviewer #1 | Reviewer #2 | Consensus |
|---|---|---|---|
| T1 overstatement / reposition as diagnostic | ✅ R1.1 | ✅ R2.2, R2.10 | **BOTH — central concern** |
| T2 statistics (deterministic-vs-stochastic; magnitude over p) | ✅ R1.6 | ✅ R2.3 | **BOTH** |
| T3 τ/r divergence | ✅ R1.3 | ✅ R2.4 ("one of the most important findings") | **BOTH** |
| T4 memory scaling not empirical | ✅ R1.13 | ✅ R2.5 | **BOTH** |
| T5 real data limited / conservative | ✅ R1.4 | ✅ R2.6 | **BOTH** |
| T6 target function / proxy target-independent | — | ✅ R2.9 (+ Count-Min R2.8) | R2 |
| T7 C/δ sensitivity | ✅ R1.5 | — | R1 |
| T8 figures | ✅ R1.12 | — (R2 rated figures "Yes") | R1 |
| T9 threats to validity | ✅ R1.10 | — | R1 |
| T10 conclusion | ✅ R1.15 | ✅ R2.10 | **BOTH** |
| NEW: proxy-vs-framework relationship | ✅ R1.2 | ✅ R2.1 (key issue) | **BOTH** |
| NEW: baseline justification | ✅ R1.7 | — | R1 |
| NEW: τ/r sufficiency limits | ✅ R1.8 | — | R1 |
| NEW: regime realism | ✅ R1.9 | — | R1 |
| NEW: estimator complexity/scalability | ✅ R1.11 | — | R1 |
| NEW: other domains | ✅ R1.14 | — | R1 |
| NEW: percentile-target look-ahead | — | ✅ R2.7 | R2 |

## Phase 2 — Claim-language audit (all applied)

Bibliography titles (lines ~799/805/931, e.g. "Exponential quantum advantage…") are citations of other papers' titles and are untouched. Theory attributions with \citep anchors are retained but marked as theory. Live-risk items and their dispositions:

| Original statement | Location | Risk | Replacement (applied) |
|---|---|---|---|
| "Quantum oracle sketching achieves exponential memory advantages…" (abstract opener, unqualified) | Abstract | **HIGH** | "In theory, quantum oracle sketching achieves…" + "no quantum algorithm is implemented or simulated" |
| "…shows that quantum oracle sketching achieves exponential quantum advantages…" | §1 ¶2 | HIGH | "establishes, at the level of theory, exponential quantum advantages…"; "task-specific proofs, not empirical measurements" |
| Title: "Quantum Oracle Sketching for Non-IID … Streams" | Title | MED | "Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for …" |
| "the proxy model suggests a favourable region…" (Contribution 4) | §1 | MED | "predicts a hypothesized quantum-favourable region… proxy-level hypotheses …, not demonstrated quantum performance" |
| "classical significantly exceeds the proxy (p = 0.037)" | Abstract, Exp 5, Disc. | MED | effect sizes + CIs; p descriptive; Holm disclosure; "closes … ρ* ≈ 0.82; by ρ = 0.88 … exceeds … by 0.040 on average" |
| "exponential scaling of the quantum-vs-classical memory gap" | Exp 6 | MED | "…of the *theoretical* quantum-vs-classical memory references"; "no memory lower bound is measured empirically" |
| "an exponential gap that widens with n" | Exp 6 | MED | "a separation that is exponential in n at the level of the cited bounds" |
| "favourable region" (bare, multiple) | Figs 3/8 captions, §Disc, Table 6 | MED | "proxy-favourable" / "hypothesized favourable"; Table 6 column "Proxy prediction"→"Proxy hypothesis" + caption note |
| Fig 5 legend: "Proxy-model advantage (empirical τ)" | figure-internal text | **HIGH** (missed by tex grep; caught in visual audit) | regenerated: "Proxy-estimated separation (empirical τ)" |
| "Three structural factors are consistent with a favourable proxy-estimated regime" | §Disc 8.1 | LOW | "…with a hypothesized proxy-favourable regime" + "none is a measured quantum result" |
| "Stability of the proxy-baseline crossover" (para title implying confirmatory tests) | Exp 5 | LOW | "What these tests do and do not establish" |
| "the proxy is absent in the long-memory regime" (ambiguous) | Conclusion RQ3 | LOW | "the proxy-predicted separation vanishes entirely in the long-memory regime" |

Guarantees now held everywhere: the manuscript never implies a quantum algorithm was implemented, a circuit was simulated, or an advantage was experimentally demonstrated. Explicit denials appear in the abstract, §1, §3.3, Exp 5, Exp 6, Exp 8, §Scope-of-Diagnostic, Threats, and Conclusions.

## Phases 4 & 10 — Statistical re-analysis (deterministic, no new experiments)

Reproduction: Experiment 5's 10×15 per-seed matrices were regenerated from the pinned seeds (42–51); **all five published Wilcoxon p-values match `figures/exp5_wilcoxon.txt` to six significant digits** (the one 4th-decimal mean discrepancy, 0.9276 vs 0.9277, is print-rounding of the same value — exact p-match rules out any data difference).

New quantities (paired proxy − classical):

| ρ | HL Δ̂ | exact 95% CI | r_rb | p (unadj.) | p (Holm) |
|---|---|---|---|---|---|
| 0.00 | +0.042 | [+0.011, +0.087] | +0.82 | 0.020 | 0.098 |
| 0.27 | +0.042 | [+0.012, +0.090] | +0.78 | 0.027 | 0.109 |
| 0.47 | +0.031 | [+0.002, +0.080] | +0.71 | 0.049 | 0.111 |
| 0.68 | +0.019 | [−0.011, +0.069] | +0.35 | 0.375 | 0.375 |
| 0.88 | −0.046 | [−0.079, −0.003] | −0.75 | 0.037 | 0.111 |

C–δ crossover grid (ρ*, mean curves, empirical τ): rows C = 1.0/1.5/2.0/2.5 × cols δ = 0.01/0.05/0.10 → (0.93, >0.95, >0.95), (0.84, 0.89, 0.92), (0.61, **0.82**, 0.86), (0.38, 0.61, 0.76). Regime ordering invariant (κ = C²log(1/δ) is a monotone rescaling); absolute boundary is not.

Artifacts: `revision/exp5_reanalysis.py` + `revision/exp5_reanalysis.json` (copies of the executed script and output).

## Phase 11 — Figure audit outcome

| Figure | Verdict | Action |
|---|---|---|
| fig1 accuracy vs samples | OK | none |
| fig2 accuracy vs memory | OK | caption theory-reference wording (tex) |
| fig3 (τ,r) landscape | OK (CVD-safe, labelled contours since v13) | caption: hypothesis-map sentence |
| fig4 empirical vs analytic | fix | titles "vs.\\"→"vs." (regenerated) |
| fig5 Markov sweep | **fix** | legend "advantage"→"Proxy-estimated separation"; titles fixed (regenerated); caption ρ*≈0.82 wording |
| fig6 dimension scaling | fix | title fixed (regenerated); caption "theoretical … not a witnessed empirical floor" |
| fig7 rolling accuracy | OK | none |
| fig8 real streams | OK (v13 layout fixes hold) | caption "proxy-favourable/-unfavourable" prefixes |
| fig9 ablations | OK | caption target-independence sentence |

Numerical outputs: unchanged (rendering-only edits; seeded pipeline; `results/*.json` untouched by regeneration of figs 4/5/6 — verified via git status).

## Build state

`pdflatex` ×2: **0 errors, 0 undefined references, 0 overfull boxes, 25 pages** (was 21 — growth = new §3.3 + provenance table + two Discussion subsections + Threats + Future Work + C–δ table).

---

## Phase 13 addendum — internal hostile re-review panel (2026-08-11, post-revision)

Three independent reviewer agents re-attacked the revised manuscript (adversarial re-review; claim→evidence traceability over 59 promises; full statistics audit). Their confirmations and every accepted finding, with dispositions:

**Independently confirmed correct:** all five exact signed-rank p-values (re-derived from the DP null to 7 digits, proving no ties/zeros); Walsh-average exact CI construction (k=8, achieved coverage 95.1%); Holm arithmetic (min adjusted p = 0.098); all sign flips between the classical−proxy source and proxy−classical presentation; Table 2 and C–δ grid arithmetic; the five test points pre-specified in code (retires forking-paths).

**Fixed in this pass (second commit on the branch):**
1. RQ3/contribution/figure captions bound each regime verdict to its estimator convention — under empirical τ, burst and long-memory *swap places* vs the analytic landscape (burst → saturated worst via measured τ·r; long-memory → best tier via the short-lag artifact). Now stated in Exp 9, §Divergence, RQ2/RQ3, Fig 3/7 captions, Table 8 caption.
2. Exp 5 factual error: the analytic-τ proxy does **not** "cross below classical at ρ≈0.3" — it starts below (0.88 vs 0.928 at ρ=0). Text + Fig 5 caption corrected.
3. τ-lattice fragility disclosed: one integer step of τ̂ = 0.012 accuracy; the ρ=0.48 interval moves to [−0.010,+0.068] under τ̂+1 (not robust); ±0.000 explained as lattice quantisation; ρ* reported as a 0.81–0.88 bracket; r_rb reframed as sign-concordance (scale-free vs a near-constant arm); Holm non-survival framed as outcome (min attainable adjusted p at n=10 is 0.0098), not design ceiling.
4. CI mislabel: all marginal bands relabelled "±1.96·SE (normal approximation)" in captions/tables/Threats (true 95% t-intervals would be ~15% wider at n=10 — regeneration with t-quantiles listed as follow-up); paired inference unaffected (exact).
5. Balanced-accuracy/F1 intervals restored to Table 6 from results/real_stream_summary.json (Count-Min NYC F1 = 0.644±0.062, per-ordering range 0.43–0.72 now visible); sub-majority SGD accuracy and Averaged-SGD balanced accuracy straddling chance stated plainly; majority rows marked as by-construction.
6. Real-stream honesty: permuted-ordering CIs = ordering variability of one fixed series (and permutation removes the dependence (τ,r) measure → classical arm upper-optimistic); metric-switch concession (as an accuracy projection the proxy is falsified on Machine Temperature, 0.500 vs 0.72–0.80; the rare-class reading is reinterpretation); Count-Min-only support for the favourable-side separation stated; "pre-committed" softened to a disclosure.
7. **New Table (encoder/resolution sensitivity)** — the adversarial reviewer's #1 objection (τ/r as sampling-rate/encoder-field artifact) answered with a 10-configuration deterministic re-analysis: NYC > MT proxy ordering holds in all 10 configurations (matched 30-min: 0.884 vs 0.576; matched 60-min: 0.908 vs 0.719); MT leaves saturation only when downsampled 6–12×. Placements = encoder-conditional, ordering-stable.
8. Fig 8(d) synthetic crosses disclosed as *schematic/indicative* placements (they were hard-coded to neither analytic nor empirical coordinates); "near the synthetic Markov point" replaced by numeric coordinates; n=6-stars-on-n=10-contours caveat added.
9. In-image label leaks removed by regenerating figs 1/2/6 ("n=10 qubits", "Quantum proxy (IID, 10 qubits)", "Quantum O(n)" → proxy/cited-theory labels).
10. Exp 9 "at most 0.04" caption corrected (Burst moves 0.155 from saturation under AR(1)); "confirming W=20 representative" → ordering-insensitivity; target-ablation protocol difference disclosed (prequential lr=0.01 T=4096 vs held-out lr=0.05 T=10⁴ — resolves the apparent 0.5-vs-0.928 contradiction).
11. Falsifiable hypotheses H1–H3 with explicit refutation criteria and estimator binding added to §Scope of the Diagnostic.
12. r-only (theory-faithful) accounting stated in §3.3: Markov/seasonal/long-memory collapse to the IID point and no crossover exists under it; the τ·r landscape is a classical effective-sample stress test, strictly more pessimistic than the framework's accounting.
13. C=2 justification honesty ("no specific constant in the cited theory fixes it"); §4.3(iii) false arcsine-correction claim deleted; arcsine law re-attributed to Van Vleck & Middleton (1966, DOI 10.1109/PROC.1966.4567, verified); effective-sample-size lineage cited to Geyer (1992, DOI 10.1214/ss/1177011137, verified); §3.3(iii) hardness attribution corrected to the framework itself.
14. Seeds bug fixed everywhere (Exps 1–7 use 42–51, not 0–9): Table 3, Data Availability, REPRODUCE.md (which also gained the missing Machine-Temperature download step, the effect-size script, and the revised title).
15. Table 9 rows recomputed from the paper's own formula (financial-returns verdict now "Favourable (H≲0.6) → borderline"; contour labels corrected); §6.3 "0.7–0.8 contour" → correct 0.6-region values; abstract now carries the C–δ conditionality, the HL value (0.046), "diverge" instead of "overestimate", and "proxy-favourable" (the last two "quantum-favourable" instances removed); §Results retitled from "Results and Discussion".

**Explicitly deferred (honest follow-ups, not blockers):** regenerate all bands with t(9)=2.262 multipliers (relabelling makes current bands honest; conclusions unaffected); seed-bootstrap CI on ρ*; full 15-point Δ̂ supplement; ordered-stream (non-permuted) baseline reference row; more real streams (≥6) for a placement-vs-outcome rank correlation; Mohri–Rostamizadeh/Yu mixing-bounds related-work paragraph; renaming `compute_advantage_ratio`/`theoretical_advantage_landscape` in the code package at next Zenodo re-archive.

**Panel verdicts on the pre-fix revision:** adversarial reviewer — Reject as sole reviewer, 40% acceptance estimate at Sensors; claim-evidence — 26/59 evidenced, 3 critical + 6 major gaps; statistics — "defensible conditional on a focused round of corrections," borderline-pass scorecard. All three named the same root causes (estimator binding, uncertainty on headline numbers, encoder artifact), all of which the fix pass above addresses; the encoder objection is answered with new deterministic data rather than wording.
