# Revision Audit — MDPI Sensors variant (branch `mdpi-revision-r1`)

Date: 2026-08-11. Scope: `paper/mdpi_sensors/` only; the other four venue variants are untouched on `main`.

**Provenance note.** No verbatim reviewer reports are present in the repository; the audit below is reconstructed from the revision brief's 13-phase digest of Reviewer #1 and Reviewer #2. Theme 1 is explicitly attributed to both reviewers by the brief ("the reviewers' central concern"); per-reviewer attribution of the remaining themes should be confirmed against the verbatim reports before the response letter is sent (quotation slots are provided in `response_to_reviewers.md`).

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
| T1 overstatement | ✅ (per brief) | ✅ (per brief) | **BOTH — central concern** |
| T2 statistics | ● | ● | attribute on receipt of verbatim reports |
| T3 τ/r divergence | ● | ● | " |
| T4 memory scaling | ● | ● | " |
| T5 real data | ● | ● | " |
| T6 target function | ● | ● | " |
| T7 C/δ | ● | ● | " |
| T8 figures | ● | ● | " |
| T9 validity | ● | ● | " |
| T10 conclusion | ● | ● | " |

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
