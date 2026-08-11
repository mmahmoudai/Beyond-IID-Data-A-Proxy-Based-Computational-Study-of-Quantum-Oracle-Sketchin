# Response to Reviewers

**Manuscript:** *Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for Non-IID Sensor and Telemetry Streams* (previously *Quantum Oracle Sketching for Non-IID Sensor and Telemetry Streams: A Proxy-Based Computational Study*)
**Manuscript ID:** [to be inserted]
**Journal:** Sensors (MDPI)

> **Note to authors before sending:** this letter is organised by the reviewers' concern themes. Paste each reviewer's verbatim comment into the marked quotation slots and, if a theme was raised by only one reviewer, adjust the attribution line. Every "Changes made" entry cites the revised manuscript by section/table/figure so the editor can verify quickly.

---

We thank both reviewers for their careful reading and for a critique that has materially improved the paper. We agree with the central point: the previous version, while internally hedged, still *presented itself* as evidence about quantum performance, when what it can honestly support is a **proxy-based diagnostic and hypothesis-generation framework for correlation-sensitive streaming analysis**. The revision repositions the entire manuscript accordingly — title, abstract, contributions, statistics, figures, and conclusion — without weakening the legitimate empirical content: no experiment was removed, no result was altered, and all numerical outputs are regenerated bit-identically from the same seeded pipeline.

**Summary of the major structural changes:**

1. **Retitled** to *Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for Non-IID Sensor and Telemetry Streams*; the abstract now opens with "In theory, …" and explicitly states that no quantum algorithm is implemented or simulated.
2. **New Section 3.3, "Relationship Between Operational Proxies and Quantum Oracle-Sketching Theory,"** separating established theory / assumptions / heuristic constructions / scope limitations, with a provenance table (Table 3) stating for every quantity what is derived from prior theory, what this paper introduces, and what is *not* theoretically guaranteed.
3. **Statistics rewritten around effect sizes** (Experiment 5): Hodges–Lehmann paired-difference estimates with exact signed-rank 95% CIs, matched-pairs rank-biserial correlations, p-values demoted to descriptive, and an explicit disclosure that no comparison survives Holm correction at n = 10 seeds. A new paragraph states plainly that these tests pair a stochastic learner with a model prediction, not two implemented algorithms.
4. **New C–δ sensitivity analysis** (Experiment 9, Table 8): the proxy–baseline crossover ρ* moves from 0.38 to beyond the sweep across a C × δ grid, so the landscape is now explicitly framed as ordinal.
5. **New Discussion subsections**: "Why Analytic and Empirical Correlation Estimates Diverge" (technical explanation, interpretation, implications) and "Scope of the Diagnostic: What the Landscape Can and Cannot Predict."
6. **New "Threats to Validity" section** (construct / internal / external / statistical-conclusion).
7. **Future Work promoted to its own section**, now naming network telemetry, cybersecurity event streams, financial series, and IoT fleets, and putting algorithm-level validation first.
8. **Conclusion rewritten**: primary contribution = the (τ, r) sensitivity landscape; secondary = proxy-based hypothesis generation; explicit statement that the paper demonstrates no quantum advantage.
9. **Figures**: the Experiment 5 legend entry "Proxy-model advantage" was renamed "Proxy-estimated separation"; captions now state that landscapes are hypothesis-generation maps; minor typographic fixes in five panel titles.

---

## Theme 1 — Overstated quantum implications (both reviewers; central concern)

> *[Insert Reviewer #1 verbatim comment]*
> *[Insert Reviewer #2 verbatim comment]*

**Response.** We agree, and this concern drove the revision. The paper is now positioned, from the title onward, as a diagnostic and hypothesis-generation framework. Every remaining occurrence of "favourable" is qualified as *proxy-* or *hypothesized* favourable; the Introduction states that the paper "neither demonstrates nor claims a realised quantum–classical separation" (§1, Scope paragraph); the contribution list labels the regime-dependent findings "proxy-level hypotheses intended for future algorithm-level validation, not demonstrated quantum performance" (§1, Contribution 4); and the Conclusion opens by naming the landscape as the primary contribution and states: "The paper demonstrates no quantum advantage and implements no quantum algorithm" (§Conclusions).

**Changes made.** Title; abstract; §1 scope paragraph and contributions; §3.3 (new) with provenance Table 3; captions of Figures 3, 5, 6, 8, 9; Table 6 column retitled "Proxy hypothesis"; §Conclusions rewritten.

## Theme 2 — Wilcoxon tests between stochastic baselines and deterministic proxy values

> *[Insert verbatim comment]*

**Response.** The reviewer is right that a paired signed-rank test in which one arm is (essentially) deterministic cannot be read as an algorithm-versus-algorithm comparison, and that p-value-led interpretation overreached. We have rebuilt the statistical reporting around estimation rather than testing. Table 5 now reports, per sweep point, the Hodges–Lehmann estimate of the paired (proxy − classical) difference with its **exact** signed-rank 95% CI (Walsh-average construction) and the matched-pairs rank-biserial correlation; the p-values remain only as descriptive companions. We additionally disclose that **none of the five comparisons survives Holm correction** at the family-wise 0.05 level at n = 10 seeds (smallest adjusted p = 0.098), and we state explicitly that the evidence for the crossover is the monotone sign reversal of the paired difference (+0.042 → +0.019 → −0.046) with CIs excluding zero at both ends — not statistical significance. A dedicated paragraph ("What these tests do and do not establish", §Experiment 5) states that the pairing is stochastic-learner-versus-model-prediction, that the proxy's seed variability is inherited entirely from the per-seed τ estimator, and that "no quantum algorithm is implemented anywhere in this comparison."

**Changes made.** Table 5 (rebuilt with Δ̂ [95% CI] and r_rb columns and a fully rewritten caption); §Experiment 5 body paragraph; new interpretation paragraph; abstract (p-value removed, replaced by the mean difference); §Discussion implications item (3); Threats to Validity, "Statistical-conclusion validity."

## Theme 3 — Divergence between analytic and empirical τ/r, especially under long-range dependence

> *[Insert verbatim comment]*

**Response.** We added a dedicated subsection (§Discussion, "Why Analytic and Empirical Correlation Estimates Diverge") giving (i) the technical explanation — the analytic proxies target model-level, worst-case dependence scales of the generator (joint-chain mixing time; tail-dominated integrated autocorrelation), while the empirical estimator measures the first 1/e crossing of the marginal, binarised autocorrelation, an instance-level short-lag quantity; the two are answers to different questions; (ii) the interpretation — the divergence is partly expected behaviour (for a non-mixing LRD process no finite scalar exists for the two estimators to agree on) and partly a proxy limitation (a single scalar τ is intrinsically lossy there), and is *not* a data-generation defect (the gap persists under all four alternative τ estimators of Experiment 9); and (iii) the implications — empirical τ is the operationally meaningful input, analytic τ is a conservative stress-test envelope, and the analytic-to-empirical *ratio* is itself a long-memory diagnostic.

**Changes made.** New §Discussion subsection with three labelled paragraphs; Experiment 4 closing sentence now forward-references it; Conclusion RQ1 references it.

## Theme 4 — Memory-scaling claims could be read as experimentally demonstrated

> *[Insert verbatim comment]*

**Response.** Agreed. The revision makes the theoretical status of every memory quantity explicit at each occurrence. Experiment 6 now opens: "No memory lower bound is measured empirically in this experiment," and states that no result constitutes an empirical memory floor for online SGD, averaged SGD, or Count-Min; §3.3 ("Memory references") makes the same statement structurally, and the provenance table lists the Ω(√N) curve as a "conservative theoretical reference" whose empirical-floor status is explicitly *not guaranteed*. Figure 6's caption now reads "Theoretical worst-case memory reference … not a witnessed empirical floor for any classical baseline studied here."

**Changes made.** §3.3 "Memory references" paragraph and Table 3 row; §Experiment 6 opening and panel-(a) text; Figure 6 caption; §Discussion implications item (1) ("All three factors are properties of the proxy model and the cited theory; none is a measured quantum result").

## Theme 5 — Only two real datasets; external validity

> *[Insert verbatim comment]*

**Response.** We now state in Experiment 8 that "with only two datasets, this experiment is illustrative rather than statistically representative," and the Threats to Validity section (External validity) records that generalisation to other domains is unverified. Future Work names the concrete next corpora: network telemetry (flow/latency series), cybersecurity event streams (intrusion/log-anomaly feeds), financial tick and returns series, and large-scale IoT monitoring fleets.

**Changes made.** §Experiment 8 closing paragraph; §Threats to Validity (External validity, items ii and iv); §Future Work item (iii).

## Theme 6 — Target-function dependence; the proxy cannot predict downstream performance

> *[Insert verbatim comment]*

**Response.** The target-function ablation paragraph now states the boundary plainly: "classical learner performance is strongly target-dependent, while the proxy … is target-independent by construction. The proxy therefore cannot universally predict downstream classifier performance; it characterises the correlation structure of the stream … not task-specific learnability." A new Discussion subsection ("Scope of the Diagnostic") enumerates what the landscape can and cannot predict, including the Count-Min memorisation route as a representation-specific effect the proxy cannot anticipate.

**Changes made.** §Experiment 9 target-function paragraph; new §Discussion "Scope of the Diagnostic" subsection; Figure 9 caption.

## Theme 7 — Sensitivity to the constants C and δ

> *[Insert verbatim comment]*

**Response.** Because the proxy is a closed-form function of the measured (τ, r), this analysis required no new stochastic experiments: we re-evaluated the proxy on the same per-seed empirical τ values from the Markov sweep over C in {1, 1.5, 2, 2.5} crossed with δ in {0.01, 0.05, 0.1}. New Table 8 reports the crossover ρ* for all twelve combinations: regime *ordering* is invariant (C and δ enter only through the monotone factor κ = C²log(1/δ)), but ρ* moves from ≈0.38 (C=2.5, δ=0.01) to beyond the sweep range (C=1), with ρ* ≈ 0.82 at the operating point. The text now instructs readers to use the landscape ordinally. The re-analysis script and its JSON output are included in the revision package.

**Changes made.** New "Proxy constants C and δ" paragraph and Table 8 in §Experiment 9; §3.2 proxy-status paragraph and §6.3 justification items (i)–(ii) updated to cite Table 8; §Threats to Validity (Statistical-conclusion validity).

## Theme 8 — Figure legibility and framing

> *[Insert verbatim comment]*

**Response.** All nine figures were re-audited. Two figure-internal texts required changes: the Experiment 5 legend entry "Proxy-model advantage (empirical τ)" is now "Proxy-estimated separation (empirical τ)", and five panel titles contained a stray typographic backslash ("vs.\\") now corrected. Figures 4, 5, and 6 were regenerated from the same seeded pipeline (numerical outputs unchanged). Captions of Figures 3, 5, 6, 8, and 9 were rewritten to match the diagnostic framing (e.g., Figure 3: "The landscape is a hypothesis-generation map: contour values are proxy estimates under the modelling assumptions, not measured quantum performance").

**Changes made.** figures/fig4, fig5, fig6 (regenerated); captions of Figures 3, 5, 6, 8, 9; Tables 1, 4, 5, 6 captions/headers tightened (also removing all overfull-width warnings).

## Theme 9 — Missing systematic limitations treatment

> *[Insert verbatim comment]*

**Response.** The former Limitations list is replaced by a dedicated **Threats to Validity** section with construct, internal, external, and statistical-conclusion validity paragraphs, each stating the threat, the mitigation, and the residual risk — including the deterministic-vs-stochastic comparison design, synthetic-generator selection, feature binarisation (arcsine shrinkage makes real-stream (τ, r) estimates lower bounds), dataset selection, target definition, and the n = 10 seed budget.

**Changes made.** New §Threats to Validity (four paragraphs); §Future Work promoted to its own section.

## Theme 10 — Conclusion overstates the contribution

> *[Insert verbatim comment]*

**Response.** The Conclusion is rewritten. It now opens by naming the primary contribution (the (τ, r) sensitivity landscape) and the secondary contribution (proxy-based hypothesis generation), states explicitly that no quantum advantage is demonstrated and no quantum algorithm implemented, answers the four research questions in effect-size terms, adds a practical-contribution paragraph (the landscape as a cheap pre-screening step, illustrated by the Machine Temperature stream at T_eff = 52), and closes with the required validation steps — foremost an implemented or simulated oracle-sketching pipeline evaluated at the landscape's predicted-favourable and predicted-unfavourable coordinates.

**Changes made.** §Conclusions (full rewrite); §Future Work item (i).

---

## Verification statement

All stochastic results are unchanged: the revision alters framing, statistics presentation, and figure text only. The Experiment 5 per-seed matrices were reproduced bit-exactly before the new effect sizes were computed (all five published Wilcoxon p-values match to six significant digits), and the regenerated figures use the same seeded pipeline. The full re-analysis script and output accompany the revision (revision/exp5_reanalysis.py, revision/exp5_reanalysis.json).
