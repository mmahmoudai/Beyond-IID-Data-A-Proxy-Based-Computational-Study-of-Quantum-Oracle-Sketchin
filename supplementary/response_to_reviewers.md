# Response to Reviewers and Revision Summary

**Manuscript:** *Beyond IID Data: A Proxy-Based Computational Study of Quantum Oracle Sketching Robustness under Structured Non-IID Streaming*

This document accompanies the submission and summarises the revision work performed across multiple internal review cycles. It is **not** part of the manuscript body — it is intended for the editor and reviewers as transparency on how prior concerns were addressed. If your venue prefers a cover-letter format instead, the same content can be merged into the cover letter.

---

## Revision summary

This version of the manuscript extends earlier proxy-only treatments with:

1. **Real-stream sanity check on two NAB datasets** (Section 6.8). NYC Taxi falls in the proxy-favourable region of the landscape (Count-Min tracks the proxy within ≈ 2 percentage points; SGD baselines are task-limited). Machine Temperature falls in the unfavourable region where the proxy saturates and is exceeded by all classical baselines. Both datasets were chosen *before* examining the results; honest reporting of the negative case (Machine Temperature) is included as a genuine data point.

2. **Four ablation families** (Section 6.9): alternative τ estimators (1/e default, 1/2-crossing, integrated ACF, AR(1) effective), forward-window sizes W ∈ {10, 20, 50}, stream lengths T ∈ {1024, 2048, 4096, 8192}, and target functions (linear, parity, threshold). The qualitative regime ordering is preserved across all alternative τ estimators (proxy values shift by at most 0.04).

3. **Wilcoxon test interpretation clarified** (Section 6.5). The paired test compares per-seed empirical classical accuracy against the deterministic proxy curve at each ρ; the p-value tests crossover stability under seed variability, not algorithm-versus-algorithm significance.

4. **Code repository with end-to-end reproduction recipe**, deposited at Zenodo with DOI `10.5281/zenodo.19831893` and mirrored on GitHub. Pinned `requirements.txt`, three reproduction scripts (`run_experiments.py`, `real_streams.py`, `ablations.py`), and the two NAB datasets are bundled.

5. **Strengthened proxy-status statements** throughout (abstract, scope, §3.2, §4, §6.8, §7.1, §7.3, §8). No remaining own-voice claim of empirical quantum advantage. The single explicit quantum-advantage statement in the abstract is attributed to Zhao et al. with citation.

---

## Summary of recommendations executed

The following table summarises major-revision recommendations from prior review cycles, the form of the response, and the residual limitation in each row.

| Previous concern | Response in this version | Remaining limitation |
|---|---|---|
| Lack of real data | NYC Taxi and Machine Temperature NAB case studies (Section 6.8) | Only two real streams; no financial or network-traffic data |
| Ad hoc proxies | Section 6.9 ablations across four τ estimators, W, T, and target functions | No formal theorem-equivalence proof |
| Statistical ambiguity | Wilcoxon paragraph clarifies what is paired and what the p-value tests (Section 6.5) | Proxy is deterministic; tests describe baseline-vs-proxy stability only |
| Reproducibility | Code directory, pinned `requirements.txt`, end-to-end reproduction recipe, and Zenodo-DOI-archived release `10.5281/zenodo.19831893` | GitHub repository will keep evolving; the cited DOI snapshot is the canonical reproducibility artefact |
| Fairness of comparison | Proxy/status disclaimers strengthened across abstract, scope, and §6.5 | Classical task is still simple linear classification, not the hard task class |

---

## Code and data availability

- **Code archive (DOI):** [10.5281/zenodo.19831893](https://doi.org/10.5281/zenodo.19831893) — versioned snapshot, MIT licence
- **GitHub mirror:** <https://github.com/mmahmoudai/Beyond-IID-Data-A-Proxy-Based-Computational-Study-of-Quantum-Oracle-Sketchin>
- **Datasets:** NYC Taxi and Machine Temperature, both from the Numenta Anomaly Benchmark (Ahmad et al. 2017), MIT licence, bundled in `data/raw/` of the archive

Reproducing every figure and table:

```bash
python code/run_experiments.py     # Figures 1-7 + Table 3
python code/real_streams.py        # Figure 8 + Table 4 (real-stream sanity check)
python code/ablations.py           # Figure 9 + Table 6 (sensitivity ablations)
```

Total runtime: ≈ 15–20 minutes on a standard laptop. Every stochastic experiment uses 10 seeds (indices 0–9) and reports mean plus 95% CI half-width.
