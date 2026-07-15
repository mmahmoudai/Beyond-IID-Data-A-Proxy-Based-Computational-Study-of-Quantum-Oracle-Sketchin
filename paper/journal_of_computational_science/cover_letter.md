# Cover Letter — Journal of Computational Science

14 July 2026

Dear Editors of the *Journal of Computational Science*,

On behalf of my co-authors, I am pleased to submit our original research article, **"Beyond IID Data: Proxy-Based Computational Modelling of Quantum Oracle Sketching Robustness for Structured Non-IID Streams"**, for consideration in the *Journal of Computational Science*.

**What the paper does.** The paper is a computational-modelling study in the journal's core tradition: analytic heuristic models combined with multi-seed stochastic simulation to map a design space that is not yet accessible to direct experimentation. The subject is quantum oracle sketching — a recently proven route to exponential memory advantage in streaming learning — and the question is how its correlated-data robustness behaves under five structured non-IID regimes (Markov switching, seasonal drift, burst repetition, long-range dependence, and an IID control). We specify analytic correlation proxies (refreshing time τ, repetition number r), validate them against empirical estimators on generated streams (10 seeds, 95% CIs, paired Wilcoxon tests), benchmark three classical memory-bounded learners, and condense the results into a (τ, r) sensitivity landscape that practitioners can use to place their own streams. Analytic proxies overestimate empirical refreshing time by up to 125× for long-memory data; the proxy-model advantage closes at Markov strength ρ ≈ 0.88. A two-dataset real-stream check (Numenta Anomaly Benchmark) demonstrates the pipeline end-to-end on real telemetry.

**Fit to the Journal of Computational Science.** The contribution is methodological computational science: transparent proxy models, reproducible simulation, sensitivity and ablation analyses (estimator choice, window size, stream length, target function), and an operational visual artefact — rather than new theory or hardware results. This is precisely the simulation-and-modelling scope of JoCS.

**Declarations.** The manuscript is original, unpublished, and not under consideration elsewhere. All authors approved the submission. No competing interests; no external funding. A generative-AI-assistance declaration is included per Elsevier policy. All code and data are open (Zenodo DOI: 10.5281/zenodo.19831893, MIT licence); every figure and table regenerates from seeded scripts.

*Suggested reviewers: [to be entered by the authors in Editorial Manager].*

Thank you for considering our work.

Sincerely, on behalf of all authors,

**AbdelMoniem Helmy** (corresponding author)
Department of Information Systems and Technology, Faculty of Graduate Studies for Statistical Research, Cairo University, Cairo, Egypt
abdelmoniem.hafez@cu.edu.eg · ORCID 0000-0001-5996-6019

Co-authors: Mohammed Farsi (Taibah University, Yanbu, Saudi Arabia); Muhammad Mahmoud (Matrouh University, Egypt)
