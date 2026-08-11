# Cover Letter — Sensors (MDPI)

11 August 2026

Dear Editors of *Sensors*,

On behalf of my co-authors, I am pleased to submit our original research article, **"Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for Non-IID Sensor and Telemetry Streams"**, for consideration in *Sensors* (suggested section: Sensor Networks / sensor-data processing).

**What the paper does.** Sensor and telemetry streams are rarely IID: they mix short-term Markov structure, seasonal cycles, burst repetition, and long-range dependence. This paper presents a proxy-based diagnostic and hypothesis-generation framework for asking where quantum oracle sketching — a route to exponential memory advantage for streaming learning that has been proven at the level of theory — is most likely to remain robust under exactly these correlation patterns. We introduce two operational, estimator-friendly correlation parameters (refreshing time τ and repetition number r), evaluate five structured stream regimes against three classical memory-bounded baselines (10 seeds, 95% CIs, effect-size-based paired comparisons), and produce a (τ, r) sensitivity landscape on which any measured stream can be placed. The framework is demonstrated end-to-end on two real telemetry streams from the Numenta Anomaly Benchmark — NYC Taxi demand and an industrial machine-temperature sensor series — which land on opposite sides of the hypothesized favourable region, with imbalance-aware metrics (balanced accuracy, F1) exposing majority-class effects. No quantum algorithm is implemented or simulated: all quantum-side quantities are clearly labelled heuristic proxies, the provenance of every quantity is tabulated, and a dedicated Threats to Validity section delimits the claims.

**Fit to Sensors.** The contribution is a data-processing viability map for sensor-style streams: it gives practitioners a cheap, measurement-based procedure (estimate τ and r from a stream sample) to decide whether emerging quantum streaming approaches are worth prototyping for their telemetry workload, and it is validated on genuine sensor data. The manuscript is prepared in the official MDPI template.

**Declarations.** The manuscript is original, unpublished, and not under consideration elsewhere. All authors approved the submission. No competing interests; no external funding (no APC waiver requested). A generative-AI-assistance declaration is included per MDPI policy. All code and data are open (Zenodo DOI: 10.5281/zenodo.19831893, MIT licence); every figure and table regenerates from seeded scripts.

*Suggested reviewers: [to be entered by the authors in the submission system].*

Thank you for considering our work.

Sincerely, on behalf of all authors,

**AbdelMoniem Helmy** (corresponding author)
Taibah University, Yanbu, Saudi Arabia; and Department of Information Systems and Technology, Faculty of Graduate Studies for Statistical Research, Cairo University, Cairo, Egypt
abdelmoniem.hafez@cu.edu.eg · ORCID 0000-0001-5996-6019

Co-authors: Mohammed Farsi (Taibah University, Yanbu, Saudi Arabia); Muhammad Mahmoud (Matrouh University, Egypt). Author order: Farsi, Mahmoud, Helmy.
