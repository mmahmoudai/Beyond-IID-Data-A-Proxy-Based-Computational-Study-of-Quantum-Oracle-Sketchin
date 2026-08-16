# Zhao et al. (2026) Minimal-Diff Removal Audit

Date: 16 August 2026  
Working branch: `remove-zhao-minimal`  
Protected parent: `zhao-preprint-fix` at `52275f838528fcb1a7b03dc448b81091e1243b92`

## A. Old-version protection

- The new branch was created directly from the current final-manuscript commit.
- `zhao-preprint-fix` still points to `52275f838528fcb1a7b03dc448b81091e1243b92`.
- `mdpi-revision-r1` still points to `14c9016718e19a6a312cb9b6a1b5c85cf8ffe714`.
- The existing `manuscript_revised.tex`, `manuscript_revised.pdf`, `tracked_changes.*`, all reviewer responses, and the existing editor response were not edited.
- Protected source SHA-256: `10626FE6CBF177FFBA2060B297069635738959B903B6F8597A7CF892E99202A6`.
- Protected PDF SHA-256: `68D7C2E2E16C47E5C5F7438445C0312B36D8B1DFBF03974443D74F2E4564E4C3`.

## B. Phase 1 citation map (audit performed before editing)

The citation key was `zhao2026exponential`: five in-text uses plus the bibliography key (six key occurrences). The first citation contained a separate preprint-status footnote. Directly dependent uncited clauses and Table 1 were also audited.

| Location | Exact original Zhao-dependent statement | Scientific purpose | Can remain without Zhao? | Existing published support for the exact claim? | Disposition |
|---|---|---|---|---|---|
| Introduction, original line 72 | “Recent work by `\citet{zhao2026exponential}` establishes, at the level of theory, exponential quantum advantages for classification and dimension reduction on massive classical data via quantum oracle sketching, using a polylogarithmic-size quantum computer (`O(n)` qubits for `n = log_2 N`).” | Introduce QOS, its tasks, and claimed resource advantage. | No, not as written. | No. Kallaugher 2021/2024/2025 support different streaming results. | Replaced with accurate published quantum-streaming background and an explicit manuscript-level definition of the retained term “quantum oracle sketching.” |
| Introduction footnote, original line 72 | “This reference is a preprint (arXiv:2604.07639) ... it is the specific framework whose robustness under correlated data this paper examines ... no already-published work can stand in for it without misattributing its results.” | Disclose preprint status and justify retention. | No; it becomes obsolete once the preprint is removed. | Not applicable. | Footnote removed completely. |
| Introduction, original line 74 (dependent uncited paragraph) | “The framework extends to correlated, non-IID data via a hierarchical data-generation model in which the quantum sample complexity is multiplied by the repetition number ...” | Motivate non-IID analysis and `r`. | The motivation can remain; the theorem cannot. | No exact published substitute found. | Rewritten as a general effective-information motivation; `tau` and `r` are stated as paper-defined operational quantities. |
| Related Work, original line 110 | “The quantum oracle sketching framework of `\citet{zhao2026exponential}` avoids qRAM entirely and supports exponential separations for classification and dimension-reduction tasks.” | Contrast QOS with qRAM-dependent work. | No. | No. | Zhao-specific sentence removed; the verified Kallaugher summaries remain unchanged. |
| Related Work, original line 114 | “The quantum oracle sketching framework `\citep{zhao2026exponential}` is, to our knowledge, the first to address noisy, correlated, and time-varying classical data in a quantum streaming setting.” | Novelty/priority claim and provenance for correlated-data discussion. | No. | No. | Priority claim removed; replaced by the narrower statement that this diagnostic question is comparatively underexplored and that the proxies are paper-defined. |
| Preliminaries, original line 124 | “The originating work `\citep{zhao2026exponential}` shows that a polylogarithmic-size quantum computer ... can perform classification and dimension reduction ... Their correlated-data extension ... [motivates] refreshing time and repetition number ...” | Define QOS and give provenance for terminology, task claims, memory gap, and correlation quantities. | Only the manuscript’s operational concept can remain. | No exact substitute found. | Replaced with an explicit definition: published streaming/sketching work supplies general motivation, while QOS, `tau`, `r`, and the performance proxy used here are manuscript-level constructs. |
| Section 3.3, original line 163 | “The exact framework `\citep{zhao2026exponential}` proves three results ... `O(n)` qubits ... repetition-number sample overhead ... `O(N^c)` classical hardness.” | Establish the theoretical provenance boundary and interpret the landscape. | No, not as a theorem statement. | No. | Replaced with exact, published Kallaugher results and a boundary statement that those papers do not establish this manuscript’s operational model. All Zhao-derived `r`-only consequences were removed. |
| Table 1, original lines 173–185 | “`x R` sample overhead; `O(n)` qubits; `O(N^c)` classical hardness — established theory” and dependent equivalence/guarantee cells. | Provenance and epistemic-status summary. | No. | No. | The Zhao-derived row and cells were relabelled as paper-defined illustrative or operational constructs. No unrelated row changed. |
| Bibliography, original lines 1054–1059 | Zhao et al. (2026), title, arXiv ID, preprint status, and URL. | Bibliographic record. | No. | Not applicable. | Entry removed completely. |

Search coverage before editing included `2604.07639`, the exact title, `Zhao 2026`, `Zhao et al.`, both capitalisations of “Quantum Oracle Sketching framework,” “refreshing time,” “repetition number,” the citation key, and the bibliography entry. Searches covered the final manuscript source and the other submission-package text sources; only the newly created manuscript was edited.

## C. Phase 2 published-substitute audit

No candidate supports the removed framework-specific claims. No candidate was used as a mechanical replacement.

| Candidate | Verified metadata and exact result | Replacement verdict |
|---|---|---|
| Kallaugher, FOCS 2021, “A Quantum Advantage for a Natural Streaming Problem” | DOI [10.1109/FOCS52979.2021.00091](https://doi.org/10.1109/FOCS52979.2021.00091). One-pass triangle counting with a polynomial quantum space advantage in specified parameter regimes. | Valid general quantum-streaming background only; not QOS classification/dimension reduction, correlated-data theory, or Zhao’s overhead/hardness theorem. |
| Kallaugher, Parekh & Voronova, STOC 2024, “Exponential Quantum Space Advantage for Approximating Maximum Directed Cut in the Streaming Model” | DOI [10.1145/3618260.3649709](https://doi.org/10.1145/3618260.3649709). Exponential quantum space advantage for maximum directed cut, a natural streaming problem. | Valid for that exact general statement only; not a substitute for QOS-specific claims. |
| Kallaugher, Parekh & Voronova, SOSA 2025, “How to Design a Quantum Streaming Algorithm Without Knowing Anything About Quantum Computing” | DOI [10.1137/1.9781611978315.2](https://doi.org/10.1137/1.9781611978315.2). Published black-box quantum-sketch construction for applicable classical sketches. | Valid general sketching motivation only; not a QOS theorem substitute. |
| Huang et al., Science 2022, “Quantum Advantage in Learning from Experiments” | DOI [10.1126/science.abn7293](https://doi.org/10.1126/science.abn7293). Quantum advantage for learning properties/dynamics of quantum systems and quantum PCA from experiments. | Different data and learning setting; not a substitute. |
| Zhao & Deng, npj Quantum Information 2025, “Entanglement-induced provable and robust quantum learning advantages” | DOI [10.1038/s41534-025-01078-x](https://doi.org/10.1038/s41534-025-01078-x), published 29 July 2025. | Different entanglement-based quantum-learning result; no verified QOS streaming, refreshing-time, repetition-overhead, or classical-hardness result. Not added. |

## D. Citation replacement report

1. Introduction theorem/footnote: replaced by problem-specific published quantum-streaming background; no claim that those papers establish QOS. The footnote was deleted.
2. Introduction correlated-data clause: changed from a Zhao theorem to a general effective-information motivation and paper-defined terminology.
3. Related Work qRAM/classification clause: deleted only the Zhao-specific sentence.
4. Related Work priority claim: removed the “first” claim; retained the scientific gap in narrower form.
5. Preliminaries provenance paragraph: redefined retained QOS terminology at manuscript level; kept `tau` and `r` definitions unchanged.
6. Section 3.3 theorem paragraph: replaced by exact Kallaugher result summaries and an explicit non-equivalence boundary; removed unsupported sample-overhead and hardness consequences.
7. Table 1: reclassified Zhao-derived theory as paper-defined illustrative/operational content.
8. Bibliography: removed Zhao 2026 without replacement.

## E. Reference audit

- Old count: 55.
- New count: 54.
- Removed: Zhao, H. et al. (2026), “Exponential quantum advantage in processing massive classical data,” arXiv:2604.07639.
- Added: none.
- New DOI verification: not applicable because no reference was added.
- Candidate DOI metadata: verified as listed in Section C.
- Citation consistency: 0 dangling keys; 0 orphan bibliography entries.
- Other Zhao authors: P. Zhao remains correctly in Hoi et al. (2021); no independently valid Zhao paper was removed.

## F. Complete diff report

The complete source-to-source diff was generated with:

```text
git diff --no-index -- manuscript_revised.tex manuscript_remove_zhao_minimal.tex
```

Diff size: 34 inserted lines, 40 deleted lines, 33 zero-context hunks. Because the LaTeX source stores most paragraphs as single long lines, each hunk corresponds to one targeted paragraph/caption/table block or the bibliography removal. No section was broadly rewritten.

Every changed block is listed below:

1. Abstract (line 58): two unavoidable Zhao-dependent clauses corrected; all result statements retained.
2. Introduction (line 72): direct Zhao theorem, footnote, and misapplied hardness clause replaced with verified general background and the manuscript-level QOS definition.
3. Introduction (line 74): hierarchical/repetition theorem replaced with general operational motivation.
4. Introduction Scope (line 78): “inspired by this framework” changed to “introduced in this paper”; follow-on “theorem reproduction” changed to “formal analysis.”
5. Related Work (line 110): Zhao-specific final sentence deleted.
6. Related Work (line 114): Zhao priority claim removed; proxy ownership clarified.
7. Preliminaries (line 124): QOS and terminology provenance rewritten minimally.
8. Heuristic model proxy-status paragraph (line 129): exact-framework attribution removed.
9. Section 3.3 lead (line 161): “inherits from established theory” changed to published-background/assumption/construction wording.
10. Section 3.3 published-theory paragraph (line 163): Zhao theorems removed; exact published Kallaugher boundary inserted.
11. Section 3.3 assumptions paragraph (line 165): removed inheritance claim only.
12. Section 3.3 heuristic-constructions paragraph (line 167): removed Zhao overhead/oracle-cost comparisons; equations and interpretation retained.
13. Section 3.3 memory-references paragraph (line 169): Zhao scalings relabelled as illustrative paper-defined curves.
14. Table 1 caption (line 173): provenance terminology corrected.
15. Table 1 first three rows (lines 179–181): Zhao theorem/equivalence cells corrected.
16. Table 1 epsilon row (line 183): oracle-construction guarantee changed to any-quantum-algorithm guarantee.
17. Table 1 memory row (line 185): theoretical-reference status corrected.
18. Section 3.3 scope-limitations paragraph (line 190): exact-framework comparison removed.
19. Correlation-proxy section lead (line 198): exact-framework equivalence changed to formal effective-information bounds.
20. Markov provenance paragraph (line 217): exact-framework equivalence changed to unvalidated effective-information use.
21. Seasonal paragraph (line 223): “quantum oracle construction” changed to “proxy diagnostic.”
22. Experimental-framework comparison paragraph (line 282): exact-framework hard instances changed to published problem-specific separations.
23. Modelling-assumptions memory bullet (line 309): “established theory” changed to illustrative curve.
24. Experiment 2 interpretation (line 372): unsupported lower-bound description changed to paper-defined illustrative curve.
25. Experiment 6 opening paragraph (line 435): theoretical/cited-quantity wording corrected.
26. Figure 6 caption (line 440): curves marked illustrative; legacy internal artwork labels explicitly disclaimed. Figure artwork was not changed.
27. Experiment 6 interpretation (line 444): Zhao lower-bound and oracle-construction claims removed; experimental observations retained.
28. Discussion (line 619): theoretical-separation description changed to an explicit within-proxy extrapolation.
29. Threats—construct validity (line 670): exact-framework equivalence/oracle-cost claims removed.
30. Threats—internal validity (line 672): Zhao hard instances changed to the problems addressed by published streaming literature.
31. Threats—external validity (line 674): unsupported `n >= 20` theoretical-memory claim removed.
32. Future Work (line 684): formal bridging to Zhao changed to formal foundations for the paper-defined proxies.
33. Bibliography (original lines 1054–1059): Zhao entry removed.

The title is byte-identical. The Conclusion section is byte-identical. The abstract changed only where unavoidable. The core methodology and all operational definitions remain byte-identical.

## G. Final QA

| Check | Result | Evidence |
|---|---|---|
| Zhao 2026 completely removed | PASS | Zero source/PDF occurrences of the arXiv ID, exact title, citation key, or bibliography entry. |
| No fake substitution | PASS | Candidate audit above; no reference added. |
| Reviewer revisions preserved | PASS | New source is a copy of the final revised source with only the 33 listed targeted hunks. Existing tracked-changes and response files are untouched. |
| Title preserved | PASS | Exact source hash comparison of the title command matches. |
| Abstract preserved except unavoidable clauses | PASS | Only the Zhao-dependent theoretical opening and provenance phrase changed; all methods/results statements remain. |
| `tau`, `r`, `Teff`, epsilon, and accuracy-proxy definitions | PASS | Both definition environments and all eight equation environments are byte-identical. |
| Numerical results | PASS | All result tables except provenance Table 1 are byte-identical; no measured table or figure data changed. |
| Figures | PASS | No figure asset changed. Figure 6’s legacy internal wording is transparently qualified in its caption. |
| Tables | PASS | Only `tab:provenance` changed; the other nine table environments are byte-identical. |
| Conclusions | PASS | Conclusion-section source hash is identical. |
| Citation/reference integrity | PASS | 54 bibliography entries, 0 dangling citations, 0 orphans. |
| Compilation | PASS | Three-pass `pdflatex` build; 29 pages; no LaTeX errors, undefined citations/references, or overfull boxes. |
| Visual PDF QA | PASS | Front matter, Introduction, Section 3.3/Table 1, Figure 6/caption, Discussion/Threats/Future Work, and final References page rendered and inspected. |

The separate new editor-response source and PDF were created only after the manuscript passed the source, citation, compilation, and visual checks.
