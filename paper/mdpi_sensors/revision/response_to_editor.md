# Response to the Academic Editor

**Manuscript:** *Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for Non-IID Sensor and Telemetry Streams*\
**Journal:** Sensors (MDPI) · **Manuscript ID:** sensors-4470240

---

**Editor's comment.** *"I strongly invite the Authors to not include preprints among references. These works are still undergoing the peer-review process, thus meaning that their content may change prior to publication (if they will be finally published). The Authors can substitute them with related works already published."*

*Response.* We thank the Academic Editor for this instruction. We agree with the principle behind it and have acted on it: the entire reference list was audited entry by entry, every preprint that has since been published has been replaced by its peer-reviewed version, and the reference list now contains **one** preprint, which is the specific framework whose robustness this paper studies. Below we report what the audit found, explain why that single entry cannot be substituted without misattributing its results, and describe the safeguards — including a new footnote added in response to this comment — that ensure no conclusion of this paper depends on it.

## 1. What the audit found

Every one of the 54 references was classified by publication type (journal article, conference paper, preprint, report, thesis, book, website) and then verified at the metadata level: all 51 entries carrying a DOI were resolved through doi.org content negotiation and machine-compared against the cited title, first author, venue, and year; the three entries without a DOI were checked against their publisher's own record.

| Outcome | Count |
|---|---|
| References audited | 54 |
| Preprints present before the audit | 2 |
| **Preprints replaced with the published version** | **1** |
| Preprints remaining (no published version exists) | 1 |
| DOIs resolved and metadata-matched | 51 / 51 |
| Metadata errors found and corrected | 1 |

**Replaced.** Kallaugher, Parekh and Voronova, *How to design a quantum streaming algorithm without knowing anything about quantum computing*, previously cited as an arXiv preprint, has since appeared in the *Proceedings of the 2025 Symposium on Simplicity in Algorithms (SOSA)*, pp. 9–45. The entry now carries the publisher DOI 10.1137/1.9781611978315.2, and the author order was confirmed against the publisher's article page and the proceedings table of contents.

**Corrected.** The audit also caught an unrelated defect that would otherwise have reached print: the entry for Su, Guo and Wang (2024) carried a transposed article number whose DOI resolved to a different paper. It now reads *Information Sciences* **676**, article 120799, DOI 10.1016/j.ins.2024.120799, verified against the publisher's record.

## 2. The one preprint that remains

The remaining entry is Zhao et al., *Exponential quantum advantage in processing massive classical data* (arXiv:2604.07639), labelled "Preprint" in the bibliography.

We confirmed — and, in preparing this reply, re-confirmed — that no peer-reviewed version exists to substitute. As of 16 August 2026: the arXiv record remains at version 1 with no journal reference and no related DOI; Crossref, Semantic Scholar, dblp, INSPIRE-HEP, Google Scholar, and OpenReview all record the work as arXiv-only; it appears in none of the accepted-papers lists of FOCS 2026, STOC 2026, QIP 2026, TQC 2026, CCC 2026, or SODA 2026, and on no publisher page of *Quantum*, *PRX Quantum*, *npj Quantum Information*, *Nature*, or *Science*; the publication pages of all seven authors list it as arXiv-only; and none of the works citing it records a venue or an in-press status for it.

We respectfully submit that this entry is a different case from the one the Editor's comment is aimed at. It is not cited as supporting evidence for a claim of ours, where a related published work could stand in its place. **It is the object of study.** This paper examines how the correlation assumptions of that specific framework — its refreshing time and repetition number, its ×R sample overhead, its task-specific classical hardness results — behave when the data stream is not IID. Substituting a related published work would attribute those particular theorems to authors who did not prove them, and removing the citation would leave the framework we analyse unattributed. Either course would introduce a citation-integrity problem more serious than the one the instruction is intended to prevent.

## 3. Why no conclusion of this paper depends on it

The Editor's concern is that a preprint's content may change before publication. We would like to show concretely why that risk does not propagate into this paper's findings:

1. **No quantum algorithm is implemented or simulated.** The manuscript states this verbatim in the Abstract, Section 3.3, four separate results subsections, the Threats to Validity section, and the Conclusions. Nothing in the paper executes, reproduces, or empirically confirms any result of the preprint.
2. **Every inherited quantity is tabulated with its provenance.** Table 1 (Section 3.3) lists, per quantity, what is inherited from the framework, what this paper introduces, and what is *not* theoretically guaranteed. If any theorem of the preprint changes, the affected rows are identifiable at a glance rather than diffused through the text.
3. **The paper's own contribution is independent of it.** The (τ, r) sensitivity landscape, the classical streaming baselines, the correlation diagnostics, and the two real-telemetry case studies are classical measurements that stand on their own regardless of the preprint's fate.
4. **The revision already removed any dependence on the framework's claims being settled.** Following the reviewers' recommendations, the manuscript was repositioned as a proxy-based diagnostic and hypothesis-generation framework; it treats the framework's results as premises to be stress-tested, not as established facts to be relied upon, and states plainly that it "demonstrates no quantum advantage and implements no quantum algorithm".
5. **Its theorems are never invoked alone.** Each invocation is accompanied by peer-reviewed anchors: Kallaugher, *A quantum advantage for a natural streaming problem* (IEEE FOCS 2021, pp. 897–908); Kallaugher, Parekh and Voronova (SOSA 2025, pp. 9–45); and Huang et al., *Quantum advantage in learning from experiments* (*Science* **376**, 1182–1186, 2022). These are precisely the "related works already published" the Editor suggests, and they now carry the general claims about quantum streaming separations wherever such claims are made.

## 4. What we have changed in response to this comment

Beyond the audit, we have reduced the manuscript's reliance on the preprint to the minimum that honest attribution permits, and strengthened the published anchors around it:

1. **Citation reduction (eight occurrences → six).** Two repeat citations carried no independent attribution and were consolidated: a second citation within the same paragraph of Section 3.3 now reads "the originating framework", and the quantum-memory bullet in Section 5.3 now points to Section 3.3, where the provenance is established. The preprint is now cited only where attributing its specific results requires it.
2. **A stronger published anchor added.** We added Kallaugher, Parekh and Voronova, *Exponential quantum space advantage for approximating maximum directed cut in the streaming model* (Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC 2024, pp. 1805–1815, DOI 10.1145/3618260.3649709) — the first exponential quantum–classical space separation for a natural streaming problem, and the strongest peer-reviewed result of the kind the preprint extends. It is co-cited in the Related Work and in the disclosure footnote (bibliography now 55 entries; citation–bibliography consistency re-verified).
3. **A corrected attribution.** In verifying the anchors we found that our Related Work sentence credited Kallaugher's 2021 FOCS paper with the *first exponential* separation; that paper's advantage is polynomial (the exponential result for a natural streaming problem is the STOC 2024 paper above). The sentence now attributes each result correctly — a further accuracy improvement prompted directly by the Editor's comment.

At the first citation of the preprint in the Introduction we have also added a footnote making its status explicit to every reader:

> "This reference is a preprint (arXiv:2604.07639) and, at the time of writing, had not appeared in a peer-reviewed venue; it is labelled as such in the bibliography. It is cited because it is the specific framework whose robustness under correlated data this paper examines, so no already-published work can stand in for it without misattributing its results. No conclusion of this paper depends on its final published form: no quantum algorithm is implemented or simulated here, every quantity inherited from the framework is listed with its provenance and epistemic status in Table 1, and its theorems are invoked alongside peer-reviewed results on quantum streaming separations and on experimentally demonstrated quantum advantage."

We also undertake to monitor the record and, should the work appear in a peer-reviewed venue before this manuscript reaches proofs, to replace the entry with the published version at that stage.

## 5. If the Editor still prefers removal

We recognise that this remains the Editor's decision. We would only note that removing the citation entirely would require either dropping the attribution for the framework the paper analyses, or presenting its assumptions as our own — neither of which we are willing to do. If the Editor judges that the single labelled preprint is nevertheless unacceptable, we would be grateful for guidance on the preferred remedy, and we will follow it.

We are grateful for the attention the Editor has given the reference list; the audit prompted by this comment improved the manuscript beyond the preprint question itself, and every one of the 54 references has now been verified against its publisher's record.

On behalf of all authors,

**AbdelMoniem Helmy** (corresponding author)
abdelmoniem.hafez@cu.edu.eg · ORCID 0000-0001-5996-6019
