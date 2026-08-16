# Response to the Academic Editor

**Manuscript:** *Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for Non-IID Sensor and Telemetry Streams*\
**Journal:** Sensors (MDPI) · **Manuscript ID:** sensors-4470240

---

**Editor's comment.** *"I strongly invite the Authors to not include preprints among references. These works are still undergoing the peer-review process, thus meaning that their content may change prior to publication (if they will be finally published). The Authors can substitute them with related works already published."*

*Response.* We thank the Academic Editor for this instruction and agree with the principle behind it. We acted on it in three steps. First, we audited every reference and replaced every preprint for which a published version exists. Second, immediately before submitting this revision, we re-verified that the single remaining preprint still has no published version anywhere. Third, we reduced the manuscript's reliance on that one entry to the minimum that honest attribution permits, and strengthened the peer-reviewed anchors around it. The result: of the 55 references in the revised manuscript, exactly one is a preprint — the specific framework this paper studies — explicitly labelled as such, cited only where attributing its results requires it, and load-bearing for none of the paper's conclusions.

## 1. Audit and re-verification

Before answering the substance of the comment, we audited the reference list in full. Each of the 54 references in the reviewed manuscript was classified by publication type, and every entry was then checked against the publisher's own metadata. For the 51 entries that carry a DOI, we resolved the DOI itself and compared the registered title, first author, venue, and year with what we cite; the three entries without a DOI were checked directly against the publisher's records. All 51 DOIs resolved and matched. The audit found two preprints among the 54 references.

The first of the two had in fact been published already, and we have replaced it. The Kallaugher, Parekh and Voronova paper on designing quantum streaming algorithms, previously cited through its arXiv posting, now points to its published version in the *Proceedings of the 2025 Symposium on Simplicity in Algorithms (SOSA)*, pp. 9–45, DOI 10.1137/1.9781611978315.2, and we confirmed the author order against the publisher's article page and table of contents. The same sweep caught one further defect, unrelated to preprints, that would otherwise have reached print: the entry for Su, Guo and Wang (2024) carried a transposed article number whose DOI led to a different paper altogether. That entry now reads *Information Sciences* **676**, article 120799, DOI 10.1016/j.ins.2024.120799, again verified against the publisher's record.

The second preprint is the Zhao et al. reference discussed below. Before writing this reply we checked once more, on 16 August 2026, whether it has been published or accepted anywhere in the meantime. It has not. The arXiv record is still at version 1, with no journal reference and no related DOI. Crossref, Semantic Scholar, dblp, INSPIRE-HEP, Google Scholar and OpenReview all list the work as arXiv-only. It does not appear in the accepted-paper lists of FOCS, STOC, QIP, TQC, CCC or SODA for 2026, nor on any page of *Quantum*, *PRX Quantum*, *npj Quantum Information*, *Nature* or *Science*. The publication pages of all seven of its authors give only the arXiv posting, and none of the papers citing it names a venue or an in-press version. After the changes described in this reply, the bibliography contains 55 entries, of which this is the only preprint, and we have re-checked that every in-text citation matches exactly one bibliography entry.

## 2. The one preprint that remains — and why substitution is not available

The remaining entry is Zhao et al., *Exponential quantum advantage in processing massive classical data* (arXiv:2604.07639), labelled "Preprint" in the bibliography.

We respectfully submit that this entry is a different case from the one the Editor's instruction is aimed at. It is not cited as supporting evidence for a claim of ours, where a related published work could stand in its place. **It is the object of study.** This paper examines how the correlation assumptions of that specific framework — its refreshing time and repetition number, its ×R sample-overhead theorem, its task-specific classical hardness results — behave when the data stream is not IID. Substituting a related published work would attribute those particular theorems to authors who did not prove them; removing the citation would leave the framework the paper analyses unattributed. Either course would introduce a citation-integrity problem more serious than the one the instruction is intended to prevent.

We did examine the closest published works in this literature as potential substitutes — including peer-reviewed results on learning quantum observables from classical data, on the circuit cost of quantum access models, and on quantum streaming advantages. None of them contains the framework studied here; where they support adjacent general claims, the manuscript already lets published references carry those claims, as described next.

## 3. What we changed in the manuscript

1. **In-text citations of the preprint reduced from eight to six.** Two repeat citations carried no independent attribution and were consolidated: a second citation within the same paragraph of Section 3.3 now reads "the originating framework", and the quantum-memory bullet in Section 5.3 now points to Section 3.3, where the provenance is established. The preprint is now cited only at the sites where attributing its specific results requires it.
2. **A stronger published anchor added, and an attribution corrected.** We added Kallaugher, Parekh and Voronova, *Exponential quantum space advantage for approximating maximum directed cut in the streaming model* (Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC 2024, pp. 1805–1815, DOI 10.1145/3618260.3649709) — the first exponential quantum–classical space separation for a natural streaming problem, and the strongest peer-reviewed result of the kind the preprint extends. It is co-cited in the Related Work and in the disclosure footnote. In verifying the anchors we also found that our Related Work sentence had credited Kallaugher's 2021 FOCS paper with the *first exponential* separation, although that paper's advantage is polynomial; the sentence now attributes each result correctly. The bibliography stands at 55 entries with the citation–bibliography consistency re-verified.
3. **An explicit status footnote at the first citation.** So that the preprint's status is visible to every reader, the first citation in the Introduction now carries the following footnote:

> "This reference is a preprint (arXiv:2604.07639) and, at the time of writing, had not appeared in a peer-reviewed venue; it is labelled as such in the bibliography. It is cited because it is the specific framework whose robustness under correlated data this paper examines, so no already-published work can stand in for it without misattributing its results. No conclusion of this paper depends on its final published form: no quantum algorithm is implemented or simulated here, every quantity inherited from the framework is listed with its provenance and epistemic status in Table 1, and its theorems are invoked alongside peer-reviewed results on quantum streaming separations and on experimentally demonstrated quantum advantage."

## 4. Why no conclusion of the paper depends on its final form

The Editor's underlying concern is that a preprint's content may change before publication. That risk does not propagate into this paper's findings, for five reasons:

1. **No quantum algorithm is implemented or simulated.** The manuscript states this verbatim in the Abstract, Section 3.3, four results subsections, the Threats to Validity section, and the Conclusions; nothing in the paper executes or empirically confirms any result of the preprint.
2. **Every inherited quantity is tabulated with its provenance.** Table 1 lists, per quantity, what is inherited from the framework, what this paper introduces, and what is *not* theoretically guaranteed — so if any theorem of the preprint were to change, the affected rows are identifiable at a glance.
3. **The paper's own contribution is classical and self-standing.** The (τ, r) sensitivity landscape, the streaming baselines, the correlation diagnostics, and the two real-telemetry case studies are classical measurements that hold regardless of the preprint's fate.
4. **The revision already treats the framework as a premise, not a fact.** Following the reviewers' recommendations, the manuscript is repositioned as a proxy-based diagnostic and hypothesis-generation framework that stress-tests the framework's assumptions; it states plainly that it "demonstrates no quantum advantage and implements no quantum algorithm".
5. **Its theorems are never invoked alone.** Every invocation is accompanied by peer-reviewed anchors: Kallaugher (IEEE FOCS 2021, pp. 897–908), Kallaugher, Parekh and Voronova (STOC 2024, pp. 1805–1815), Kallaugher, Parekh and Voronova (SOSA 2025, pp. 9–45), and Huang et al. (*Science* **376**, 1182–1186, 2022) — precisely the "related works already published" the Editor recommends, now carrying the general claims wherever they are made.

## 5. Commitment

We will monitor the record and, should the work appear in a peer-reviewed venue before this manuscript reaches proofs, we will replace the entry with the published version at that stage. If the Editor nevertheless judges that the single labelled preprint is unacceptable, we would be grateful for guidance on the preferred remedy and will follow it — our reservation concerns only the two remedies that would damage attribution (crediting the framework to authors who did not build it, or leaving it uncredited), not the Editor's instruction itself, which has already improved the manuscript: the audit it prompted verified every reference against its publisher's record, corrected one wrong article number, upgraded one citation to its published version, and strengthened the published anchors around the one reference that must remain a preprint.

On behalf of all authors,

**AbdelMoniem Helmy** (corresponding author)
abdelmoniem.hafez@cu.edu.eg · ORCID 0000-0001-5996-6019
