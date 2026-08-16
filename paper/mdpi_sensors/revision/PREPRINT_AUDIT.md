# Preprint Reference Audit — MDPI Sensors variant

Date: 2026-08-11 · Branch `mdpi-revision-r1` · Trigger: reviewer request that preprints not be cited when equivalent peer-reviewed publications exist.

**Method.** Every bibliography entry of `main.tex` was classified by publication type. For each preprint-linked entry, a web search verified whether a peer-reviewed version exists (title, authors, venue, pages, DOI checked against dblp / publisher records / the arXiv record). Nothing was fabricated: every replaced field was independently verified; entries whose status could not be verified would have been flagged, and none needed to be.

## Summary

| Metric | Count |
|---|---|
| **Total references** | **54** |
| Peer-reviewed journal articles | 37 |
| Peer-reviewed conference papers | 7 |
| Peer-reviewed monograph series (Foundations & Trends) | 4 |
| Books (academic publishers) | 5 |
| **Preprints detected** | **2** (one pure preprint; one published paper cited via its arXiv link) |
| **Replaced with published version** | **1** (`kallaugher2025designquantum` — pointer corrected to the SIAM DOI) |
| **Retained — no published alternative exists** | **1** (`zhao2026exponential` — flagged below) |
| Requiring author verification | 0 |

Consistency check (post-edit): 54 in-text citation keys ↔ 54 bibliography entries; every citation maps to exactly one entry and every entry is cited at least once; no undefined and no uncited keys. Author–year labels and alphabetical ordering unchanged (the one edit altered only a URL), so no renumbering was needed. Clean recompile confirmed.

## Action detail — the two preprint-linked entries

### 1. `kallaugher2025designquantum` — REPLACED (pointer corrected)

- **Before:** venue and pages already cited correctly (*Proceedings of the 2025 Symposium on Simplicity in Algorithms (SOSA)*, pp. 9–45) **but the URL pointed to the arXiv preprint** (arxiv.org/abs/2410.18922) with no publisher DOI.
- **Verification:** dblp (John Kallaugher, pid 188/6358) lists the SOSA 2025 entry, pp. 9–45, publisher DOI **10.1137/1.9781611978315.2** (SIAM). Venue, authors (Kallaugher, Parekh, Voronova), year, and pages all match the existing entry.
- **Action:** URL replaced with `https://doi.org/10.1137/1.9781611978315.2`. No other field required change. The citation is now fully peer-reviewed with the publisher's DOI.

### 2. `zhao2026exponential` — RETAINED, EXPLICITLY FLAGGED (no published version exists)

- **Entry:** Zhao, Zlokapa, Neven, Babbush, Preskill, McClean, Huang, "Exponential quantum advantage in processing massive classical data", arXiv:2604.07639 — already explicitly labelled "Preprint" in the bibliography entry.
- **Verification (2026-08-11):** searched for a subsequently published peer-reviewed version. Google Research's publication page and the arXiv record list it as arXiv-only; no journal or conference version exists as of this audit.
- **Why it must be retained:** this preprint *is the framework the manuscript studies* — the paper's subject, not a supporting citation. Removing or substituting it is scientifically impossible.
- **Same-claim peer-reviewed support (already co-cited in the manuscript, satisfying the "closely related peer-reviewed publication" requirement):** Kallaugher, FOCS 2021 (DOI 10.1109/FOCS52979.2021.00091 — first quantum space advantage for a natural streaming problem; the advantage there is polynomial, not exponential — corrected 2026-08-16); Kallaugher–Parekh–Voronova, STOC 2024 (DOI 10.1145/3618260.3649709 — establishes an *exponential* quantum–classical space separation for a natural streaming problem; added to the bibliography 2026-08-16); Huang et al., *Science* 376:1182–1186, 2022 (DOI 10.1126/science.abn7293 — experimentally demonstrated exponential advantage in learning); Kallaugher–Parekh–Voronova, SOSA 2025 (DOI 10.1137/1.9781611978315.2 — black-box quantum-sketch construction). Wherever the manuscript invokes the framework's theorems, at least one of these peer-reviewed anchors appears alongside.
- **Author verification needed:** none — status is unambiguous. Recommended maintenance: re-check for a published version at proof stage.

## Full classification table

| # | Key | Citation (short) | Type | Preprint? | Published version found? | Action |
|---|---|---|---|---|---|---|
| 1 | aaronson2015readthefineprint | Aaronson 2015, *Nat. Physics* 11:291 | journal | no | — | none |
| 2 | ahmad2017numenta | Ahmad et al. 2017, *Neurocomputing* 262:134 | journal | no | — | none |
| 3 | alon1999space | Alon–Matias–Szegedy 1999, *JCSS* 58:137 | journal | no | — | none |
| 4 | arora2024driftreview | Arora et al. 2024, *WIREs DMKD* 14:e1536 | journal | no | — | none |
| 5 | arunachalam2018optimal | Arunachalam–de Wolf 2018, *JMLR* 19(71) | journal (JMLR; canonical URL, journal has no DOIs) | no | — | none |
| 6 | benmazziane2022countmin | Ben Mazziane et al. 2022, *Comput. Netw.* 217 | journal | no | — | none |
| 7 | beran1994statistics | Beran 1994, Chapman & Hall book | book | no | — | none |
| 8 | ben2010agnostic | Ben-David et al. 2010, *Mach. Learn.* 79:151 | journal | no | — | none |
| 9 | besbes2015non | Besbes–Gur–Zeevi 2015, *Oper. Res.* 63:1227 | journal | no | — | none |
| 10 | biamonte2017qml | Biamonte et al. 2017, *Nature* 549:195 | journal | no | — | none |
| 11 | bifet2007adwin | Bifet–Gavaldà 2007, SIAM SDM | conference | no | — | none |
| 12 | bifet2010moa | Bifet et al. 2010, *JMLR* 11:1601 | journal | no | — | none |
| 13 | cerezo2021variational | Cerezo et al. 2021, *Nat. Rev. Phys.* 3:625 | journal | no | — | none |
| 14 | chakraborty2019blockencodings | Chakraborty et al. 2019, ICALP (LIPIcs) | conference | no | — | none |
| 15 | chen2024qmlsurvey | Chen et al. 2024, *Connection Science* 36 | journal | no | — | none |
| 16 | cormode2005improved | Cormode–Muthukrishnan 2005, *J. Algorithms* 55:58 | journal | no | — | none |
| 17 | cormode2011synopses | Cormode et al. 2011, *FnT Databases* 4:1 | monograph series | no | — | none |
| 18 | ditzler2015nonstationary | Ditzler et al. 2015, *IEEE CIM* 10(4):12 | journal | no | — | none |
| 19 | gama2010knowledge | Gama 2010, Chapman & Hall book | book | no | — | none |
| 20 | gama2014driftsurvey | Gama et al. 2014, *ACM CSUR* 46(4) | journal | no | — | none |
| 21 | geyer1992practical | Geyer 1992, *Statist. Sci.* 7:473 | journal | no | — | none |
| 22 | ghashami2016frequent | Ghashami et al. 2016, *SICOMP* 45:1762 | journal | no | — | none |
| 23 | gilyen2019qsvt | Gilyén et al. 2019, STOC | conference | no | — | none |
| 24 | giovannetti2008quantum | Giovannetti et al. 2008, *PRL* 100:160501 | journal | no | — | none |
| 25 | gomes2017adaptiverf | Gomes et al. 2017, *Mach. Learn.* 106:1469 | journal | no | — | none |
| 26 | harrow2009hhl | Harrow–Hassidim–Lloyd 2009, *PRL* 103:150502 | journal | no | — | none |
| 27 | havlicek2019qkernels | Havlíček et al. 2019, *Nature* 567:209 | journal | no | — | none |
| 28 | hoi2021online | Hoi et al. 2021, *Neurocomputing* 459:249 | journal | no | — | none |
| 29 | huang2020shadows | Huang–Kueng–Preskill 2020, *Nat. Phys.* 16:1050 | journal | no | — | none |
| 30 | huang2021powerofdata | Huang et al. 2021, *Nat. Commun.* 12:2631 | journal | no | — | none |
| 31 | huang2022experiments | Huang et al. 2022, *Science* 376:1182 | journal | no | — | none |
| 32 | kallaugher2021streaming | Kallaugher 2021, IEEE FOCS, pp. 897–908 | conference | no | — | none |
| 33 | kallaugher2025designquantum | Kallaugher–Parekh–Voronova 2025, SIAM SOSA, pp. 9–45 | conference | **was arXiv-linked** | **yes — same venue/pages** | **URL → DOI 10.1137/1.9781611978315.2** |
| 34 | leland1994ethernet | Leland et al. 1994, *IEEE/ACM ToN* 2:1 | journal | no | — | none |
| 35 | levin2017mixing | Levin–Peres 2017, AMS book | book | no | — | none |
| 36 | lloyd2014quantum | Lloyd et al. 2014, *Nat. Phys.* 10:631 | journal | no | — | none |
| 37 | mandelbrot1968fbm | Mandelbrot–Van Ness 1968, *SIAM Rev.* 10:422 | journal | no | — | none |
| 38 | muthukrishnan2005data | Muthukrishnan 2005, *FnT TCS* 1:117 | monograph series | no | — | none |
| 39 | park2000selfsimilar | Park–Willinger (eds.) 2000, Wiley book | book | no | — | none |
| 40 | polyak1992averaging | Polyak–Juditsky 1992, *SICON* 30:838 | journal | no | — | none |
| 41 | preskill2018nisq | Preskill 2018, *Quantum* 2:79 | journal (peer-reviewed OA) | no | — | none |
| 42 | rebentrost2014quantum | Rebentrost et al. 2014, *PRL* 113:130503 | journal | no | — | none |
| 43 | schuld2018supervised | Schuld–Petruccione 2018, Springer book | book | no | — | none |
| 44 | schuld2021encoding | Schuld et al. 2021, *PRA* 103:032430 | journal | no | — | none |
| 45 | shalev2012online | Shalev-Shwartz 2012, *FnT ML* 4:107 | monograph series | no | — | none |
| 46 | su2024elasticonline | Su et al. 2024, *Inf. Sci.* 676:120783 | journal | no | — | none |
| 47 | tang2019quantum | Tang 2019, STOC | conference | no | — | none |
| 48 | vanvleck1966spectrum | Van Vleck–Middleton 1966, *Proc. IEEE* 54:2 | journal | no | — | none |
| 49 | weinberger2009featurehashing | Weinberger et al. 2009, ICML | conference | no | — | none |
| 50 | willinger1997selfsimilarity | Willinger et al. 1997, *IEEE/ACM ToN* 5:71 | journal | no | — | none |
| 51 | woodruff2014sketching | Woodruff 2014, *FnT TCS* 10:1 | monograph series | no | — | none |
| 52 | wu2024evolvingprototypes | Wu et al. 2024, *Inf. Sci.* 679:120979 | journal | no | — | none |
| 53 | yan2024bidynamic | Yan et al. 2024, *Inf. Sci.* 676:120796 | journal | no | — | none |
| 54 | zhao2026exponential | Zhao et al. 2026, arXiv:2604.07639 | **preprint** | **no** (verified: arXiv-only as of 2026-08-11) | **RETAINED + FLAGGED** — subject of the study; entry labelled "Preprint"; same-claim peer-reviewed anchors co-cited (#31, #32, #33) |

Notes: entries 5 and 12 (JMLR) carry the journal's canonical URLs because JMLR does not assign DOIs; they are peer-reviewed journal articles, not preprints. All other entries carry publisher DOIs.

---

## Metadata verification addendum (2026-08-11, five-agent workflow)

A second, adversarial pass verified every entry at the metadata level: a mechanical sweep resolved **all 51 DOIs** via doi.org content negotiation (CSL JSON; 0 unresolved) and machine-compared resolved title / first author / venue / year against the bibliography, while four independent web agents re-checked the judgment cases against primary sources.

**Result: 42 exact matches, 7 benign differences (documented below), 1 false positive, and 1 genuine defect — now fixed.**

### The defect (found and fixed)

`su2024elasticonline` cited DOI `10.1016/j.ins.2024.120783`, which resolves to an **unrelated** *Information Sciences* article ("Fine-grained complexity-driven latency predictor in hardware-aware neural architecture search using composite loss", first author Lin). Crossref locates the cited paper — "Elastic online deep learning for dynamic streaming data", **Su, Rui; Guo, Husheng; Wang, Wenjian**, *Information Sciences* **676**, issued 2024-08 — at DOI **10.1016/j.ins.2024.120799** (content-negotiation re-verified: title, all three authors, container, and volume match the entry exactly; only the article-number digits had been transposed). Fixed (article number and URL, `120783 → 120799`) in **all five venue variants** (the shared bibliography carried the same error); the frozen `main_as_reviewed.tex` snapshot is untouched so the tracked-changes document shows the correction to the reviewers.

### Confirmations from the independent agents

- **zhao2026exponential — PREPRINT_ONLY, quadruple-confirmed**: arXiv abs page has no journal-reference field (single version v1, only the arXiv DataCite DOI); six exact-title+publisher searches return zero publisher-domain hits; Google Research's official listing states the venue verbatim as "arXiv:2604.07639 (preprint)"; a Crossref API bibliographic query on the exact title returns **no registered publisher DOI**. Retained and flagged stands on registry evidence.
- **kallaugher2025designquantum — CONFIRMED, including author order**: DOI 10.1137/1.9781611978315.2 resolves to the SIAM article page titled "2025 Symposium on Simplicity in Algorithms (SOSA) | How to Design a Quantum Streaming Algorithm…", pages 9–45. The printed byline is **Kallaugher, Parekh, Voronova** (publisher article page contributor markup, the proceedings table of contents, and the authors' own arXiv listing all agree); the conflicting "Parekh first" ordering exists only in Crossref's publisher-deposited metadata and dblp's mirror of that deposit — a deposit artifact, not the publication byline. Our entry is correct as-is.
- **JMLR pair — BOTH CONFIRMED verbatim** against JMLR's official BibTeX files (jmlr.org): Arunachalam–de Wolf 2018, 19(71):1–36; Bifet et al. 2010, 11:1601–1604 (JMLR's retroactive per-paper number 52 omitted, as is standard for the classic pagination form).
- **Publication-status sample — ALL SIX CONFIRMED by registry type fields**: SDM 2007 and ICML 2009 and FOCS 2021 = Crossref `proceedings-article`; ICALP 2019 = DataCite `ConferencePaper` (LIPIcs registers via DataCite — its Crossref 404 is expected, not a red flag); Park–Willinger 2000 = Crossref `edited-book`; Gama 2010 = Crossref `book`.

### The seven benign differences (no changes warranted)

| Key | Difference | Why no change |
|---|---|---|
| beran1994statistics | DOI issued 2017, publisher "Routledge" | The DOI is the Routledge digital reissue of the 1994 Chapman & Hall monograph; citing the original year with the live DOI is standard practice |
| ben2010agnostic | DOI issued 2009 | Online-first date; the print citation (*Mach. Learn.* 79, 2010) is correct |
| kallaugher2021streaming | DOI issued 2022 | IEEE dates the FOCS proceedings volume 2022; the conference year 2021 is the standard citation form |
| chakraborty2019blockencodings | container "LIPIcs, Volume 132" | Same venue, registry naming convention vs proceedings name |
| kallaugher2025designquantum | container field = publisher (SIAM) | Same venue; CSL field-population difference |
| schuld2018supervised | publisher string "Springer International Publishing" | Same publisher family as "Springer, Cham" |
| weinberger2009featurehashing | container includes "Annual" | Phrasing-only |

False positive: `park2000selfsimilar` failed the mechanical first-author check only because an edited-book CSL record has editors, no `author` field; title, venue, and year match exactly.

### Post-fix state

51/51 DOIs resolve to the cited works; 3 non-DOI entries verified at source (2 × JMLR, 1 × arXiv flagged preprint); citation–bibliography bijection re-checked (54 ↔ 54); all five variants recompiled with 0 errors.

---

## Re-verification addendum (2026-08-16, branch `zhao-preprint-fix`)

Prompted by the Academic Editor's First-Decision comment ("I strongly invite the Authors to not include preprints among references"), the status of arXiv:2604.07639 was re-verified five days after the audit by a six-agent workflow (`wf_ff27d364-e1c`; three independent hunters, a citation-load mapper, a candidate assessor, and a completeness critic). Verdict: **PREPRINT_ONLY re-confirmed**. Negative evidence, all checked 2026-08-16: arXiv abs page still v1-only (8 Apr 2026), no journal-ref, no related DOI; arXiv API: no journal_ref/doi elements; Crossref exact-title bibliographic query: no record; Semantic Scholar: venue "arXiv.org" only; INSPIRE-HEP: typed preprint, no publication_info; dblp: CoRR "Informal" entry only; Google Scholar cluster: all versions arXiv-hosted; OpenReview: zero notes; Google Research listing: still "arXiv (preprint)"; accepted-papers lists of FOCS 2026 (~350 papers; Zlokapa appears only with an unrelated SYK paper), STOC 2026 (~565), QIP 2026 (chronologically impossible; checked anyway), TQC 2026 (~70; no LIPIcs volume yet), CCC 2026, SODA 2026: absent from all; Quantum, PRX Quantum, npj QI, Nature/Science: no publisher page; all seven authors' publication pages (incl. Preskill pub.html entry 186 and Haimeng Zhao's own site): arXiv-only; the authors' Quantum Frontiers post: no acceptance language; all 19 citing works: none cites a venue or "to appear". One false lead (a search-engine AI summary claiming PRL 133:230604) was debunked via Crossref: that DOI belongs to Oh et al. 2024.

**Substitute-candidate assessment** (all four DOI-verified; none can replace the citation): Molteni–Gyurik–Dunjko, npj QI 12:19 (2026), 10.1038/s41534-025-01162-2 — learning quantum observables, different team/theorems; Zhang–Yuan, npj QI 10:42 (2024), 10.1038/s41534-024-00835-8 — access-model construction *costs* (opposite direction); Kallaugher FOCS 2021 — already cited, polynomial advantage; Zhao–Deng, npj QI 11:127 (2025), 10.1038/s41534-025-01078-x — same first author, different collaboration, communication-based advantage. Replacing the framework citation with any of these would misattribute the paper's object of study; each is at most a co-citation anchor.

**Actions taken (2026-08-16):** (i) in-text zhao2026exponential citations reduced 8 → 6 by consolidating two repeat attributions (§3.3 second same-paragraph citation → "the originating framework"; §5.3 memory bullet → cross-reference to §3.3); no attribution site removed; (ii) Kallaugher–Parekh–Voronova STOC 2024 added to the bibliography (Crossref content-negotiation verified) and co-cited in Related Work and in the preprint-disclosure footnote; (iii) the Related Work sentence crediting FOCS 2021 with the "first exponential" separation corrected (polynomial → advantage; exponential → STOC 2024). Bibliography now 55 entries; cite↔bib bijection re-verified 55↔55. The same FOCS-2021 wording persists in the four sibling variants (main branch) — sync before any sibling submission.

## Deep replacement hunt (2026-08-16, workflow `wf_ed9282b1-497`, 7 agents)

A second, deeper investigation asked whether the Zhao et al. entry could be *replaced* rather than merely disclosed. Four routes never previously attempted were run, then adversarially verified. All returned clean negatives.

1. **Published under a different title?** Every earlier search was title- or arXiv-ID-based and would have missed a retitled journal version. An author-based sweep of all seven authors across OpenAlex and Crossref (2025-10-01 → 2026-08-16) found no such publication. Six topically adjacent peer-reviewed papers were surfaced and each DOI-resolved; every one fails on authorship or subject matter (none contains the O(n)-qubit sketch construction, the ×R overhead, the refreshing-time/repetition-number model, or the O(N^c) hardness).
2. **A published paper that builds on it, usable as a secondary description?** The citing set was unioned across three indexes (Semantic Scholar 19, INSPIRE-HEP 18 incl. 4 that S2 missed, OpenAlex 0 — an index gap): **23 citing works, of which zero are peer-reviewed published.** Twenty-one are arXiv preprints with no journal-ref and no DOI; the remaining two are a self-labelled workshop preprint (workshop not yet held) and an off-topic manuscript. Expected for a preprint four months old.
3. **A same-team published companion?** None. Checked Huang et al. *Science* 2022, Huang et al. *Nat. Commun.* 2021, Chen–Cotler–Huang–Li FOCS 2021, Gilboa et al. NeurIPS 2024, Lewis–Gilboa–McClean *Nat. Commun.* 2025, Zhao–Deng npj QI 2025 — each verified to contain none of the four inherited results.
4. **Independent published equivalents?** None for results (i)–(iii). Result (iv), the effective-sample-size-under-correlation concept, already has its published home in the manuscript: Geyer (1992), cited as `geyer1992practical`. Note the framework's refreshing time "carries no independent penalty" (§3.3); the τ-division is this paper's own more conservative construction, so (iv) is not in fact inherited.

**Decisive linkage evidence (orthogonal to all search-based checks).** DataCite `relatedIdentifiers` and `relatedItems` on 10.48550/arXiv.2604.07639 are **both empty arrays** — this is the exact field a publisher populates (IsPreviousVersionOf / IsIdenticalTo) the moment a journal version is deposited. DBLP returns a single hit typed "Informal and Other Publications", venue CoRR, closing the CS-proceedings route that journal-centric indexes cannot rule out. OpenAlex types the work `preprint` with one location, `is_published: false`, `is_accepted: false`. The authors' own recommended BibTeX in their code repository (github.com/haimengzhao/quantum-oracle-sketching, pushed 2026-05-21) carries an **empty `journal={}` field**.

**Instrument validation.** A negative from an index counts only if the index is live: INSPIRE-HEP demonstrably back-fills July-2026 APS publication data (a 2026 PRX article carries full `publication_info` and its APS DOI), while the target record 3142627 has neither `dois` nor `publication_info`. The same control exposed a trap worth recording: that PRX article is published yet its arXiv record still shows no journal-ref, so *arXiv journal-ref absence alone is weak evidence* — the conclusion here rests on DataCite, DBLP and INSPIRE, not on arXiv.

**Decision: retain and disclose; all six citation sites KEEP_AS_IS.** Adding further co-citations was considered and rejected: it does not answer the Editor (an added reference does not remove the preprint), the disclosure footnote already carries four peer-reviewed anchors, a 56th entry would falsify the "55 entries / 55↔55" statement in the response letters and the submitted `submission_R1/` copies, and placing same-author work beside the framework sentence would imply support that verifiably does not exist. The single change applied is label-only, in the bibliography entry: "Preprint." → "Preprint; not peer-reviewed as of August 2026." No reference added, no attribution altered, no result or number touched, entry count unchanged at 55.

**Integrity note.** One agent reported the PRX Perspective's DOI as 10.1103/PhysRevX.16.030501; the verifier found that DOI **does not resolve** (APS has moved to opaque DOIs; the real one is 10.1103/tn89-g1xz). It was not propagated anywhere. Recorded as a reminder that constructed-by-convention DOIs must always be resolved before use.

**Timeline.** The preprint is four months old and typical quant-ph submission-to-publication latency exceeds that comfortably, so a zero result is the predicted outcome rather than a search failure. Re-running is unlikely to change the answer for several months; the proof-stage re-check remains the right control.

## Renamed-publication hunt (2026-08-16, workflow `wf_7e8571c0-5f7`, 6 agents)

Final question tested: could the work have been published under a **different title with the framework itself renamed**, defeating every metadata search? Verdict: **REFUTED — no published version exists under any name.** The negative is now structural, not merely accumulated.

**Why a rename cannot hide.** Five of the systems checked are keyed on the arXiv identifier or the arXiv DOI, never on the title string, and each merges or back-links a published version onto the preprint record irrespective of retitling: (1) Semantic Scholar's merged cluster fetched by arXiv ID returns `externalIds` = {ArXiv, DBLP CoRR, 10.48550/arXiv.2604.07639} and `publicationVenue` arXiv.org; (2) OpenAlex on the same DOI reports `type: preprint`, `is_published: false`, `locations_count: 1`, `versions: null`; (3) DataCite `relatedIdentifiers` empty; (4) the arXiv record carries no journal-ref, no related DOI, and no v2 — `published == updated` to the second, meaning the eprint has been untouched for four months, which is anomalous for a paper that had cleared review (camera-ready normally produces a v2); (5) INSPIRE-HEP, human-curated, holds a record curators have demonstrably edited (affiliations normalised) yet which lacks the `publication_info`, `dois` and `refereed` keys entirely.

**Citation forensics (independent third-party evidence).** The bibliographies of **21 citing works** were retrieved and read: **21/21 cite it as an arXiv preprint; zero cite a journal; zero say "to appear" or "in press".** If a published version existed under any title, citing authors would be the first to record it.

**Framework name confirmed.** The authors' own repository README states: "Quantum Oracle Sketching (QOS) … Code repository for our paper ['Exponential quantum advantage in processing massive classical data'](arxiv.org/abs/2604.07639) … we introduce **Quantum Oracle Sketching**, a framework that enables access to the classical world in quantum superposition for large-scale machine learning." The manuscript's terminology matches the authors' official name exactly. README last edited 2026-04-10, repo still active 2026-05-21, citation still arXiv-only; `CITATION.cff` returns 404.

**No alias in the published literature.** The fingerprint vocabulary — "quantum oracle sketching", and "refreshing time" / "repetition number" in this technical sense — appears in no published work across the major publisher sites. The framework is not known under another name, by these authors or anyone else. Split publication is likewise ruled out: no component result appears as a standalone published paper.

**Third-party activity worth knowing (all non-peer-reviewed).** Four Zenodo self-deposits by unaffiliated authors engage with the framework, including "Exact Counterexamples and Interface Gaps in Quantum Oracle Sketching" and "Quantum Oracle Sketching from Correlated Classical Samples" (which states it "does not claim to retroactively establish the arguments in arXiv:2604.07639v1"). None is peer-reviewed, none is citable here, and none affects this manuscript — whose design already treats the framework's theorems as premises to be stress-tested rather than as established facts. Noted because independent scrutiny of the framework is now circulating.

**Excluded from evidence:** a Zenodo record surfaced by keyword search is this project's own software deposit (Helmy & Mahmoud, 10.5281/zenodo.19831893, resource type Software). It is not independent corroboration and was discarded.

**Honest gap:** the ORCID/author-identity sweep agent forwarded its task and returned no findings, so ORCID records were not directly enumerated. This does not affect the conclusion — the five arXiv-ID-keyed systems above would surface a retitled publication regardless of author-record completeness, and the citation forensics provide independent confirmation.

**Consequence: no manuscript change.** The retain-and-disclose configuration stands, unchanged.
