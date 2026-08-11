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
- **Same-claim peer-reviewed support (already co-cited in the manuscript, satisfying the "closely related peer-reviewed publication" requirement):** Kallaugher, FOCS 2021 (DOI 10.1109/FOCS52979.2021.00091 — first exponential quantum–classical space separation for a natural streaming problem); Huang et al., *Science* 376:1182–1186, 2022 (DOI 10.1126/science.abn7293 — experimentally demonstrated exponential advantage in learning); Kallaugher–Parekh–Voronova, SOSA 2025 (DOI 10.1137/1.9781611978315.2 — black-box quantum-sketch construction). Wherever the manuscript invokes the framework's theorems, at least one of these peer-reviewed anchors appears alongside.
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

Notes: entries 5 and 12 (JMLR) carry the journal's canonical URLs because JMLR does not assign DOIs; they are peer-reviewed journal articles, not preprints. All other entries carry publisher DOIs (verified in the earlier full-bibliography DOI audit of this project).
