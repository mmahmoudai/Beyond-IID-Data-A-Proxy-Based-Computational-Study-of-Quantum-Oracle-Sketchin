Sensors (MDPI) revision R1 - upload set (2026-08-11)
=====================================================

manuscript_revised.pdf   Clean revised manuscript (29 pp).
manuscript_revised.tex   Its LaTeX source.
tracked_changes.pdf      Highlighted version (built with latexdiff against
                         the 16 July submission): new and modified text in
                         blue; removed text omitted entirely (no strikeout),
                         so the document reads as the final manuscript with
                         changes highlighted; 29 pp. NOTE: the MDPI class
                         keeps the title and abstract in the preamble, which
                         latexdiff cannot mark - both are fully rewritten
                         (stated in the response letter).
tracked_changes.tex      Its LaTeX source.
response_to_reviewers.pdf  Point-by-point response (Reviewer 1: 15 points;
                           Reviewer 2: 10 points; verbatim quotes;
                           manuscript ID sensors-4470240). 9 pp.
response_to_reviewers.docx Same letter as an editable Word document.
response_to_reviewers.tex  Its standalone LaTeX source (compiles alone
                           with xelatex; Times New Roman).
source_package.zip       Complete self-contained source: main.tex,
                         Definitions/ (official MDPI class), all figure
                         PDFs, cover letter, revision materials
                         (re-analysis + sensitivity scripts and outputs).
                         main.tex compiles standalone inside the zip
                         (pdflatex x2).

To compile manuscript_revised.tex or tracked_changes.tex, place the file
in the root of the unpacked source_package.zip (they expect Definitions/
and the figure PDFs beside them); response_to_reviewers.tex compiles
anywhere with xelatex.
