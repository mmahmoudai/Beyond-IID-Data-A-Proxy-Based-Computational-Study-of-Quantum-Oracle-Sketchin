Sensors (MDPI) revision R1 - upload set
Manuscript ID: sensors-4470240
=======================================

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
RESPONSE LETTERS - three files, same content, two packagings. Upload the
per-reviewer pair if the system asks for one response per reviewer; upload
the combined letter if it asks for a single response document. Each exists
as .pdf, .docx (editable Word) and .tex (standalone, xelatex, Times New
Roman). All carry manuscript ID sensors-4470240 and no date.

response_to_reviewer_1.*   Self-contained response to Reviewer 1
                           (15 points, verbatim quotes). 6 pp.
response_to_reviewer_2.*   Self-contained response to Reviewer 2
                           (10 points, verbatim quotes). 5 pp.
response_to_reviewers.*    Combined letter, both reviewers in one document
                           (15 + 10 points) - convenient for the academic
                           editor. 9 pp.

Each per-reviewer letter repeats the shared header, the summary of principal
changes, and the verification note, so it stands alone; every point-by-point
response is identical to the combined letter word for word. The one wording
difference: in R1.10 the combined letter says "the look-ahead point raised by
Reviewer 2", which the Reviewer-1 letter states directly as "the full-series
threshold look-ahead raised in review", since that reviewer does not see the
other report.
source_package.zip       Complete self-contained source: main.tex,
                         Definitions/ (official MDPI class), all figure
                         PDFs, cover letter, revision materials
                         (re-analysis + sensitivity scripts and outputs).
                         main.tex compiles standalone inside the zip
                         (pdflatex x2).

To compile manuscript_revised.tex or tracked_changes.tex, place the file
in the root of the unpacked source_package.zip (they expect Definitions/
and the figure PDFs beside them); the response-letter .tex files compile
anywhere with xelatex.
