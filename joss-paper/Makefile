default: pdf

MDFILE=paper.md
BIBFILE=paper.bib
LATEXFILE=paper.tex

# some variables for the paper
REPOSITORY="https://github.com/pyfstat/pyfstat"
ARCHIVE="https://doi.org/${FORMATTEDDOI}"
REVIEW="TBD"
ARCHIVEDOI=${ARCHIVE}
FORMATTEDDOI="TBD"
YEAR="2021"
TITLE="PyFstat: a Python package for continuous gravitational-wave data analysis"
AUTHOR="D. Keitel, R. Tenorio, G. Ashton, R. Prix"
ISSUE="TBD"
VOLUME="TBD"
PAGE="TBD"
JOURNALNAME="Journal of Open Source Software"
PUBLISHED="TBD"
SUBMITTED="26 January 2021"
PAPERURL="TBD"

# get JOSS template and logo
latex.template logo.png:
	curl "https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss/$@" > $@

PREPRINT=true
ifdef PREPRINT
  LOGO=logo-empty.png
  SUBMITBIT="Submitted to The "
else
  LOGO=logo.png
  SUBMITBIT=""
endif


# use pandoc and pandoc-citeproc to create LaTeX source
latex: latex.template $(LOGO) $(MDFILE) $(BIBFILE)
	pandoc --filter pandoc-citeproc --bibliography=$(BIBFILE) -V repository=$(REPOSITORY) -V journal_name=$(SUBMITBIT)$(JOURNALNAME) -V paper_url=$(PAPERURL) -V archive_doi=$(ARCHIVEDOI) -V review_issue_url=$(REVIEW) -V formatted_doi=$(FORMATTEDDOI) -V logo_path=$(LOGO) -V year=$(YEAR) -V paper_title=$(TITLE) -V citation_author=$(AUTHOR) -V issue=$(ISSUE) -V volume=$(VOLUME) -V page=$(PAGE) -V submitted=$(SUBMITTED) -V published=$(PUBLISHED) -V graphics=true -V papersize=a4paper -V geometry:margin=1in -t latex -s $(MDFILE) -o $(LATEXFILE) --template latex.template

pdf: latex
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm paper.bcf paper.aux paper.log paper.out paper.run.xml

clean_downloads:
	rm latex.template logo.png
