default: pdf

MDFILE=paper.md
BIBFILE=paper.bib
LATEXFILE=paper.tex

# some variables for the paper
REPOSITORY="https://github.com/pyfstat/pyfstat"
REVIEW="https://github.com/openjournals/joss-reviews/issues/3000"
ARCHIVEDOI="https://doi.org/10.5281/zenodo.4660591"
FORMATTEDDOI="10.21105/joss.03000"
YEAR="2021"
TITLE="PyFstat: a Python package for continuous gravitational-wave data analysis"
AUTHOR="D. Keitel, R. Tenorio, G. Ashton, R. Prix"
ISSUE="60"
VOLUME="6"
PAGE="3000"
JOURNALNAME="Journal of Open Source Software"
PUBLISHED="06 April 2021"
SUBMITTED="26 January 2021"
PAPERURL="https://doi.org/10.21105/joss.03000"
EDITOR="Daniel S. Katz"
EDITORURL="https://danielskatz.org/"
REVIEWER1="khanx169"
REVIEWER2="RobertRosca"

# get JOSS template and logo
latex.template logo.png:
	curl "https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss/$@" > $@

# PREPRINT=true
ifdef PREPRINT
  LOGO=logo-empty.png
  SUBMITBIT="Submitted to The "
else
  LOGO=logo.png
  SUBMITBIT=""
endif


# use pandoc and pandoc-citeproc to create LaTeX source
latex: latex.template $(LOGO) $(MDFILE) $(BIBFILE)
	pandoc --filter pandoc-citeproc --bibliography=$(BIBFILE) -V repository=$(REPOSITORY) -V journal_name=$(SUBMITBIT)$(JOURNALNAME) -V paper_url=$(PAPERURL) -V archive_doi=$(ARCHIVEDOI) -V review_issue_url=$(REVIEW) -V formatted_doi=$(FORMATTEDDOI) -V logo_path=$(LOGO) -V year=$(YEAR) -V paper_title=$(TITLE) -V citation_author=$(AUTHOR) -V issue=$(ISSUE) -V volume=$(VOLUME) -V page=$(PAGE) -V submitted=$(SUBMITTED) -V published=$(PUBLISHED) -V editor_name=$(EDITOR) -V editor_url=$(EDITORURL) -V reviewers=$(REVIEWER1) -V reviewers=$(REVIEWER2) -V graphics=true -V papersize=a4paper -V geometry:margin=1in -t latex -s $(MDFILE) -o $(LATEXFILE) --template latex.template

pdf: latex
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm paper.bcf paper.aux paper.log paper.out paper.run.xml

clean_downloads:
	rm latex.template logo.png
