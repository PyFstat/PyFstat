SHELL = /bin/bash

main = paper_cw_mcmc.pdf

.PHONY : main clean figclean arxiv jones

main : $(main)

git_tag.tex :
	./git-tag.sh $@

$(main): *.tex git_tag.tex
	pdflatex ${@:.pdf=} && bibtex ${@:.pdf=} && pdflatex ${@:.pdf=} && pdflatex ${@:.pdf=}

clean :
	rm -f git_tag.tex $(main:.pdf=){.aux,.bbl,.blg,.log,.out,.pdf,Notes.bib} $(texfigs) $(texonlyfigs)

arxiv :
	rm -f $(main)
	rm -f git_tag.tex
	tar -cvf arxiv_submission.tar *tex *bib bibliography.bib *pdf *.bbl
