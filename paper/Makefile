LATEX=pdflatex
BIBTEX=bibtex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex tables/*.tex
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf
