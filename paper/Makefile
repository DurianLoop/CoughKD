LATEX=pdflatex
BIBTEX=bibtex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex ../runs/kd_failure_analysis/paper_ready_tables/paper_ready_tables.tex
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf
