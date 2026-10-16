# Build the technical report: figures (checked against RESULTS.md) -> PDF.
# Usage: make -C paper        (needs MacTeX at /Library/TeX/texbin and the repo .venv)
TEXBIN ?= /Library/TeX/texbin
PY     ?= .venv/bin/python  # relative to the repository root
FIGS   := figures/fig1_staged_drift.pdf figures/fig2_temperature.pdf figures/fig3_benchmarks.pdf figures/fig4_matched.pdf

.PHONY: all figures clean check
all: main.pdf

figures: $(FIGS)
# One script writes all four figures; the first stands for the set (GNU make 3.81 has no grouped targets).
$(firstword $(FIGS)): figures/make_figures.py
	cd .. && $(PY) -m paper.figures.make_figures
$(wordlist 2,4,$(FIGS)): $(firstword $(FIGS))

main.pdf: main.tex refs.bib $(FIGS)
	PATH=$(TEXBIN):$$PATH latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check: main.pdf
	@! grep -E "undefined|Rerun to get" main.log || { echo "undefined references or citations"; exit 1; }
	@echo "main.log: no undefined references or citations"

clean:
	PATH=$(TEXBIN):$$PATH latexmk -C main.tex
	rm -f main.bbl
