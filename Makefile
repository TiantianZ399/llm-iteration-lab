PYTHON ?= python
OUT_DIR ?= .

.PHONY: benchmark tex

benchmark:
	$(PYTHON) scripts/run_benchmark_comparison.py --out_dir $(OUT_DIR)

tex:
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
