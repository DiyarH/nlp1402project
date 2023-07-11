#!/usr/bin/bash
python src/dataset
xelatex -syntex=1 -interaction=nonstopmode -file-line-error -pdf --shell-escape --output-directory=. latex/Phase1-Report.tex