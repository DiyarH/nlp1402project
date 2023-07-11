#!/usr/bin/bash
python src/word2vec/run.py
python src/language_model/run.py
xelatex -syntex=1 -interaction=nonstopmode -file-line-error -pdf --shell-escape --output-directory=. latex/Phase2-Report.tex