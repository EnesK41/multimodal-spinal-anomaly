# Spinal Anomaly Detection Report

This folder is a working LaTeX report draft based on the GTU graduation project template.

Patient-derived figures are intentionally kept local and ignored by Git. If the report is compiled from a fresh clone, add the private figure assets back under `Imgs/` or remove the corresponding `\includegraphics` calls.

Main file:

`main.tex`

Compile order when LaTeX tools are installed:

1. `pdflatex main.tex`
2. `biber main`
3. `pdflatex main.tex`
4. `pdflatex main.tex`
