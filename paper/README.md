# Paper (Deliverable B)

`main.tex` and `references.bib` are the LaTeX source of the final Phase 2 paper
(IEEEtran conference format); `Phase2_G3-07_23i-0661_Paper.pdf` is the compiled paper.

Build on Overleaf (upload `main.tex` and `references.bib`, compiler pdfLaTeX) or locally with
`latexmk -pdf main.tex` or `tectonic main.tex`.

`tables/` and `figures/` are regenerated from `results/` by `experiments/make_tables.py` and
`experiments/make_figures.py`; every number reported in the paper can be checked against them.
