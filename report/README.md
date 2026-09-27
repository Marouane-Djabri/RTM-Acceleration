# Internship report

LaTeX sources of the internship report on Reverse Time Migration on GPU.

| File | What |
|---|---|
| `paper.tex`, `paper.pdf` | full report |
| `paper_short.tex`, `paper_short.pdf` | short version (15 pages): same structure and company presentation, condensed text, key figures only |
| `figures/` | figures used by both reports (generated from `results/`) |
| `paper.sty`, `paper.bst` | page and bibliography style, from Pascal Michaillat's [latex-paper](https://github.com/pmichaillat/latex-paper) template (`TEMPLATE_LICENSE.md`) |
| `paper.bib` | references |

Build either report from this folder:

```bash
latexmk -pdf paper.tex
latexmk -pdf paper_short.tex
```
