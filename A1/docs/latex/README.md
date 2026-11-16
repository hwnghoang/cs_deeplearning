# A1.2 LaTeX report

| File | Content |
|---|---|
| `A1.2_report.tex` | A1.2 — final report skeleton; fill every `\todo{...}` and `--` |
| `bkreport.sty` | Shared style: fonts, cover page, group info (edit group info here once) |
| `refs.bib` | Shared bibliography |
| `latexmkrc` | Forces XeLaTeX (required for Vietnamese) |
| `HCMUT_official_logo.png` | University logo on the cover page |
| `figures/` | Result figures; missing figures show a grey box |

## Compile on Overleaf
1. Zip this folder and upload it as a new project (New Project → Upload Project).
2. Set the main document (Menu → Main document) to `A1.2_report.tex`.
3. Compiler is XeLaTeX (set by `latexmkrc`; if not, choose it under Menu → Compiler).

## Compile locally
```
latexmk A1.2_report.tex
```
