# Report — roadmap

This folder contains the long-form write-up of the project. Files are ordered
so they can be read end-to-end or pulled into LaTeX section-by-section. Math
uses `$...$` (inline) and `$$...$$` (display) so it renders in any markdown
viewer that supports KaTeX/MathJax (GitHub web, Cursor preview, pandoc).

| File | What's in it |
|---|---|
| `01_introduction.md` | Motivation, research question, what we found in two sentences. |
| `02_methods.md` | All the math: bar construction, rolling cointegration, MS-AR(1), the four strategies, PnL accounting, and metric definitions. The longest file — this is the one to copy into the Methods chapter. |
| `03_implementation.md` | The code side: directory layout, class responsibilities, end-to-end pipeline, the `MONTH` orchestrator, and the notebook skeleton. |
| `04_results.md` | The 9 month-pair Sharpe / PnL / trade-count tables and the sensitivity sweep summary. |
| `05_discussion.md` | What works, what doesn't, when, why. The GBPEUR-Aug-24 case study, the NOKSEK negative control, the general MS-AR-vs-AR finding. |
| `06_conclusion.md` | One-paragraph summary for the paper, learnings (specific + general), limitations, future work. |

Quick navigation:

- **Strategy definitions** → `02_methods.md` §2.5
- **The headline result table** → `04_results.md` §4.1
- **The GBPEUR-Aug-24 finding** → `05_discussion.md` §5.2
- **The one-paragraph paper version** → `06_conclusion.md` §6.5
