# Reports

Three sections, each a self-contained LaTeX document built with **Tectonic**
(installed via Homebrew; auto-downloads packages and resolves citations — no biber needed):

```
report/
├── proposal/      proposal.tex      → the research proposal
├── dissertation/  report.tex        → the main MSc dissertation
│                  references.bib       (data source; report.tex embeds its bibliography)
└── progress/
    └── week1/     week1.tex         → weekly progress report (one folder per week)
```

## Build (terminal)
Easiest — rebuild **everything** and make each PDF GitHub-viewable in one go:

```bash
report/build.sh                 # needs: brew install tectonic ghostscript
```

Or build a single document with Tectonic inside its folder:

```bash
cd report/proposal       && tectonic --synctex --keep-logs --keep-intermediates proposal.tex
cd report/dissertation   && tectonic --synctex --keep-logs --keep-intermediates report.tex
cd report/progress/week1 && tectonic --synctex --keep-logs --keep-intermediates week1.tex
```

Each produces a `.pdf` next to its source.

## Make a PDF viewable inline on GitHub
GitHub's inline viewer cannot render Tectonic's PDFs ("Error loading PDF page number 1"),
though they are valid and download fine. `build.sh` handles this automatically by
re-rendering each PDF with Ghostscript (PDF 1.4). To do it manually after a one-off build:

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
   -dNOPAUSE -dBATCH -dQUIET -sOutputFile=out.pdf in.pdf && mv out.pdf in.pdf
```

## Build (VS Code, live preview)
With the LaTeX Workshop extension, open any `.tex` and press **Cmd+S** — it rebuilds on save
and refreshes the PDF. Open the preview with **Cmd+Alt+V**. Cmd+click in the PDF jumps to the
source (SyncTeX). No per-folder configuration is needed.

## Adding a new week
`mkdir report/progress/week2` and add `week2.tex` (copy `week1/week1.tex` as a starting point).

## Notes
- First build of a new package needs internet (Tectonic fetches it once, then caches).
- Build artifacts (`*.aux`, `*.log`, `*.synctex.gz`, …) are gitignored; `*.pdf` is not.
- Clean a folder with: `rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk`
- The dissertation title page renders the Glasgow crest if you drop
  `report/dissertation/glasgow_logo.png` in (optional; it compiles without it).
