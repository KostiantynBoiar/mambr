#!/usr/bin/env bash
# Rebuild every LaTeX document under report/ and make each PDF viewable on GitHub.
#
# For each .tex it builds the PDF with Tectonic, then re-renders that PDF with Ghostscript
# (PDF 1.4, classic xref) so GitHub's inline viewer can display it. Assumes each .tex is a
# standalone document (one PDF per .tex) — which matches this repo's layout.
#
# Usage:
#   report/build.sh            # rebuild everything under report/
#   report/build.sh <path>     # rebuild only the .tex files under <path>
set -euo pipefail

# report/ root = the directory this script lives in (so it works from anywhere).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH="${1:-$ROOT}"

for tool in tectonic gs; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found (brew install $tool)"; exit 1; }
done

count=0
while IFS= read -r -d '' tex; do
  dir="$(dirname "$tex")"
  base="$(basename "$tex" .tex)"
  echo "==> building $tex"
  ( cd "$dir" && tectonic --synctex --keep-logs --keep-intermediates "$base.tex" )
  pdf="$dir/$base.pdf"
  # Re-render with Ghostscript (PDF 1.4, classic xref) so GitHub's viewer can display it.
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
     -dNOPAUSE -dBATCH -dQUIET -sOutputFile="$pdf.tmp" "$pdf" && mv "$pdf.tmp" "$pdf"
  echo "    github-ready: $pdf"
  count=$((count + 1))
done < <(find "$SEARCH" -name '*.tex' -print0 | sort -z)

echo "Done — rebuilt $count document(s)."
