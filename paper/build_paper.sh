#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.run.xml main.synctex.gz
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
printf 'Built paper: %s\n' "$ROOT/main.pdf"
