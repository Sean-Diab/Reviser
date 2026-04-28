$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Remove-Item main.aux, main.bbl, main.blg, main.fdb_latexmk, main.fls, main.log, main.out, main.run.xml, main.synctex.gz -ErrorAction SilentlyContinue
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
Write-Host "Built paper: $root/main.pdf"
