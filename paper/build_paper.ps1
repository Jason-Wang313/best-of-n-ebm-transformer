$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
pdflatex best_of_n_ebm_transformer.tex
bibtex best_of_n_ebm_transformer
pdflatex best_of_n_ebm_transformer.tex
pdflatex best_of_n_ebm_transformer.tex

$target = Join-Path $env:USERPROFILE "Downloads\best-of-n-ebm-transformer.pdf"
Copy-Item -LiteralPath "best_of_n_ebm_transformer.pdf" -Destination $target -Force
Write-Host "Copied final PDF to $target"

