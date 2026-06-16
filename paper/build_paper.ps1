$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode -halt-on-error energy_tail_audit.tex
bibtex energy_tail_audit
pdflatex -interaction=nonstopmode -halt-on-error energy_tail_audit.tex
pdflatex -interaction=nonstopmode -halt-on-error energy_tail_audit.tex

$final = Join-Path $PSScriptRoot "final"
if (-not (Test-Path $final)) {
    New-Item -ItemType Directory -Path $final | Out-Null
}
$repoTarget = Join-Path $final "best-of-n-ebm-transformer-v4.pdf"
Copy-Item -LiteralPath "energy_tail_audit.pdf" -Destination $repoTarget -Force

$desktop = Join-Path $env:USERPROFILE "OneDrive\Desktop"
if (-not (Test-Path $desktop)) {
    $desktop = [Environment]::GetFolderPath("Desktop")
}
$target = Join-Path $desktop "best-of-n-ebm-transformer-v4.pdf"
Copy-Item -LiteralPath "energy_tail_audit.pdf" -Destination $target -Force
Remove-Item -LiteralPath "energy_tail_audit.pdf" -Force
Write-Host "Copied repo PDF to $repoTarget"
Write-Host "Copied final PDF to $target"
