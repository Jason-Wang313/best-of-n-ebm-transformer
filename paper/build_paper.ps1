$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
pdflatex energy_tail_audit.tex
bibtex energy_tail_audit
pdflatex energy_tail_audit.tex
pdflatex energy_tail_audit.tex

$final = Join-Path $PSScriptRoot "final"
if (-not (Test-Path $final)) {
    New-Item -ItemType Directory -Path $final | Out-Null
}
$repoTarget = Join-Path $final "best-of-n-ebm-transformer-v3.pdf"
Copy-Item -LiteralPath "energy_tail_audit.pdf" -Destination $repoTarget -Force

$desktop = Join-Path $env:USERPROFILE "OneDrive\Desktop"
if (-not (Test-Path $desktop)) {
    $desktop = [Environment]::GetFolderPath("Desktop")
}
$target = Join-Path $desktop "best-of-n-ebm-transformer-v3.pdf"
Copy-Item -LiteralPath "energy_tail_audit.pdf" -Destination $target -Force
Write-Host "Copied repo PDF to $repoTarget"
Write-Host "Copied final PDF to $target"
