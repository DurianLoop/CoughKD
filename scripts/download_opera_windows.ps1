param(
  [string]$RepoUrl = "https://github.com/evelyn0414/OPERA.git",
  [string]$OutDir = "external\teacher_repos\OPERA"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "[clone] $RepoUrl -> $OutDir"
Write-Host "[note] This clones OPERA only. The first --run-gate extraction may trigger upstream checkpoint downloads."

if (Test-Path $OutDir) {
  Write-Host "[skip] $OutDir already exists"
} else {
  D:\CoughKD\tools\mingit\cmd\git.exe clone --depth 1 $RepoUrl $OutDir
}

Write-Host "[deps] OPERA likely requires timm, omegaconf, and hydra-core in addition to torch/torchaudio/librosa."
Write-Host "[next preflight] D:\conda\envs\CoughKD\python.exe -B scripts\audit_opera_embedding_upper_bound.py --out runs\opera_embedding_upper_bound_seed7"
Write-Host "[next gate after approval] D:\conda\envs\CoughKD\python.exe -B scripts\audit_opera_embedding_upper_bound.py --out runs\opera_embedding_upper_bound_seed7 --run-gate"
