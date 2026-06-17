param(
  [string]$RepoId = "google/hear-pytorch",
  [string]$OutDir = "pretrained\teachers\hear_pytorch"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[download] $RepoId -> $OutDir"
Write-Host "[note] This model is gated. Accept the Hugging Face terms for $RepoId before running."

$code = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="""$RepoId""",
    local_dir=r"""$OutDir""",
    local_dir_use_symlinks=False,
)
"@

D:\conda\envs\CoughKD\python.exe -B -c $code
if ($LASTEXITCODE -ne 0) {
  throw "HeAR PyTorch download failed with exit code $LASTEXITCODE"
}

Write-Host "[done] HeAR PyTorch files under $OutDir"
Write-Host "[next] D:\conda\envs\CoughKD\python.exe -B scripts\audit_hear_pytorch_embedding_upper_bound.py --out runs\hear_pytorch_embedding_upper_bound_seed7 --device auto --batch-size 8"
