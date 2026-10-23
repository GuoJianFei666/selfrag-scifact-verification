param([switch]$WithModel, [int]$Port = 8787)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw '仓库内隔离环境不存在。请先按 README 创建环境。'
}
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $projectRoot
try {
    $serverArguments = @('-m', 'evidence_rag', 'serve', '--data', 'var/scifact',
        '--index', 'var/contriever-index', '--cache', 'var/model-cache',
        '--reports', 'var/reports', '--port', $Port)
    if ($WithModel) { $serverArguments += @('--config', 'configs/model.yaml') }
    & $pythonPath @serverArguments
    exit $LASTEXITCODE
} finally { Pop-Location }
