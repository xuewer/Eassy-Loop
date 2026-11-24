param([switch]$Quiet)

$essayLoopRoot = Split-Path -Parent $PSScriptRoot
$essayLoopPython = Join-Path $essayLoopRoot '.venv/Scripts/python.exe'
$essayLoopDotenv = Join-Path $essayLoopRoot '.env'
if (-not (Test-Path -LiteralPath $essayLoopPython) -or -not (Test-Path -LiteralPath $essayLoopDotenv)) {
    throw 'Eassy Loop interpreter or .env is missing.'
}
$essayLoopReadCode = 'import json,sys; from dotenv import dotenv_values; v=dotenv_values(sys.argv[1]).get("DASHSCOPE_API_KEY"); print(json.dumps({"key": v.strip() if v else None}))'
$essayLoopPayload = $essayLoopReadCode | & $essayLoopPython -B - $essayLoopDotenv
if ($LASTEXITCODE -ne 0) { throw 'Could not load the project .env.' }
$essayLoopParsed = $essayLoopPayload | ConvertFrom-Json
if (-not $essayLoopParsed.key) { throw 'DASHSCOPE_API_KEY is missing in the project .env.' }
$env:DASHSCOPE_API_KEY = $essayLoopParsed.key
Remove-Variable essayLoopPayload,essayLoopParsed -ErrorAction SilentlyContinue
if (-not $Quiet) {
    Write-Host 'Eassy Loop terminal now uses DASHSCOPE_API_KEY from project .env (value hidden).'
}
