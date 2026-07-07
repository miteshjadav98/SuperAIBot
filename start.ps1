# Starts the MegaProject backend (LangGraph, all 3 agents) and the frontend
# (Next.js chat UI) in two new PowerShell windows.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\start.ps1

$root = $PSScriptRoot

# Self-contained: uses MegaProject's own virtual environment.
$venvLangGraph = Join-Path $root '.venv\Scripts\langgraph.exe'
if (-not (Test-Path $venvLangGraph)) {
    Write-Warning "Local venv not found at $venvLangGraph"
    Write-Warning "Create it first:  python -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
    return
}
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

Write-Host 'Starting backend (LangGraph dev) on http://localhost:2024 ...' -ForegroundColor Cyan
# --allow-blocking: the PDF Chatbot does (legitimate) blocking PDF parse/embed work
# in a before_agent hook; without this flag langgraph dev's blocking-call detector
# fails the ingestion.
Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location '$root\backend'; & '$venvLangGraph' dev --no-browser --allow-blocking"

Write-Host 'Starting backend (FastAPI for RAG) on http://localhost:8000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location '$root\backend'; & '$venvPython' main.py"

Write-Host 'Starting frontend (Next.js) on http://localhost:3000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location '$root\frontend'; npm run dev"

Write-Host ''
Write-Host 'Three windows are starting up. Once all are ready, open:' -ForegroundColor Green
Write-Host '  http://localhost:3000' -ForegroundColor Green
