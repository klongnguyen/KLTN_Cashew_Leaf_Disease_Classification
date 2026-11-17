@echo off
rem Launch the web app (DenseNet121 x5 + YOLO26s, EXP-035) at http://127.0.0.1:8080/
rem Uses the project venv .venv-web when present, otherwise falls back to system python.
rem Model binaries (*.keras, best.pt) are not in Git - see Application\README.md.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-web\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%ROOT%Application"
"%PY%" -m uvicorn server:app --host 127.0.0.1 --port 8080
