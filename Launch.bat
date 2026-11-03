@echo off
rem Double-click to start the hand tracker and the Processing sketch together.
rem Any options are passed on, e.g.  Launch.bat --camera 1   or   Launch.bat --check
setlocal
cd /d "%~dp0"
title Visuocontrol launcher

set "PYEXE="
set "PYARGS="

rem 1) Python launcher (python.org installs)
where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && (set "PYEXE=py" & set "PYARGS=-3")

rem 2) python on PATH (skips the Microsoft Store placeholder, which fails this test)
if not defined PYEXE python -c "import sys" >nul 2>nul && set "PYEXE=python"

rem 3) Anaconda / Miniconda in their usual places
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%LOCALAPPDATA%\anaconda3\python.exe"
  "%LOCALAPPDATA%\miniconda3\python.exe"
  "%ProgramData%\anaconda3\python.exe"
  "%ProgramData%\miniconda3\python.exe"
) do if not defined PYEXE if exist %%P set "PYEXE=%%~P"

if not defined PYEXE (
  echo [launcher] ERROR: Python was not found.
  echo [launcher] Install Python 3.9-3.13 from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)

"%PYEXE%" %PYARGS% launcher.py %*
if errorlevel 1 (
  echo.
  pause
)
