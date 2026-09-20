@echo off
REM ---------------------------------------------------------------------------
REM Windows wrapper for build.sh -- lets you run it from cmd / PowerShell /
REM Explorer without caring that .sh is not an executable extension on Windows.
REM
REM   build.cmd run
REM   build.cmd run leetgpu\01_vector_add
REM   build.cmd clean
REM
REM Why this exists: Windows has no .SH in PATHEXT, so "running" build.sh goes
REM through the shell file association, which opens it in your editor instead
REM of executing it. A .sh file needs a bash interpreter; this finds one.
REM
REM Messages here are ASCII on purpose -- a .cmd file with non-ASCII bytes gets
REM mangled by the console code page, same reason the .cu files print ASCII.
REM ---------------------------------------------------------------------------

setlocal

set "DIR=%~dp0"

REM Prefer Git Bash. `where` returns it first on a normal Git for Windows
REM install; the other hit is the WSL stub in WindowsApps, which also works but
REM would build inside the Linux filesystem view instead.
set "BASH="
for /f "delims=" %%i in ('where bash.exe 2^>nul') do (
    if not defined BASH set "BASH=%%i"
)

if not defined BASH (
    echo error: bash not found in PATH.
    echo        Install Git for Windows, or run ./build.sh from WSL.
    exit /b 1
)

"%BASH%" "%DIR%build.sh" %*
exit /b %ERRORLEVEL%
