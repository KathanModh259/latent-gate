@echo off
echo ========================================
echo LatentGate VSCode Extension Builder
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js is not installed
    echo Download from https://nodejs.org/
    exit /b 1
)

REM Check if npm is installed
where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm is not installed
    exit /b 1
)

echo [1/4] Installing dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo [2/4] Compiling TypeScript...
call npm run compile
if %ERRORLEVEL% neq 0 (
    echo ERROR: Compilation failed
    exit /b 1
)

echo.
echo [3/4] Packaging extension...
call npx vsce package
if %ERRORLEVEL% neq 0 (
    echo ERROR: Packaging failed
    exit /b 1
)

echo.
echo [4/4] Done!
echo.
echo Extension packaged successfully!
echo.
echo To install locally:
echo   code --install-extension latent-gate-0.5.0.vsix
echo.
echo To publish to marketplace:
echo   npx vsce publish
echo.
pause
