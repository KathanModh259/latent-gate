#!/bin/bash
echo "========================================"
echo "LatentGate VSCode Extension Builder"
echo "========================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Download from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not installed"
    exit 1
fi

echo "[1/4] Installing dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "[2/4] Compiling TypeScript..."
npm run compile
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo ""
echo "[3/4] Packaging extension..."
npx vsce package
if [ $? -ne 0 ]; then
    echo "ERROR: Packaging failed"
    exit 1
fi

echo ""
echo "[4/4] Done!"
echo ""
echo "Extension packaged successfully!"
echo ""
echo "To install locally:"
echo "  code --install-extension latent-gate-0.5.0.vsix"
echo ""
echo "To publish to marketplace:"
echo "  npx vsce publish"
echo ""
