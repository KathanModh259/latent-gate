#!/usr/bin/env python3
"""
Publish LatentGate to PyPI.

Usage:
    python publish.py              # Build and upload to PyPI
    python publish.py --test       # Build and upload to TestPyPI
    python publish.py --build-only # Build only, don't upload
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result


def clean_dist():
    """Clean dist directory."""
    import shutil
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("Cleaned dist/")


def build_package():
    """Build the package."""
    print("\n=== Building package ===")
    run_command([sys.executable, "-m", "build"])
    print("Build complete!")


def check_package():
    """Check the package with twine."""
    print("\n=== Checking package ===")
    run_command(["twine", "check", "dist/*"])


def upload_to_pypi(test: bool = False):
    """Upload to PyPI or TestPyPI."""
    repository = "testpypi" if test else "pypi"
    print(f"\n=== Uploading to {repository} ===")
    
    cmd = ["twine", "upload", "dist/*"]
    if test:
        cmd.extend(["--repository", "testpypi"])
    
    # Check for API token
    if not os.getenv("TWINE_USERNAME") and not os.getenv("PYPI_API_TOKEN"):
        print("\nWarning: No PyPI credentials found.")
        print("Set TWINE_USERNAME and TWINE_PASSWORD, or PYPI_API_TOKEN")
        print("Or configure ~/.pypirc")
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)
    
    run_command(cmd)
    print("Upload complete!")


def main():
    parser = argparse.ArgumentParser(description="Publish LatentGate to PyPI")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI")
    parser.add_argument("--build-only", action="store_true", help="Build only, don't upload")
    parser.add_argument("--skip-check", action="store_true", help="Skip package check")
    parser.add_argument("--clean", action="store_true", help="Clean dist before building")
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    print("LatentGate Publisher")
    print("=" * 50)
    
    if args.clean:
        clean_dist()
    
    build_package()
    
    if not args.skip_check:
        check_package()
    
    if not args.build_only:
        upload_to_pypi(test=args.test)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
