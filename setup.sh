#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0

def main():
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/void").mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(exist_ok=True)

    # List of steps to run
    steps = [
        ("python -m pip install -r requirements.txt", "Installing dependencies"),
        ("python init_model.py", "Generating model files"),
        ("python verify_setup.py", "Verifying setup"),
        ("pytest", "Running tests")
    ]

    # Run each step
    success = True
    for command, description in steps:
        if not run_command(command, description):
            print(f"\n❌ Failed at: {description}")
            success = False
            break

    if success:
        print("\n✅ All setup steps completed successfully!")
        print("\nYou can now:")
        print("1. Run locally: python chat_api.py")
        print("2. Build Docker: docker build -t void-ai .")
        print("3. Deploy to Render")
    else:
        print("\n❌ Setup failed. Please fix the errors and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
