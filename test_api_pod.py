"""Debug helper to launch APIPod CLI from this project root."""

import subprocess
import sys


if __name__ == "__main__":
    # Use module invocation instead of importing apipod internals directly.
    subprocess.run([sys.executable, "-m", "apipod.cli"], check=True)