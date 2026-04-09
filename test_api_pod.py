# debug_apipod.py (place in face2face root directory)
import sys
import os

# Add apipod to path
sys.path.insert(0, r'A:\projects\apipod')

# Import and run the CLI
from apipod.cli import main

if __name__ == "__main__":
    main()