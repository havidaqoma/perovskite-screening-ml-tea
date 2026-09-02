import os
import sys

# Make `from src...` work regardless of pytest invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
