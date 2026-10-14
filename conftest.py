"""Pytest root configuration ensuring `src` is in the Python search path."""
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
