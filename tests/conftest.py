"""
Pytest configuration.

Sets thread-count env vars before any import to avoid the OpenBLAS/OMP
deadlock on Python 3.14 + Windows (sklearn/numpy thread pool init).
Must be the very first thing executed — hence conftest.py.
"""
import os
import sys

# Ensure user site-packages are on path (needed on some Windows Python 3.14 setups)
import site
for sp in site.getusersitepackages() if hasattr(site, "getusersitepackages") else []:
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Fix OpenBLAS / OMP deadlock on Windows + Python 3.14
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Test DB / service overrides
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_shlp.db")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_tests")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./test_mlruns.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "shlp_test")
