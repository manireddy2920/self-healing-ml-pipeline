import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./debug_mlruns.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "debug")

from src.config import get_settings
from src.ingestion.generator import generate_reference, generate_batch
from src.retraining.trainer import train_and_log, load_model
from src.ingestion.schema import ALL_FEATURES, TARGET
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

cfg = get_settings()
cfg.__dict__["promotion_threshold_delta"] = 0.05

ref     = generate_reference(n=5_000, seed=0)
drifted = generate_batch(n=2_000, seed=200, drift_alpha=1.0 * 2.0)  # severe drift

champ_r = train_and_log(ref, run_name="champ", hyperparams={"n_estimators": 50, "max_depth": 4})

import pandas as pd
combined = pd.concat([ref, drifted], ignore_index=True).sample(frac=1, random_state=42)
print(f"Combined size: {len(combined):,} (ref={len(ref):,} + drifted={len(drifted):,})")
chal_r = train_and_log(combined, run_name="chal_combined",
                        hyperparams={"n_estimators": 50, "max_depth": 4})

_, val = train_test_split(ref, test_size=0.25, random_state=99, stratify=ref[TARGET])
X_val, y_val = val[ALL_FEATURES], val[TARGET].values

champ = load_model(champ_r.mlflow_model_uri)
chal  = load_model(chal_r.mlflow_model_uri)

c_f1  = f1_score(y_val, chal.predict(X_val),  zero_division=0)
p_f1  = f1_score(y_val, champ.predict(X_val), zero_division=0)
c_rec = recall_score(y_val, chal.predict(X_val),  zero_division=0)
p_rec = recall_score(y_val, champ.predict(X_val), zero_division=0)

print(f"Champion   F1={p_f1:.4f}  recall={p_rec:.4f}")
print(f"Challenger F1={c_f1:.4f}  recall={c_rec:.4f}")
print(f"F1 gap:       {c_f1 - p_f1:.4f}  (need >= -0.10 with delta=0.10)")
print(f"Recall gap:   {c_rec - p_rec:.4f}  (need >= -0.05)")
print(f"F1 passes:    {c_f1 >= p_f1 - 0.10}")
print(f"Recall passes:{c_rec >= p_rec - 0.05}")
print(f"Would promote:{(c_f1 >= p_f1 - 0.10) and (c_rec >= p_rec - 0.05)}")
