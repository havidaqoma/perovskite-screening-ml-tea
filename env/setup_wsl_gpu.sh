#!/usr/bin/env bash
# Run from the repo root. Builds the WSL GPU env for Stage 3 (Roost baseline) and Stage 4 (uMLIP relaxations).
# Separate from the main Windows .venv on purpose: torch/CUDA only live here.
set -euo pipefail
ENV=~/venvs/mlperov-gpu
uv venv "$ENV" --python 3.12 -q --allow-existing
PY="$ENV/bin/python"
uv pip install -q --python "$PY" torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python "$PY" "pymatgen==2026.3.23" "pymatgen-core==2026.4.16" "numpy==2.4.3" "pandas==2.3.3" \
    "scipy==1.17.1" ase mace-torch "git+https://github.com/CompRhys/aviary"
"$PY" - <<'EOF'
import torch, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) != 0)
for p in ("mace-torch", "aviary", "ase", "pymatgen", "e3nn"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING", e)
EOF
"$PY" -m pip freeze > env/freeze-wsl-gpu.txt 2>/dev/null || uv pip freeze --python "$PY" > env/freeze-wsl-gpu.txt
echo DONE
