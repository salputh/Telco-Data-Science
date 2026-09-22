#!/usr/bin/env bash
# setup_ds_env.sh — Bootstrap a Python data science working environment.
# Usage: ./setup_ds_env.sh [project_name]
# Safe to re-run: it never overwrites files/folders that already exist.

set -e

PROJECT_NAME="${1:-ds-project}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Setting up data science environment for '$PROJECT_NAME'"

# 1. Folder structure
for dir in data/raw data/interim data/processed notebooks src models reports/figures tests; do
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
    echo "  created $dir/"
  fi
done

# 2. Virtual environment
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
  echo "  created .venv"
else
  echo "  .venv already exists, skipping"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip --quiet

# 3. Core packages
CORE_PACKAGES="numpy pandas matplotlib seaborn scikit-learn jupyterlab ipykernel scipy statsmodels"
echo "==> Installing core packages: $CORE_PACKAGES"
pip install $CORE_PACKAGES --quiet

# 4. Freeze requirements
pip freeze > requirements.txt
echo "  wrote requirements.txt"

# 5. Register Jupyter kernel
python -m ipykernel install --user --name="$PROJECT_NAME" --display-name "Python ($PROJECT_NAME)" >/dev/null
echo "  registered Jupyter kernel: $PROJECT_NAME"

# 6. .gitignore
if [ ! -f ".gitignore" ]; then
  cat > .gitignore << 'EOF'
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*
!data/raw/.gitkeep
.env
*.egg-info/
.DS_Store
EOF
  echo "  wrote .gitignore"
fi

# 7. Git init (only if not already a repo)
if [ ! -d ".git" ]; then
  git init -q
  echo "  initialized git repo"
fi

# 8. Verify
python -c "import pandas, numpy, sklearn, matplotlib; print('Environment OK: pandas', pandas.__version__, '| scikit-learn', sklearn.__version__)"

echo "==> Done. Activate with: source .venv/bin/activate"
