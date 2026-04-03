from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ARTIFACTS = ROOT / "data" / "artifacts"
MODELS = ROOT / "models"
CHARTS = ROOT / "charts"

# Folders should be pre-created in the Docker image to avoid runtime PermissionErrors
# No runtime mkdir here for production safety
