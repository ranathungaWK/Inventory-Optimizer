import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from app.utils.artifacts import (
        MODEL_BOOSTER_PATH,
        MODEL_CONTRACT_PATH,
        DIAG_REPORT_PATH,
        SCORING_CONTRACT_PATH,
        SCORING_MANIFEST_PATH,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from utils.artifacts import (
            MODEL_BOOSTER_PATH,
            MODEL_CONTRACT_PATH,
            DIAG_REPORT_PATH,
            SCORING_CONTRACT_PATH,
            SCORING_MANIFEST_PATH,
        )
    except (ImportError, ModuleNotFoundError):
        from .artifacts import (
            MODEL_BOOSTER_PATH,
            MODEL_CONTRACT_PATH,
            DIAG_REPORT_PATH,
            SCORING_CONTRACT_PATH,
            SCORING_MANIFEST_PATH,
        )

def compute_file_sha256(filepath: Path) -> str:
    """Computes full 64-character SHA-256 hex digest of a file."""
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_pipeline_integrity() -> Dict[str, Any]:
    """Validates complete pipeline artifact health, checksums, and readiness gates."""
    status = {
        "is_healthy": True,
        "model_sha256_match": False,
        "model_sha256": "",
        "expected_sha256": "",
        "diagnostics_passed": False,
        "scoring_manifest_valid": False,
        "total_partitions": 0,
        "issues": []
    }

    # 1. Model SHA-256 Check
    if MODEL_BOOSTER_PATH.exists():
        actual_hash = compute_file_sha256(MODEL_BOOSTER_PATH)
        status["model_sha256"] = actual_hash
        
        # Check against diagnostic report
        if DIAG_REPORT_PATH.exists():
            try:
                with open(DIAG_REPORT_PATH, "r") as f:
                    diag = json.load(f)
                expected_hash = diag.get("model_sha256", "")
                status["expected_sha256"] = expected_hash
                if actual_hash == expected_hash and len(actual_hash) == 64:
                    status["model_sha256_match"] = True
                else:
                    status["is_healthy"] = False
                    status["issues"].append("Model SHA-256 hash mismatch")
                
                # Check NB05 gating
                if diag.get("diagnostic_conclusion", {}).get("production_block") is False:
                    status["diagnostics_passed"] = True
                else:
                    status["is_healthy"] = False
                    status["issues"].append("NB05 diagnostic gate flagged production block")
            except Exception as e:
                status["is_healthy"] = False
                status["issues"].append(f"Failed to read diagnostic report: {str(e)}")
    else:
        status["is_healthy"] = False
        status["issues"].append("Frozen LightGBM model booster file missing")

    # 2. Scoring Manifest Check
    if SCORING_MANIFEST_PATH.exists():
        try:
            with open(SCORING_MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
            n_part = manifest.get("total_partitions", 0)
            status["total_partitions"] = n_part
            if n_part == 227:
                status["scoring_manifest_valid"] = True
            else:
                status["issues"].append(f"Expected 227 partitions, found {n_part}")
        except Exception as e:
            status["is_healthy"] = False
            status["issues"].append(f"Failed to read scoring manifest: {str(e)}")
    else:
        status["is_healthy"] = False
        status["issues"].append("Scoring manifest file missing")

    return status
