import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging

import numpy as np
import pandas as pd
import torch
torch.set_default_device("cpu")

from pathlib import Path

logger = logging.getLogger(__name__)

from oracle.custom_datasets.ELAsTiCC import (
    LSST_passband_to_wavelengths,
    time_dependent_feature_list,
)
from oracle.presets import get_model

MODEL_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "best_model_f1_lsst.pth"
DETECTION_SNR_THRESHOLD = 5.0

_model = None


def _get_model():
    global _model
    if _model is None:
        m = get_model("ELAsTiCCv2-lite")
        m.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"), strict=False)
        m.eval()
        _model = m
    return _model


def get_taxonomy():
    return _get_model().taxonomy


def run_oracle(object_id, prv_candidates):
    """Run Oracle ELAsTiCCv2-lite classification on an LSST alert.

    Args:
        object_id: LSST diaObjectId string.
        prv_candidates: list of prior detection dicts (jd, band, psfFlux, psfFluxErr, snr, ...).

    Returns:
        (class_scores_df, class_scores) tuple, or None if input is empty.
    """
    if not prv_candidates:
        return None

    prv_cand = pd.DataFrame(prv_candidates)
    prv_cand.sort_values("jd", inplace=True)

    for col in ("jd", "band", "psfFlux", "psfFluxErr", "snr"):
        if col not in prv_cand.columns:
            logger.warning("[%s] prv_candidates missing column '%s'", object_id, col)
            continue
        n_null = prv_cand[col].isna().sum()
        if n_null:
            logger.warning("[%s] prv_candidates['%s'] has %d/%d null values",
                           object_id, col, int(n_null), len(prv_cand))

    prv_cand["MJD"] = prv_cand["jd"] - prv_cand["jd"].min()

    unmapped = set(prv_cand["band"].dropna().unique()) - set(LSST_passband_to_wavelengths)
    if unmapped:
        logger.warning("[%s] unknown bands in prv_candidates: %s", object_id, sorted(unmapped))
    prv_cand["BAND"] = prv_cand["band"].map(LSST_passband_to_wavelengths)

    prv_cand["FLUXCAL"] = prv_cand["psfFlux"]
    prv_cand["FLUXCALERR"] = prv_cand["psfFluxErr"]
    prv_cand["PHOTFLAG"] = [
        1 if (x is not None and not (isinstance(x, float) and np.isnan(x)) and x >= DETECTION_SNR_THRESHOLD) else 0
        for x in prv_cand["snr"]
    ]

    ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list)))
    for i, col in enumerate(time_dependent_feature_list):
        if col not in prv_cand.columns:
            logger.warning("[%s] time-dependent feature '%s' missing", object_id, col)
            continue
        vals = prv_cand[col].values
        n_nan = int(pd.isna(vals).sum())
        if n_nan:
            logger.warning("[%s] time-dependent feature '%s' has %d/%d NaN values",
                           object_id, col, n_nan, len(vals))
        ts_tensor[0, :, i] = torch.tensor(vals)

    length = torch.tensor([len(prv_cand)], dtype=torch.long)
    batch = {"ts": ts_tensor, "length": length}

    model = _get_model()
    with torch.no_grad():
        class_scores = model.predict_class_probabilities(batch)[0]
        class_scores_df = model.predict_class_probabilities_df(batch)

    return class_scores_df, class_scores


if __name__ == "__main__":
    import json

    path_aux = Path(__file__).resolve().parents[2] / "data" / "lsst_aux.json"
    path_alert = Path(__file__).resolve().parents[2] / "data" / "lsst_alert.json"

    with open(path_aux) as f:
        aux = json.load(f)
    with open(path_alert) as f:
        alert = json.load(f)

    result = run_oracle(
        object_id=alert.get("objectId", "test"),
        prv_candidates=aux["prv_candidates"],
    )
    if result is None:
        print("no result")
    else:
        df, class_scores = result
        class_probs = dict(zip(df.columns, class_scores.tolist()))
        print(class_probs)
