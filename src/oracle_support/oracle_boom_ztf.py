import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_default_device("cpu")

import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from oracle.custom_datasets.BTS import (
    ZTF_passband_to_wavelengths,
    time_dependent_feature_list,
    time_independent_feature_list,
    meta_data_feature_list,
    flag_value,
)
from oracle.presets import get_model

MODEL_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "best_model_f1.pth"

_model = None


def _get_model():
    global _model
    if _model is None:
        m = get_model("BTSv2")
        m.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location="cpu"), strict=False)
        m.eval()
        _model = m
    return _model


def run_oracle(ztf_id, prv_candidates, candidate, cross_matches):
    """Run Oracle BTSv2 classification on a ZTF alert.

    Args:
        ztf_id: ZTF objectId string.
        prv_candidates: list of prior detection dicts (jd, band, magpsf, sigmapsf, ra, dec, ...).
        candidate: current candidate dict (sgscore1, drb, ndethist, ...).
        cross_matches: dict of catalog cross-matches (expects 'AllWISE').

    Returns:
        (class_scores_df, class_scores) tuple, or None if input is empty.
    """
    if not prv_candidates:
        return None

    prv_cand = pd.DataFrame(prv_candidates)
    prv_cand.sort_values("jd", inplace=True)

    prv_cand["jd"] = prv_cand["jd"] - prv_cand["jd"].min()
    prv_cand["band"] = prv_cand["band"].map(ZTF_passband_to_wavelengths)

    coords = SkyCoord(
        ra=prv_cand["ra"].to_numpy() * u.deg,
        dec=prv_cand["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    prv_cand["l"] = coords.galactic.l
    prv_cand["b"] = coords.galactic.b

    wise = (cross_matches or {}).get("AllWISE") or [{}]
    wise0 = wise[0] if wise else {}
    w1 = wise0.get("w1mpro", flag_value)
    w2 = wise0.get("w2mpro", flag_value)
    w3 = wise0.get("w3mpro", flag_value)
    w4 = wise0.get("w4mpro", flag_value)
    prv_cand["W1mag"] = w1
    prv_cand["W2mag"] = w2
    prv_cand["W3mag"] = w3
    prv_cand["W4mag"] = w4
    prv_cand["W1_minus_W3"] = w1 - w3 if isinstance(w1, (int, float)) and isinstance(w3, (int, float)) else flag_value
    prv_cand["W2_minus_W3"] = w2 - w3 if isinstance(w2, (int, float)) and isinstance(w3, (int, float)) else flag_value

    for key in [
        "sgscore1", "sgscore2", "distpsnr1", "distpsnr2",
        "ndethist", "nmtchps", "drb", "ncovhist",
        "sgmag1", "srmag1", "simag1", "szmag1",
        "sgmag2", "srmag2", "simag2", "szmag2",
    ]:
        prv_cand[key] = candidate.get(key, flag_value)

    ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list) + 1))
    for i, col in enumerate(time_dependent_feature_list):
        ts_tensor[0, :, i] = torch.tensor(prv_cand[col].values)

    static_tensor = torch.zeros((1, len(time_independent_feature_list)))
    for i, col in enumerate(time_independent_feature_list):
        try:
            static_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])
        except KeyError:
            static_tensor[0, i] = flag_value

    meta_data_tensor = torch.zeros((1, len(meta_data_feature_list)))
    for i, col in enumerate(meta_data_feature_list):
        try:
            meta_data_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])
        except KeyError:
            meta_data_tensor[0, i] = flag_value

    length = torch.tensor([len(prv_cand)])
    static_tensor = torch.cat((static_tensor, meta_data_tensor), dim=1)

    batch = {"ts": ts_tensor, "static": static_tensor, "length": length}

    model = _get_model()
    with torch.no_grad():
        class_scores = model.predict_class_probabilities(batch)[0]
        class_scores_df = model.predict_conditional_probabilities_df(batch)

    return class_scores_df, class_scores


if __name__ == "__main__":
    import json

    path_aux = Path("../../data/alert_aux.json")
    path_alert = Path("../../data/alert.json")

    with open(path_aux) as f:
        aux = json.load(f)
    with open(path_alert) as f:
        alert = json.load(f)

    df, _ = run_oracle(
        ztf_id=alert.get("objectId", "test"),
        prv_candidates=aux["prv_candidates"],
        candidate=alert["candidate"],
        cross_matches=aux["cross_matches"],
    )
    print(df)
