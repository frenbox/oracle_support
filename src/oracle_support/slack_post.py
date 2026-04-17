"""Post Oracle classification results to Slack."""
import logging
import os
import tempfile
from pathlib import Path

import requests

from oracle_support.plot_oracle import plot_oracle_sunburst

logger = logging.getLogger(__name__)

PERSISTENT_CLASSES = ["AGN", "CV", "Varstar"]
TRANSIENT_CLASSES = ["SN-Ia", "SN-II", "SN-Ib/c", "SLSN"]

SLACK_GET_URL = "https://slack.com/api/files.getUploadURLExternal"
SLACK_COMPLETE_URL = "https://slack.com/api/files.completeUploadExternal"
FRITZ_BASE_URL = "https://fritz.science"


LABEL_W = 10
VALUE_W = 9
COL_W = LABEL_W + 1 + VALUE_W


def _fmt_pct(p):
    if p is None:
        s = "n/a"
    else:
        try:
            pct = float(p) * 100
        except (TypeError, ValueError):
            s = "n/a"
        else:
            if pct > 0 and pct < 0.01:
                s = "<0.01%"
            else:
                s = f"{pct:.2f}%"
    return f"{s:>{VALUE_W}}"


def _col(label, value_str):
    return f"{label:<{LABEL_W}} {value_str}"


def _fritz_url(ztf_id):
    """Return the Fritz source URL if the source exists, otherwise the alerts URL."""
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        return f"{FRITZ_BASE_URL}/alerts/ztf/{ztf_id}"
    try:
        r = requests.get(
            f"{FRITZ_BASE_URL}/api/sources/{ztf_id}",
            headers={"Authorization": f"token {fritz_token}"},
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("status") == "success":
            return f"{FRITZ_BASE_URL}/source/{ztf_id}"
    except Exception:
        logger.debug("[%s] Fritz source check failed, falling back to alerts URL", ztf_id)
    return f"{FRITZ_BASE_URL}/alerts/ztf/{ztf_id}"


def format_message(ztf_id, cond_probs):
    """Build a Slack-mrkdwn-formatted message from class probabilities."""
    fritz_link = _fritz_url(ztf_id)
    p_pers = cond_probs.get("Persistent", 0)
    p_tran = cond_probs.get("Transient", 0)
    lines = [
        f"*Oracle BTSv2 — <{fritz_link}|{ztf_id}>*",
        "```",
        f"{_col('Persistent', _fmt_pct(p_pers))}    {_col('Transient', _fmt_pct(p_tran))}",
    ]
    n = max(len(PERSISTENT_CLASSES), len(TRANSIENT_CLASSES))
    for i in range(n):
        if i < len(PERSISTENT_CLASSES):
            cls = PERSISTENT_CLASSES[i]
            left = _col(f"  {cls}", _fmt_pct(cond_probs.get(cls)))
        else:
            left = " " * COL_W
        if i < len(TRANSIENT_CLASSES):
            cls = TRANSIENT_CLASSES[i]
            right = _col(f"  {cls}", _fmt_pct(cond_probs.get(cls)))
        else:
            right = ""
        lines.append(f"{left}    {right}")
    lines.append("```")
    return "\n".join(lines)


def generate_image(ztf_id, cond_probs, out_dir=None):
    fig = plot_oracle_sunburst(cond_probs, ztf_id=ztf_id)
    out_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out = out_dir / f"{ztf_id}_oracle_sunburst.png"
    fig.write_image(str(out), scale=2)
    return out


def post_to_slack(ztf_id, cond_probs, token=None, channel=None, image_path=None):
    """Upload the sunburst image and a formatted probability block to Slack.

    Returns the Slack file id on success, or None if the env isn't configured.
    """
    token = token or os.getenv("SLACK_ORACLE_BOT_TOKEN")
    channel = channel or os.getenv("SLACK_ORACLE_CHANNEL_ID")
    if not token or not channel:
        logger.warning("[%s] SLACK_ORACLE_BOT_TOKEN/CHANNEL_ID not set, skipping post", ztf_id)
        return None

    if image_path is None:
        image_path = generate_image(ztf_id, cond_probs)
    image_path = Path(image_path)
    message = format_message(ztf_id, cond_probs)

    size = image_path.stat().st_size
    r = requests.get(
        SLACK_GET_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"filename": image_path.name, "length": size},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        logger.error("[%s] Slack getUploadURLExternal failed: %s", ztf_id, j)
        return None
    upload_url = j["upload_url"]
    file_id = j["file_id"]

    with open(image_path, "rb") as f:
        r = requests.post(upload_url, files={"file": (image_path.name, f)}, timeout=60)
    r.raise_for_status()

    r = requests.post(
        SLACK_COMPLETE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json={
            "files": [{"id": file_id, "title": f"{ztf_id} Oracle BTSv2"}],
            "channel_id": channel,
            "initial_comment": message,
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        logger.error("[%s] Slack completeUploadExternal failed: %s", ztf_id, j)
        return None
    return file_id
