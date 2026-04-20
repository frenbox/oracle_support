"""Post Oracle classification results to Slack.

Generic across Oracle taxonomies. Channel and bot token are read from env
vars whose names can be overridden per consumer (e.g. LSST vs ZTF channels).
"""
import logging
import os
import tempfile
from pathlib import Path

import requests

from oracle_support.plot_oracle import plot_oracle_sunburst

logger = logging.getLogger(__name__)

SLACK_GET_URL = "https://slack.com/api/files.getUploadURLExternal"
SLACK_COMPLETE_URL = "https://slack.com/api/files.completeUploadExternal"

LABEL_W = 18
VALUE_W = 9


def _fmt_pct(p):
    if p is None:
        s = "n/a"
    else:
        try:
            pct = float(p) * 100
        except (TypeError, ValueError):
            s = "n/a"
        else:
            if 0 < pct < 0.01:
                s = "<0.01%"
            else:
                s = f"{pct:.2f}%"
    return f"{s:>{VALUE_W}}"


def format_message(object_id, class_probs, title="Oracle", link=None, top_n=8):
    """Build a Slack-mrkdwn message listing classes by probability.

    Pass top_n=None to list every class. Works for any taxonomy — no
    hardcoded class names.
    """
    if link:
        header = f"*{title} — <{link}|{object_id}>*"
    else:
        header = f"*{title} — {object_id}*"
    ranked = sorted(
        class_probs.items(),
        key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0),
    )
    if top_n is not None:
        ranked = ranked[:top_n]
    lines = [header, "```"]
    for name, p in ranked:
        lines.append(f"{name:<{LABEL_W}} {_fmt_pct(p)}")
    lines.append("```")
    return "\n".join(lines)


def generate_image(object_id, class_probs, taxonomy, title="Oracle", out_dir=None, font_size=12):
    plot_title = f"{title} — {object_id}"
    fig = plot_oracle_sunburst(class_probs, taxonomy, title=plot_title, font_size=font_size)
    out_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out = out_dir / f"{object_id}_oracle_sunburst.png"
    fig.write_image(str(out), scale=2)
    return out


def post_to_slack(
    object_id,
    class_probs,
    taxonomy,
    title="Oracle",
    link=None,
    token=None,
    channel=None,
    token_env="SLACK_ORACLE_BOT_TOKEN",
    channel_env="SLACK_ORACLE_CHANNEL_ID",
    image_path=None,
    top_n=8,
    font_size=12,
):
    """Upload a sunburst image and formatted probability block to Slack.

    Token and channel default to the env vars named by token_env/channel_env
    so different consumers (ZTF, LSST, ...) can target different channels.
    Pass explicit token/channel to override.

    Returns the Slack file id on success, or None if env isn't configured.
    """
    token = token or os.getenv(token_env)
    channel = channel or os.getenv(channel_env)
    if not token or not channel:
        logger.warning("[%s] %s/%s not set, skipping post", object_id, token_env, channel_env)
        return None

    if image_path is None:
        image_path = generate_image(object_id, class_probs, taxonomy, title=title, font_size=font_size)
    image_path = Path(image_path)
    message = format_message(object_id, class_probs, title=title, link=link, top_n=top_n)

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
        logger.error("[%s] Slack getUploadURLExternal failed: %s", object_id, j)
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
            "files": [{"id": file_id, "title": f"{object_id} {title}"}],
            "channel_id": channel,
            "initial_comment": message,
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        logger.error("[%s] Slack completeUploadExternal failed: %s", object_id, j)
        return None
    return file_id
