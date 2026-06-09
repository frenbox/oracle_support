import io
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import fastavro
import pandas as pd
import requests
from confluent_kafka import Consumer
from dotenv import load_dotenv
from pymongo import MongoClient

from oracle_support.oracle_boom_ztf import get_taxonomy, run_oracle
from oracle_support.slack_post import format_message, post_to_slack

LOG_FILE = "oracle_ztf.log"
KAFKA_TOPIC = "ZTF_alerts_results"
FILTER_NAME = "rcfdeep_partnership_ztf"
MODEL_TITLE = "Oracle Omni"
FRITZ_BASE_URL = "https://fritz.science"
RESULTS_CSV = Path("results") / "oracle_ztf_results.csv"
POST_TO_SLACK = False  # Slack updates paused; flip to True to re-enable.
FRITZ_GROUP_IDS = [1959]  # Oracle Omni Beta — annotation visibility scope.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
for _noisy in ("choreographer", "kaleido"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv(Path.home() / ".env")
_mongo_user = os.getenv("BOOM_DATABASE__USERNAME")
_mongo_pass = os.getenv("BOOM_DATABASE__PASSWORD")
_mongo_url = (
    f"mongodb://{_mongo_user}:{_mongo_pass}@localhost:27017"
    if _mongo_user and _mongo_pass
    else "mongodb://localhost:27017"
)
_db = MongoClient(_mongo_url)["boom"]
alerts_coll = _db["ZTF_alerts"]
alerts_aux_coll = _db["ZTF_alerts_aux"]


def _fritz_classifications(ztf_id):
    """Return Fritz classifications for the source, or [] on missing/error."""
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        return []
    try:
        r = requests.get(
            f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/classifications",
            headers={"Authorization": f"token {fritz_token}"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        j = r.json()
        if j.get("status") != "success":
            return []
        return j.get("data") or []
    except Exception:
        logger.exception("[%s] Fritz classifications fetch failed", ztf_id)
        return []


def _format_fritz_block(classifications):
    """Format a list of Fritz classification dicts as a Slack-mrkdwn block."""
    if not classifications:
        return None
    parts = []
    for c in classifications[:5]:
        name = c.get("classification") or "?"
        prob = c.get("probability")
        if isinstance(prob, (int, float)):
            parts.append(f"{name} ({prob:.2f})")
        else:
            parts.append(name)
    return "*Fritz:* " + ", ".join(parts)


def _append_csv(ztf_id, class_probs, fritz_classifications):
    """Append one row to the rolling results CSV (creates header on first write)."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fritz_str = "; ".join(
        f"{c.get('classification', '?')}({c.get('probability')})"
        for c in (fritz_classifications or [])
    )
    row = {
        "objectId": ztf_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in class_probs.items()},
        "fritz_classifications": fritz_str,
    }
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV,
        mode="a",
        index=False,
        header=not RESULTS_CSV.exists(),
    )


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


def _ordered_class_probs(class_probs, taxonomy):
    """Order probabilities as a pre-order walk of the taxonomy tree.

    Each parent is immediately followed by its subclasses (Persistent → AGN,
    CV, Varstar; Transient → SN-Ia, SN-II, SN-Ib/c, SLSN). The root node
    ("Alert") is dropped. Any probability keys not found in the taxonomy are
    appended at the end in their original order (defensive).
    """
    order = []

    def _walk(node):
        for child in taxonomy.successors(node):
            order.append(child)
            _walk(child)

    _walk(taxonomy.root_label)
    ordered = {n: class_probs[n] for n in order if n in class_probs}
    for k, v in class_probs.items():
        if k not in ordered and k != taxonomy.root_label:
            ordered[k] = v
    return ordered


def _get_annotation_id(ztf_id, origin, headers):
    """Fetch the annotation ID for a source by origin, or None."""
    endpoint = f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/annotations"
    try:
        resp_json = requests.get(endpoint, headers=headers, timeout=10).json()
        if resp_json.get("status") == "success":
            for ann in resp_json.get("data", []):
                if ann.get("origin") == origin:
                    return ann.get("annotation_id") or ann.get("id")
    except Exception:
        logger.exception("[%s] Failed to fetch existing annotations", ztf_id)
    return None


def annotate_fritz(class_probs, ztf_id, taxonomy, origin="oracle_omni",
                   group_ids=None, previous_annotation_id=None):
    """Post (or update) per-class probability annotations on Fritz.

    The probability dict is ordered as a pre-order tree walk (Persistent and
    its subclasses, then Transient and its classes) with the root "Alert"
    field dropped. On a duplicate-origin POST failure, falls back to fetching
    the existing annotation and updating it via PUT.

    Returns the annotation_id of the created/updated annotation, or None.
    """
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        logger.warning("[%s] FRITZ_TOKEN not set, skipping annotation", ztf_id)
        return None

    headers = {
        "Authorization": f"token {fritz_token}",
        "Content-Type": "application/json",
    }

    data = {
        k: float(v) if isinstance(v, (int, float)) else v
        for k, v in _ordered_class_probs(class_probs, taxonomy).items()
    }
    payload = {"origin": origin, "data": data}
    if group_ids is not None:
        payload["group_ids"] = group_ids

    base = f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/annotations"
    if previous_annotation_id is not None:
        response = requests.put(f"{base}/{previous_annotation_id}", json=payload,
                                headers=headers, timeout=10)
    else:
        response = requests.post(base, json=payload, headers=headers, timeout=10)

    resp_json = response.json()
    if resp_json.get("status") == "success":
        data_resp = resp_json.get("data", {})
        logger.info("[%s] Annotation saved.", ztf_id)
        return data_resp.get("annotation_id") or data_resp.get("id")

    # POST likely failed due to duplicate origin — fetch existing and PUT.
    logger.warning("[%s] Annotation POST failed: %s", ztf_id, resp_json.get("message"))
    if previous_annotation_id is None:
        existing_id = _get_annotation_id(ztf_id, origin, headers)
        if existing_id is not None:
            retry = requests.put(f"{base}/{existing_id}", json=payload,
                                 headers=headers, timeout=10).json()
            if retry.get("status") == "success":
                logger.info("[%s] Annotation updated via fallback PUT.", ztf_id)
                return existing_id
            logger.error("[%s] Fallback PUT failed: %s", ztf_id, retry.get("message"))
    return None


def read_avro(msg):
    bytes_io = io.BytesIO(msg.value())
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record
    return None


consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "umn_boom_kafka_consumer_group_oracle_ztf",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
    "session.timeout.ms": 6000,
    "max.poll.interval.ms": 300000,
    "security.protocol": "PLAINTEXT",
})
consumer.subscribe([KAFKA_TOPIC])
logger.info("Subscribed to topic: %s", KAFKA_TOPIC)


def consume():
    logger.info("Listening for messages...")
    total_consumed = 0
    consecutive_empty_polls = 0

    try:
        while True:
            msg = consumer.poll(timeout=10.0)
            if msg is None:
                consecutive_empty_polls += 1
                if consecutive_empty_polls % 6 == 1:
                    logger.info("No new messages (idle ~%ds, consumed %d)",
                                consecutive_empty_polls * 10, total_consumed)
                continue
            consecutive_empty_polls = 0
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            record = read_avro(msg)
            if record is None:
                logger.error("Failed to deserialize Avro at offset %s", msg.offset())
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            ztf_id = record["objectId"]
            candid = record.get("candid")

            passes_filter = any(
                FILTER_NAME in f["filter_name"] for f in record.get("filters") or []
            )
            if not passes_filter:
                logger.debug("[%s] did not pass %s, skipping", ztf_id, FILTER_NAME)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            aux_doc = alerts_aux_coll.find_one({"_id": ztf_id})
            alert_doc = alerts_coll.find_one({"_id": candid}) if candid is not None else None

            if aux_doc is None:
                logger.warning("[%s] no aux doc in Mongo, skipping", ztf_id)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            prv_candidates = aux_doc.get("prv_candidates") or []
            cross_matches = aux_doc.get("cross_matches") or {}
            candidate = (alert_doc or {}).get("candidate") or {}
            cutouts = {
                "cutoutScience": record.get("cutoutScience"),
                "cutoutTemplate": record.get("cutoutTemplate"),
                "cutoutDifference": record.get("cutoutDifference"),
            }

            if not prv_candidates:
                logger.warning("[%s] no prv_candidates in aux doc", ztf_id)

            logger.info("[%s] Running Oracle (prv=%d)", ztf_id, len(prv_candidates))

            try:
                result = run_oracle(
                    ztf_id=ztf_id,
                    prv_candidates=prv_candidates,
                    candidate=candidate,
                    cross_matches=cross_matches,
                    cutouts=cutouts,
                )
            except Exception:
                logger.exception("[%s] run_oracle failed", ztf_id)
                result = None

            if result is not None:
                cond_probs_df, class_scores = result
                scores_list = class_scores.tolist()
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in scores_list):
                    logger.warning("[%s] class_scores contain NaN, skipping post", ztf_id)
                    total_consumed += 1
                    consumer.commit(message=msg)
                    continue
                class_probs = dict(zip(cond_probs_df.columns, scores_list))
                link = _fritz_url(ztf_id)

                fritz_classifications = _fritz_classifications(ztf_id)
                _append_csv(ztf_id, class_probs, fritz_classifications)

                try:
                    annotation_id = annotate_fritz(class_probs, ztf_id, get_taxonomy(),
                                                   group_ids=FRITZ_GROUP_IDS)
                    logger.info("[%s] Fritz annotation: %s", ztf_id, annotation_id)
                except Exception:
                    logger.exception("[%s] Fritz annotation failed", ztf_id)

                fritz_block = _format_fritz_block(fritz_classifications)
                logger.info("[%s] classification:\n%s",
                            ztf_id, format_message(ztf_id, class_probs, title=MODEL_TITLE,
                                                   link=link, extra_text=fritz_block))

                if not POST_TO_SLACK:
                    logger.debug("[%s] Slack posting disabled, skipping", ztf_id)
                elif not fritz_classifications:
                    logger.info("[%s] no Fritz classification, skipping Slack post", ztf_id)
                else:
                    try:
                        file_id = post_to_slack(
                            ztf_id,
                            class_probs,
                            taxonomy=get_taxonomy(),
                            title=MODEL_TITLE,
                            link=link,
                            channel_env="SLACK_ORACLE_CHANNEL_ID",
                            extra_text=fritz_block,
                        )
                        if file_id:
                            logger.info("[%s] posted to Slack (file_id=%s)", ztf_id, file_id)
                    except Exception:
                        logger.exception("[%s] Slack post failed", ztf_id)
            else:
                logger.warning("[%s] no result", ztf_id)

            total_consumed += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages.", total_consumed)
        consumer.close()


if __name__ == "__main__":
    consume()
