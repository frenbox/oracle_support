import io
import logging
import math
import os
from pathlib import Path

import fastavro
from confluent_kafka import Consumer
from dotenv import load_dotenv
from pymongo import MongoClient

from oracle_support.oracle_boom_lsst import get_taxonomy, run_oracle
from oracle_support.slack_post import format_message, post_to_slack

LOG_FILE = "oracle_lsst.log"
KAFKA_TOPIC = "LSST_alerts_results"
FILTER_NAME = "superphot_lsst"
MODEL_TITLE = "Oracle ELAsTiCCv2-lite"
SLACK_CHANNEL_ENV = "SLACK_ORACLE_LSST_CHANNEL_ID"
SLACK_TOP_N = None  # show every class
SLACK_FONT_SIZE = 9  # smaller labels so long ELAsTiCC names fit
BABAMUL_OBJECT_URL = "https://babamul.caltech.edu/objects/LSST/{object_id}"

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
alerts_coll = _db["LSST_alerts"]
alerts_aux_coll = _db["LSST_alerts_aux"]


def _object_url(object_id):
    """Return the Babamul object page URL for the given LSST diaObjectId."""
    return BABAMUL_OBJECT_URL.format(object_id=object_id)


def read_avro(msg):
    bytes_io = io.BytesIO(msg.value())
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record
    return None


consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "umn_boom_kafka_consumer_group_oracle_lsst",
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

            object_id = record["objectId"]
            candid = record.get("candid")

            passes_filter = any(
                FILTER_NAME in f["filter_name"] for f in record.get("filters") or []
            )
            if not passes_filter:
                logger.debug("[%s] did not pass %s, skipping", object_id, FILTER_NAME)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            aux_doc = alerts_aux_coll.find_one({"_id": object_id})
            alert_doc = alerts_coll.find_one({"_id": candid}) if candid is not None else None

            if aux_doc is None:
                logger.warning("[%s] no aux doc in Mongo, skipping", object_id)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            prv_candidates = aux_doc.get("prv_candidates") or []

            if not prv_candidates:
                logger.warning("[%s] no prv_candidates in aux doc", object_id)

            logger.info("[%s] Running Oracle (prv=%d)", object_id, len(prv_candidates))

            try:
                result = run_oracle(
                    object_id=object_id,
                    prv_candidates=prv_candidates,
                )
            except Exception:
                logger.exception("[%s] run_oracle failed", object_id)
                result = None

            if result is not None:
                class_scores_df, class_scores = result
                scores_list = class_scores.tolist()
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in scores_list):
                    logger.warning("[%s] class_scores contain NaN, skipping post", object_id)
                    total_consumed += 1
                    consumer.commit(message=msg)
                    continue
                class_probs = dict(zip(class_scores_df.columns, scores_list))
                link = _object_url(object_id)
                logger.info("[%s] classification:\n%s",
                            object_id,
                            format_message(object_id, class_probs, title=MODEL_TITLE, link=link, top_n=SLACK_TOP_N))
                try:
                    file_id = post_to_slack(
                        object_id,
                        class_probs,
                        taxonomy=get_taxonomy(),
                        title=MODEL_TITLE,
                        link=link,
                        channel_env=SLACK_CHANNEL_ENV,
                        top_n=SLACK_TOP_N,
                        font_size=SLACK_FONT_SIZE,
                    )
                    if file_id:
                        logger.info("[%s] posted to Slack (file_id=%s)", object_id, file_id)
                except Exception:
                    logger.exception("[%s] Slack post failed", object_id)
            else:
                logger.warning("[%s] no result", object_id)

            total_consumed += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages.", total_consumed)
        consumer.close()


if __name__ == "__main__":
    consume()
