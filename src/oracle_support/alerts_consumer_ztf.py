import io
import logging
import os
from pathlib import Path

import fastavro
from confluent_kafka import Consumer
from dotenv import load_dotenv
from pymongo import MongoClient

from oracle_support.oracle_boom_ztf import run_oracle
from oracle_support.slack_post import format_message, post_to_slack

LOG_FILE = "oracle_ztf.log"
KAFKA_TOPIC = "ZTF_alerts_results"
FILTER_NAME = "superphot_ztf"
MAX_ORACLE_RUNS = 20

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
    oracle_runs = 0
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

            if not prv_candidates:
                logger.warning("[%s] no prv_candidates in aux doc", ztf_id)

            oracle_runs += 1
            logger.info("[%s] Running Oracle (prv=%d, run %d/%d)",
                        ztf_id, len(prv_candidates), oracle_runs, MAX_ORACLE_RUNS)

            try:
                result = run_oracle(
                    ztf_id=ztf_id,
                    prv_candidates=prv_candidates,
                    candidate=candidate,
                    cross_matches=cross_matches,
                )
            except Exception:
                logger.exception("[%s] run_oracle failed", ztf_id)
                result = None

            if result is not None:
                cond_probs_df, class_scores = result
                class_probs = dict(zip(cond_probs_df.columns, class_scores.tolist()))
                logger.info("[%s] classification:\n%s", ztf_id, format_message(ztf_id, class_probs))
                try:
                    file_id = post_to_slack(ztf_id, class_probs)
                    if file_id:
                        logger.info("[%s] posted to Slack (file_id=%s)", ztf_id, file_id)
                except Exception:
                    logger.exception("[%s] Slack post failed", ztf_id)
            else:
                logger.warning("[%s] no result", ztf_id)

            total_consumed += 1
            consumer.commit(message=msg)

            if oracle_runs >= MAX_ORACLE_RUNS:
                logger.info("Reached MAX_ORACLE_RUNS=%d, stopping.", MAX_ORACLE_RUNS)
                break

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages.", total_consumed)
        consumer.close()


if __name__ == "__main__":
    consume()
