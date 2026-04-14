import io
import logging

import fastavro
from confluent_kafka import Consumer

from oracle_support.oracle_boom_ztf import run_oracle

LOG_FILE = "oracle_ztf.log"
KAFKA_TOPIC = "ZTF_alerts_results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def read_avro(msg):
    bytes_io = io.BytesIO(msg.value())
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record
    return None


thumbnail_types = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]

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

            for cutout in thumbnail_types:
                record.pop(cutout, None)

            ztf_id = record["objectId"]
            logger.info("[%s] Running Oracle", ztf_id)

            try:
                result = run_oracle(
                    ztf_id=ztf_id,
                    prv_candidates=record.get("prv_candidates") or [],
                    candidate=record.get("candidate") or {},
                    cross_matches=record.get("cross_matches") or {},
                )
            except Exception:
                logger.exception("[%s] run_oracle failed", ztf_id)
                result = None

            if result is not None:
                df, _ = result
                output = df.to_dict(orient="records")
                print(f"[{ztf_id}] {output}")
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
