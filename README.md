# oracle-support

Bridge between the [BOOM](https://github.com/Theodlz/boom) alert broker and the
[Oracle](https://github.com/dev-ved30/Oracle) classifier. Consumes ZTF and LSST
alerts from Kafka, builds the feature batch the Oracle model expects, runs
inference on CPU, and posts results to Slack / a rolling CSV.

## Pipeline

```
Kafka (BOOM filter topic)  ->  MongoDB (BOOM)  ->  oracle-support  ->  Oracle model  ->  Slack + CSV
```

- Subscribes to a BOOM filter topic (`ZTF_alerts_results` or `LSST_alerts_results`).
- Looks up the matching `*_alerts` and `*_alerts_aux` documents in Mongo to
  recover the full photometry history, cross-matches, and reference cutout.
- Assembles a `(ts, static, length, postage_stamp)` batch and calls the model.
- Posts an annotated probability plot to Slack for sources Fritz has already
  classified; appends every classification to `results/oracle_*_results.csv`.

## Layout

```
src/oracle_support/
  alerts_consumer_ztf.py   - Kafka consumer, Mongo lookup, Slack/CSV write (ZTF)
  alerts_consumer_lsst.py  - same, for LSST
  oracle_boom_ztf.py       - feature assembly + model inference (ZTF)
  oracle_boom_lsst.py      - feature assembly + model inference (LSST)
  plot_oracle.py           - probability-tree plot used in Slack posts
  slack_post.py            - Slack file upload + message formatting
data/                      - model weights (best_model_f1_{ztf,lsst}.pth)
results/                   - rolling result CSVs
```

## Setup

Requires Python >=3.11, MongoDB and Kafka reachable on localhost, and Oracle's
model weights in `data/`.

```bash
poetry install
```

Create `~/.env` with at least:

```
BOOM_DATABASE__USERNAME=...
BOOM_DATABASE__PASSWORD=...
FRITZ_TOKEN=...                 # optional; needed for Fritz classifications
SLACK_BOT_TOKEN=...             # optional; needed for Slack posts
SLACK_ORACLE_CHANNEL_ID=...
SLACK_ORACLE_LSST_CHANNEL_ID=...
```

## Run

```bash
python -m oracle_support.alerts_consumer_ztf
python -m oracle_support.alerts_consumer_lsst
```

Each consumer logs to `oracle_{ztf,lsst}.log`, commits Kafka offsets per
message, and skips alerts that fail the configured BOOM filter
(`rcfdeep_partnership_ztf`, `superphot_lsst`).
