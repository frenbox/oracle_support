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
- Writes per-class probabilities to Fritz as a source annotation (see below),
  appends every classification to `results/oracle_*_results.csv`, and (optionally)
  posts an annotated probability plot to Slack.

## Fritz annotation

The ZTF consumer posts the model's per-class probabilities to
[Fritz](https://fritz.science) as a source **annotation** with origin
`oracle_omni` (`annotate_fritz` in `alerts_consumer_ztf.py`). One annotation is
maintained per source and overwritten on each new alert.

- **Ordering.** The probability dict is emitted as a pre-order walk of the model
  taxonomy — each parent immediately followed by its subclasses (Persistent →
  AGN, CV, Varstar; Transient → SN-Ia, SN-II, SN-Ib/c, SLSN). The root `Alert`
  node is dropped. (Fritz stores annotation data as an unordered map, so its UI
  may display the keys in a different order than they are sent.)
- **Visibility.** Scoped to the group ids in `FRITZ_GROUP_IDS` (default `[1959]`,
  *Oracle Omni Beta*). Without explicit group ids Fritz would default to *all*
  of the token owner's groups, including the sitewide group — i.e. public.
- **Updates.** Stateless: it POSTs a new annotation and, if one already exists
  for the origin, falls back to fetching it and updating via PUT. No local state
  file is kept.
- **Token.** Uses `FRITZ_TOKEN`, which must have the **Annotate** ACL on Fritz
  (read-only tokens return HTTP 401 on the annotation POST).

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
FRITZ_TOKEN=...                 # needed for Fritz classifications + annotation (Annotate ACL)
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

Two toggles at the top of `alerts_consumer_ztf.py`:

- `POST_TO_SLACK` — gate Slack posting (currently `False`; flip to `True` to
  re-enable). Fritz annotation and the CSV write run regardless.
- `FRITZ_GROUP_IDS` — Fritz group ids the annotation is shared with.
