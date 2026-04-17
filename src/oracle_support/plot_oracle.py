"""Sunburst plot of Oracle BTSv2 hierarchical classification output.

Tree structure (conditional probabilities):

    Alert
    ├── Persistent  (AGN, CV, Varstar)
    └── Transient   (SN-Ia, SN-II, SN-Ib/c, SLSN)
"""
import math

import plotly.graph_objects as go


PERSISTENT_CLASSES = ["AGN", "CV", "Varstar"]
TRANSIENT_CLASSES = ["SN-Ia", "SN-II", "SN-Ib/c", "SLSN"]

COLORS = {
    "Alert": "#636EFA",
    "Persistent": "#EF553B",
    "Transient": "#00CC96",
    "AGN": "#FFA15A",
    "CV": "#AB63FA",
    "Varstar": "#FF6692",
    "SN-Ia": "#19D3F3",
    "SN-II": "#B6E880",
    "SN-Ib/c": "#FF97FF",
    "SLSN": "#FECB52",
}


def plot_oracle_sunburst(scores, ztf_id=None):
    """Create an interactive sunburst figure from class (marginal) probabilities.

    Args:
        scores: dict with keys Alert, Persistent, Transient, AGN, CV, Varstar,
            SN-Ia, SN-II, SN-Ib/c, SLSN (marginal class probabilities; children
            under each parent should sum to the parent's value).
        ztf_id: optional identifier for the title.

    Returns:
        plotly.graph_objects.Figure
    """
    p_alert = scores.get("Alert", 1.0)
    p_persistent = scores.get("Persistent", 0)
    p_transient = scores.get("Transient", 0)

    ids = ["Alert", "Persistent", "Transient"]
    labels = ["<b>Alert</b>", "<b>Persistent</b>", "<b>Transient</b>"]
    parents = ["", "Alert", "Alert"]
    values = [p_alert, p_persistent, p_transient]
    texts = [f"<b>{_fmt(p_alert)}</b>", f"<b>{_fmt(p_persistent)}</b>", f"<b>{_fmt(p_transient)}</b>"]
    colors = [COLORS["Alert"], COLORS["Persistent"], COLORS["Transient"]]

    for cls in PERSISTENT_CLASSES:
        p = scores.get(cls, 0)
        ids.append(cls)
        labels.append(f"<b>{cls}</b>")
        parents.append("Persistent")
        values.append(p)
        texts.append(f"<b>{_fmt(p)}</b>")
        colors.append(COLORS[cls])

    for cls in TRANSIENT_CLASSES:
        p = scores.get(cls, 0)
        ids.append(cls)
        labels.append(f"<b>{cls}</b>")
        parents.append("Transient")
        values.append(p)
        texts.append(f"<b>{_fmt(p)}</b>")
        colors.append(COLORS[cls])

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        textinfo="label+text",
        textfont=dict(size=14),
        hovertemplate="<b>%{label}</b><br>P(class) = %{text}<extra></extra>",
        branchvalues="total",
        marker=dict(colors=colors, line=dict(width=2, color="white")),
        insidetextorientation="radial",
    ))

    title = f"Oracle BTSv2 — {ztf_id}" if ztf_id else "Oracle BTSv2"
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=20, family="Arial")),
        margin=dict(t=60, l=10, r=10, b=10),
        width=650,
        height=650,
    )
    return fig


def _fmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NaN"
    if p < 1e-4:
        return "<0.01%"
    return f"{p * 100:.2f}%"


if __name__ == "__main__":
    sample = {
        "Alert": 1.0,
        "Persistent": 0.7298641204833984,
        "Transient": 0.27013593912124634,
        "AGN": 0.0009099491871893406,
        "CV": 0.9972658157348633,
        "Varstar": 0.0018241567304357886,
        "SN-Ia": 0.35389694571495056,
        "SN-II": 0.5109876394271851,
        "SN-Ib/c": 0.13468529284000397,
        "SLSN": 0.0004300586588215083,
    }
    ztf_id = "ZTF26aaqylle"
    fig = plot_oracle_sunburst(sample, ztf_id=ztf_id)
    out = f"{ztf_id}_oracle_sunburst.png"
    fig.write_image(out, scale=2)
    print(f"saved: {out}")
