"""Generic sunburst plot of Oracle hierarchical classification output.

Walks the model's taxonomy tree so the same function works for any Oracle
Taxonomy (BTSv2, ELAsTiCC-lite, etc.).
"""
import math

import plotly.graph_objects as go


def _fmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NaN"
    if 0 < p < 1e-4:
        return "<0.01%"
    return f"{p * 100:.2f}%"


def plot_oracle_sunburst(scores, taxonomy, title=None, font_size=12):
    """Create a sunburst figure from marginal class probabilities for any Taxonomy.

    Args:
        scores: dict {class_name: probability} for every taxonomy node.
        taxonomy: Oracle Taxonomy instance (networkx DiGraph with
            get_level_order_traversal / get_parent_nodes).
        title: optional title shown at the top of the figure.
        font_size: point size for leaf labels (use smaller values when labels are long).

    Returns:
        plotly.graph_objects.Figure
    """
    nodes = list(taxonomy.get_level_order_traversal())
    parent_nodes = list(taxonomy.get_parent_nodes())

    ids, labels, parents, values, texts = [], [], [], [], []
    for node, parent in zip(nodes, parent_nodes):
        p = scores.get(node, 0) or 0
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        ids.append(node)
        labels.append(f"<b>{node}</b>")
        parents.append(parent if parent else "")
        values.append(p)
        texts.append(f"<b>{_fmt(p)}</b>")

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        textinfo="label+text",
        textfont=dict(size=font_size),
        hovertemplate="<b>%{label}</b><br>P(class) = %{text}<extra></extra>",
        branchvalues="total",
        marker=dict(line=dict(width=2, color="white")),
        insidetextorientation="radial",
    ))

    if title:
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=20, family="Arial")),
            margin=dict(t=60, l=10, r=10, b=10),
            width=750,
            height=750,
        )
    else:
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), width=750, height=750)
    return fig


if __name__ == "__main__":
    from oracle.presets import get_model

    m = get_model("BTSv2")
    sample = {
        "Alert": 1.0,
        "Persistent": 0.73, "Transient": 0.27,
        "AGN": 0.0009, "CV": 0.728, "Varstar": 0.0013,
        "SN-Ia": 0.095, "SN-II": 0.138, "SN-Ib/c": 0.036, "SLSN": 0.0001,
    }
    fig = plot_oracle_sunburst(sample, m.taxonomy, title="Oracle BTSv2 — demo")
    fig.write_image("demo_btsv2_sunburst.png", scale=2)
    print("saved: demo_btsv2_sunburst.png")
