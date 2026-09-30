from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_model import FEATURES  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Olist Commerce Intelligence", page_icon="📦", layout="wide")


@st.cache_data
def load_data():
    return {
        "orders": pd.read_parquet(PROCESSED / "orders_enriched.parquet"),
        "monthly": pd.read_csv(PROCESSED / "monthly_kpis.csv"),
        "category": pd.read_csv(PROCESSED / "category_performance.csv"),
        "states": pd.read_csv(PROCESSED / "state_performance.csv"),
        "rfm": pd.read_csv(PROCESSED / "rfm_segments.csv"),
        "importance": pd.read_csv(PROCESSED / "feature_importance.csv"),
        "metrics": json.loads((REPORTS / "model_metrics.json").read_text()),
    }


data = load_data()
orders = data["orders"]
delivered = orders.query("order_status == 'delivered'").copy()

st.title("Olist Commerce Intelligence")
st.caption("100K e-commerce orders · Business analytics · Customer segmentation · Late-delivery prediction")

with st.sidebar:
    st.header("Filters")
    states = sorted(delivered.customer_state.dropna().unique())
    selected_states = st.multiselect("Customer state", states, default=states)
    date_min = delivered.order_purchase_timestamp.min().date()
    date_max = delivered.order_purchase_timestamp.max().date()
    selected_dates = st.date_input("Purchase date", value=(date_min, date_max), min_value=date_min, max_value=date_max)

filtered = delivered[delivered.customer_state.isin(selected_states)]
if len(selected_dates) == 2:
    filtered = filtered[
        filtered.order_purchase_timestamp.dt.date.between(selected_dates[0], selected_dates[1])
    ]

overview, operations, customers, model_tab = st.tabs(
    ["Executive overview", "Delivery operations", "Customers", "Predictive model"]
)

with overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delivered orders", f"{filtered.order_id.nunique():,}")
    c2.metric("Revenue", f"R$ {filtered.total_payment.sum():,.0f}")
    c3.metric("Average order value", f"R$ {filtered.total_payment.mean():,.2f}")
    c4.metric("Late delivery rate", f"{filtered.is_late.mean():.1%}")

    monthly = filtered.groupby("purchase_year_month", as_index=False).agg(
        orders=("order_id", "nunique"), revenue=("total_payment", "sum"), late_rate=("is_late", "mean")
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.line(monthly, x="purchase_year_month", y="revenue", markers=True, title="Monthly revenue"),
        use_container_width=True,
    )
    top_categories = (
        filtered.groupby("primary_category", as_index=False)["total_payment"].sum()
        .nlargest(12, "total_payment")
        .sort_values("total_payment")
    )
    right.plotly_chart(
        px.bar(top_categories, x="total_payment", y="primary_category", orientation="h", title="Top categories by revenue"),
        use_container_width=True,
    )

with operations:
    state_ops = filtered.groupby("customer_state", as_index=False).agg(
        orders=("order_id", "nunique"), late_rate=("is_late", "mean"), delivery_days=("delivery_days", "mean")
    )
    state_ops["late_rate_pct"] = 100 * state_ops["late_rate"]
    st.plotly_chart(
        px.scatter(
            state_ops, x="delivery_days", y="late_rate_pct", size="orders", color="customer_state",
            hover_name="customer_state", title="State delivery performance", labels={"delivery_days": "Average delivery days", "late_rate_pct": "Late orders (%)"},
        ), use_container_width=True,
    )
    review_relation = filtered.dropna(subset=["review_score"]).assign(
        delivery_result=lambda x: x.is_late.map({0.0: "On time", 1.0: "Late"})
    )
    st.plotly_chart(
        px.histogram(
            review_relation, x="review_score", color="delivery_result", barmode="group", histnorm="percent",
            title="Review distribution by delivery result",
        ), use_container_width=True,
    )

with customers:
    segments = data["rfm"].groupby("segment", as_index=False).agg(
        customers=("customer_unique_id", "nunique"), monetary=("monetary", "sum"), avg_recency=("recency_days", "mean")
    )
    left, right = st.columns(2)
    left.plotly_chart(px.bar(segments, x="segment", y="customers", color="segment", title="RFM customer segments"), use_container_width=True)
    right.plotly_chart(px.treemap(segments, path=["segment"], values="monetary", title="Revenue represented by segment"), use_container_width=True)
    st.info("Most Olist customers appear once in the observation window. Treat these RFM labels as descriptive prioritization, not validated campaign outcomes.")

with model_tab:
    metrics = data["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", f"{metrics['test']['pr_auc']:.3f}")
    c2.metric("ROC-AUC", f"{metrics['test']['roc_auc']:.3f}")
    c3.metric("Recall", f"{metrics['test']['recall']:.1%}")
    c4.metric("Precision", f"{metrics['test']['precision']:.1%}")
    importance = data["importance"].head(15).sort_values("importance")
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Top model signals"), use_container_width=True)

    st.subheader("Late-delivery risk simulator")
    st.caption("Scenario tool for demonstration; this is not a production dispatch system.")
    defaults = delivered[FEATURES].copy()
    with st.form("risk_form"):
        a, b, c = st.columns(3)
        scenario = {
            "item_count": a.number_input("Item count", 1, 20, int(defaults.item_count.median())),
            "unique_product_count": a.number_input("Unique products", 1, 20, int(defaults.unique_product_count.median())),
            "seller_count": a.number_input("Seller count", 1, 10, int(defaults.seller_count.median())),
            "total_price": a.number_input("Product total (R$)", 0.0, 10000.0, float(defaults.total_price.median())),
            "total_freight": a.number_input("Freight (R$)", 0.0, 1000.0, float(defaults.total_freight.median())),
            "avg_product_weight_g": b.number_input("Average weight (g)", 0.0, 50000.0, float(defaults.avg_product_weight_g.median())),
            "avg_product_volume_cm3": b.number_input("Average volume (cm³)", 0.0, 1000000.0, float(defaults.avg_product_volume_cm3.median())),
            "avg_distance_km": b.number_input("Seller-customer distance (km)", 0.0, 5000.0, float(defaults.avg_distance_km.median())),
            "payment_installments": b.number_input("Installments", 0, 24, int(defaults.payment_installments.median())),
            "estimated_days": b.number_input("Promised lead time (days)", 1.0, 90.0, float(defaults.estimated_days.median())),
            "purchase_month": c.slider("Purchase month", 1, 12, int(defaults.purchase_month.median())),
            "purchase_weekday": c.slider("Weekday (0=Mon)", 0, 6, int(defaults.purchase_weekday.median())),
            "purchase_hour": c.slider("Purchase hour", 0, 23, int(defaults.purchase_hour.median())),
            "customer_state": c.selectbox("Customer state", sorted(defaults.customer_state.dropna().unique())),
            "primary_category": c.selectbox("Category", sorted(defaults.primary_category.dropna().unique())),
            "primary_seller_state": c.selectbox("Seller state", sorted(defaults.primary_seller_state.dropna().unique())),
            "primary_payment_type": c.selectbox("Payment type", sorted(defaults.primary_payment_type.dropna().unique())),
        }
        submitted = st.form_submit_button("Estimate risk")
    if submitted:
        model = joblib.load(ARTIFACTS / "late_delivery_model.joblib")
        probability = model.predict_proba(pd.DataFrame([scenario])[FEATURES])[:, 1][0]
        st.metric("Estimated late-delivery probability", f"{probability:.1%}")
        st.progress(float(probability))

