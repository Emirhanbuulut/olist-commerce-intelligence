"""Turn computed outputs into a concise, reproducible executive report."""

from __future__ import annotations

import json

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR


def pct(value):
    return f"{value:.1%}"


def main() -> None:
    orders = pd.read_parquet(PROCESSED_DIR / "orders_enriched.parquet")
    delivered = orders.query("order_status == 'delivered'")
    categories = pd.read_csv(PROCESSED_DIR / "category_performance.csv")
    states = pd.read_csv(PROCESSED_DIR / "state_performance.csv")
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_segments.csv")
    metrics = json.loads((REPORTS_DIR / "model_metrics.json").read_text(encoding="utf-8"))

    on_time_review = delivered.loc[delivered.is_late == 0, "review_score"].mean()
    late_review = delivered.loc[delivered.is_late == 1, "review_score"].mean()
    top_category = categories.iloc[0]
    top_state = states.iloc[0]
    repeat_rate = (rfm["frequency"] > 1).mean()

    report = f"""# Executive report

## Business snapshot

- Analysis period: {orders.order_purchase_timestamp.min():%Y-%m-%d} to {orders.order_purchase_timestamp.max():%Y-%m-%d}
- Total orders: {orders.order_id.nunique():,}
- Delivered orders: {delivered.order_id.nunique():,}
- Delivered revenue: R$ {delivered.total_payment.sum():,.0f}
- Average order value: R$ {delivered.total_payment.mean():,.2f}
- Late-delivery rate: {pct(delivered.is_late.mean())}
- Repeat-customer rate: {pct(repeat_rate)}

## Findings

1. **Delivery reliability and satisfaction move together.** On-time orders average {on_time_review:.2f}/5, while late orders average {late_review:.2f}/5. This is an association, not a causal estimate.
2. **{top_category.primary_category} is the largest category by revenue**, with R$ {top_category.revenue:,.0f} across {int(top_category.orders):,} orders.
3. **{top_state.customer_state} is the largest customer state**, producing R$ {top_state.revenue:,.0f} from {int(top_state.orders):,} delivered orders.
4. Customer repurchase is limited in this dataset. RFM segments should therefore support descriptive targeting rather than be presented as validated campaign lift.

## Late-delivery model

- Selected model: {metrics['selected_model'].replace('_', ' ').title()}
- Chronological test rows: {metrics['test_rows']:,}
- Test PR-AUC: {metrics['test']['pr_auc']:.3f}
- Test ROC-AUC: {metrics['test']['roc_auc']:.3f}
- Test precision: {metrics['test']['precision']:.3f}
- Test recall: {metrics['test']['recall']:.3f}
- Test F1: {metrics['test']['f1']:.3f}
- Decision threshold selected on the validation period: {metrics['selected_threshold']:.2f}

The model uses only information available at purchase or checkout. Delivery timestamps, lateness outcomes and review scores are excluded from its features. Performance is measured on the newest 20% of orders to better reflect future use.

## Recommended actions

- Route high-risk orders to proactive delivery monitoring before the promised date.
- Review category-state lanes with both high order volume and above-average delay rates.
- Compare seller performance within similar distance and product profiles before changing commercial terms.
- Run a controlled experiment before claiming that proactive notifications improve satisfaction.

## Limits

The data covers an anonymized Brazilian marketplace from 2016–2018. It does not include inventory, carrier identity, marketing exposure, margin or intervention outcomes. Model scores rank operational risk; they do not prove why an order will be late.
"""
    (REPORTS_DIR / "executive_report.md").write_text(report, encoding="utf-8")
    print("Wrote reports/executive_report.md")


if __name__ == "__main__":
    main()

