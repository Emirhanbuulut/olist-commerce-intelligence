"""Build analysis-ready Olist tables from the nine raw CSV files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR


FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    missing = [name for name in FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {', '.join(missing)}")
    return {key: pd.read_csv(raw_dir / filename) for key, filename in FILES.items()}


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance; missing coordinates stay missing."""
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def _primary_value(df: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
    counts = (
        df.dropna(subset=[value_col])
        .groupby(["order_id", value_col], as_index=False)
        .size()
        .sort_values(["order_id", "size", value_col], ascending=[True, False, True])
        .drop_duplicates("order_id")
    )
    return counts[["order_id", value_col]].rename(columns={value_col: output_col})


def build_order_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    for col in DATE_COLUMNS:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    geo = (
        data["geolocation"]
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(geo_lat=("geolocation_lat", "median"), geo_lng=("geolocation_lng", "median"))
    )

    customers = data["customers"].merge(
        geo.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geo_lat": "customer_lat",
                "geo_lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    sellers = data["sellers"].merge(
        geo.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geo_lat": "seller_lat",
                "geo_lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    products = data["products"].merge(data["translation"], on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    item_detail = (
        data["items"]
        .merge(
            products[["product_id", "category", "product_weight_g", "product_volume_cm3"]],
            on="product_id",
            how="left",
        )
        .merge(
            sellers[["seller_id", "seller_state", "seller_lat", "seller_lng"]],
            on="seller_id",
            how="left",
        )
        .merge(
            orders[["order_id", "customer_id"]].merge(
                customers[["customer_id", "customer_lat", "customer_lng"]],
                on="customer_id",
                how="left",
            )[["order_id", "customer_lat", "customer_lng"]],
            on="order_id",
            how="left",
        )
    )
    item_detail["distance_km"] = haversine_km(
        item_detail["customer_lat"],
        item_detail["customer_lng"],
        item_detail["seller_lat"],
        item_detail["seller_lng"],
    )

    item_agg = item_detail.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        unique_product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_volume_cm3=("product_volume_cm3", "mean"),
        avg_distance_km=("distance_km", "mean"),
    )
    item_agg = item_agg.merge(_primary_value(item_detail, "category", "primary_category"), on="order_id", how="left")
    item_agg = item_agg.merge(
        _primary_value(item_detail, "seller_state", "primary_seller_state"), on="order_id", how="left"
    )

    payments = data["payments"].copy()
    payment_agg = payments.groupby("order_id", as_index=False).agg(
        total_payment=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"),
    )
    primary_payment = (
        payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    payment_agg = payment_agg.merge(primary_payment, on="order_id", how="left")

    review_agg = data["reviews"].groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "nunique"),
    )

    result = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )
    result["purchase_date"] = result["order_purchase_timestamp"].dt.date
    result["purchase_month"] = result["order_purchase_timestamp"].dt.month
    result["purchase_year_month"] = result["order_purchase_timestamp"].dt.to_period("M").astype(str)
    result["purchase_weekday"] = result["order_purchase_timestamp"].dt.dayofweek
    result["purchase_hour"] = result["order_purchase_timestamp"].dt.hour
    result["estimated_days"] = (
        result["order_estimated_delivery_date"] - result["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    result["delivery_days"] = (
        result["order_delivered_customer_date"] - result["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    result["late_days"] = (
        result["order_delivered_customer_date"] - result["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    result["is_late"] = np.where(
        result["order_delivered_customer_date"].notna(),
        (result["late_days"] > 0).astype(int),
        np.nan,
    )
    result["freight_ratio"] = result["total_freight"] / result["total_price"].replace(0, np.nan)
    return result


def build_rfm(order_table: pd.DataFrame) -> pd.DataFrame:
    delivered = order_table.query("order_status == 'delivered'").copy()
    reference_date = delivered["order_purchase_timestamp"].max().normalize() + pd.offsets.Day(1)
    rfm = delivered.groupby("customer_unique_id", as_index=False).agg(
        recency_days=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("total_payment", "sum"),
    )
    rfm["r_score"] = pd.cut(
        rfm["recency_days"].rank(method="first", pct=True),
        bins=[0, 0.25, 0.5, 0.75, 1], labels=[4, 3, 2, 1], include_lowest=True,
    ).astype(int)
    rfm["f_score"] = pd.cut(
        rfm["frequency"].rank(method="first", pct=True),
        bins=[0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4], include_lowest=True,
    ).astype(int)
    rfm["m_score"] = pd.cut(
        rfm["monetary"].rank(method="first", pct=True),
        bins=[0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4], include_lowest=True,
    ).astype(int)

    def segment(row):
        if row.r_score >= 4 and row.f_score >= 3:
            return "Champions"
        if row.r_score >= 3 and row.f_score >= 3:
            return "Loyal"
        if row.r_score >= 3 and row.f_score <= 2:
            return "Recent"
        if row.r_score <= 2 and row.f_score >= 3:
            return "At risk"
        return "Hibernating"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def build_summary_tables(order_table: pd.DataFrame, rfm: pd.DataFrame) -> None:
    delivered = order_table.query("order_status == 'delivered'").copy()
    monthly = delivered.groupby("purchase_year_month", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("total_payment", "sum"),
        avg_order_value=("total_payment", "mean"),
        late_rate=("is_late", "mean"),
        avg_review_score=("review_score", "mean"),
    )
    category = delivered.groupby("primary_category", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("total_payment", "sum"),
        avg_order_value=("total_payment", "mean"),
        late_rate=("is_late", "mean"),
        avg_review_score=("review_score", "mean"),
    ).sort_values("revenue", ascending=False)
    states = delivered.groupby("customer_state", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("total_payment", "sum"),
        late_rate=("is_late", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_review_score=("review_score", "mean"),
    ).sort_values("revenue", ascending=False)

    cohort_source = delivered[["customer_unique_id", "order_purchase_timestamp"]].copy()
    cohort_source["order_month"] = cohort_source["order_purchase_timestamp"].dt.to_period("M")
    cohort_source["cohort_month"] = cohort_source.groupby("customer_unique_id")["order_month"].transform("min")
    cohort_source["cohort_index"] = (
        (cohort_source["order_month"].dt.year - cohort_source["cohort_month"].dt.year) * 12
        + cohort_source["order_month"].dt.month
        - cohort_source["cohort_month"].dt.month
    )
    cohort = cohort_source.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().reset_index(name="customers")
    cohort["cohort_month"] = cohort["cohort_month"].astype(str)
    cohort_sizes = cohort.query("cohort_index == 0")[["cohort_month", "customers"]].rename(columns={"customers": "cohort_size"})
    cohort = cohort.merge(cohort_sizes, on="cohort_month", how="left")
    cohort["retention_rate"] = cohort["customers"] / cohort["cohort_size"]

    monthly.to_csv(PROCESSED_DIR / "monthly_kpis.csv", index=False)
    category.to_csv(PROCESSED_DIR / "category_performance.csv", index=False)
    states.to_csv(PROCESSED_DIR / "state_performance.csv", index=False)
    cohort.to_csv(PROCESSED_DIR / "cohort_retention.csv", index=False)
    rfm.to_csv(PROCESSED_DIR / "rfm_segments.csv", index=False)


def build_quality_report(data: dict[str, pd.DataFrame], order_table: pd.DataFrame) -> dict:
    tables = {}
    for name, df in data.items():
        tables[name] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum()),
        }
    delivered = order_table.query("order_status == 'delivered'")
    report = {
        "tables": tables,
        "orders_total": int(order_table["order_id"].nunique()),
        "delivered_orders": int(delivered["order_id"].nunique()),
        "late_rate": float(delivered["is_late"].mean()),
        "date_min": str(order_table["order_purchase_timestamp"].min()),
        "date_max": str(order_table["order_purchase_timestamp"].max()),
    }
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    data = load_raw()
    orders = build_order_table(data)
    rfm = build_rfm(orders)
    orders.to_parquet(PROCESSED_DIR / "orders_enriched.parquet", index=False)
    build_summary_tables(orders, rfm)
    quality = build_quality_report(data, orders)
    print(f"Built {quality['orders_total']:,} orders; delivered late rate={quality['late_rate']:.2%}")


if __name__ == "__main__":
    main()
