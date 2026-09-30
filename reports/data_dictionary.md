# Analysis data dictionary

The central processed table is `data/processed/orders_enriched.parquet`. Its grain is one marketplace order.

| Field | Meaning | Available at checkout? |
|---|---|---|
| `order_id` | Anonymous order identifier | Yes |
| `customer_unique_id` | Stable anonymous customer identifier | Yes |
| `customer_state` | Customer delivery state | Yes |
| `item_count` | Number of order lines | Yes |
| `unique_product_count` | Unique products in the order | Yes |
| `seller_count` | Unique sellers in the order | Yes |
| `total_price` | Sum of product prices | Yes |
| `total_freight` | Freight charged | Yes |
| `avg_product_weight_g` | Mean product weight | Yes |
| `avg_product_volume_cm3` | Mean product volume | Yes |
| `avg_distance_km` | Mean great-circle distance between seller and customer ZIP centroids | Yes |
| `primary_category` | Most frequent product category in the order | Yes |
| `primary_seller_state` | State of the primary seller | Yes |
| `primary_payment_type` | Payment method with the largest value | Yes |
| `payment_installments` | Maximum installment count | Yes |
| `estimated_days` | Promised date minus purchase timestamp | Yes |
| `delivery_days` | Actual delivery duration | No; outcome |
| `late_days` | Actual date minus promised date | No; outcome |
| `is_late` | 1 when actual delivery is after the promised date | No; target |
| `review_score` | Post-delivery customer rating | No; outcome |

Geographic distance uses median latitude and longitude for each ZIP prefix because the source geolocation table contains repeated coordinates. It is an approximation, not a road-routing distance.

