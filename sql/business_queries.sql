-- DuckDB-compatible portfolio queries.
-- Run after: CREATE VIEW orders AS SELECT * FROM 'data/processed/orders_enriched.parquet';

-- 1. Executive monthly KPIs
SELECT
    purchase_year_month,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(total_payment), 2) AS revenue,
    ROUND(AVG(total_payment), 2) AS average_order_value,
    ROUND(100 * AVG(is_late), 2) AS late_delivery_pct,
    ROUND(AVG(review_score), 2) AS average_review_score
FROM orders
WHERE order_status = 'delivered'
GROUP BY purchase_year_month
ORDER BY purchase_year_month;

-- 2. Categories with enough volume and a high late-delivery rate
SELECT
    primary_category,
    COUNT(*) AS delivered_orders,
    ROUND(SUM(total_payment), 2) AS revenue,
    ROUND(100 * AVG(is_late), 2) AS late_delivery_pct,
    ROUND(AVG(review_score), 2) AS average_review_score
FROM orders
WHERE order_status = 'delivered'
GROUP BY primary_category
HAVING COUNT(*) >= 500
ORDER BY late_delivery_pct DESC, delivered_orders DESC;

-- 3. Geographic operations matrix
SELECT
    customer_state,
    primary_seller_state,
    COUNT(*) AS orders,
    ROUND(AVG(avg_distance_km), 0) AS average_distance_km,
    ROUND(AVG(delivery_days), 1) AS average_delivery_days,
    ROUND(100 * AVG(is_late), 2) AS late_delivery_pct
FROM orders
WHERE order_status = 'delivered'
GROUP BY customer_state, primary_seller_state
HAVING COUNT(*) >= 100
ORDER BY late_delivery_pct DESC, orders DESC;

-- 4. Association between lateness and satisfaction
SELECT
    CASE WHEN is_late = 1 THEN 'Late' ELSE 'On time' END AS delivery_result,
    COUNT(*) AS orders,
    ROUND(AVG(review_score), 2) AS average_review_score,
    ROUND(100 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 2) AS low_review_pct
FROM orders
WHERE order_status = 'delivered' AND review_score IS NOT NULL
GROUP BY delivery_result;

-- 5. Repeat customer distribution
WITH customer_orders AS (
    SELECT customer_unique_id, COUNT(DISTINCT order_id) AS order_count
    FROM orders
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
)
SELECT
    CASE WHEN order_count = 1 THEN 'One order' ELSE 'Repeat customer' END AS customer_type,
    COUNT(*) AS customers,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS customer_pct
FROM customer_orders
GROUP BY customer_type;

