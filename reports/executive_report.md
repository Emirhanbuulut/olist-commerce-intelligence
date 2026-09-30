# Executive report

## Business snapshot

- Analysis period: 2016-09-04 to 2018-10-17
- Total orders: 99,441
- Delivered orders: 96,478
- Delivered revenue: R$ 15,422,462
- Average order value: R$ 159.86
- Late-delivery rate: 8.1%
- Repeat-customer rate: 3.0%

## Findings

1. **Delivery reliability and satisfaction move together.** On-time orders average 4.29/5, while late orders average 2.57/5. This is an association, not a causal estimate.
2. **health_beauty is the largest category by revenue**, with R$ 1,413,963 across 8,621 orders.
3. **SP is the largest customer state**, producing R$ 5,770,266 from 40,501 delivered orders.
4. Customer repurchase is limited in this dataset. RFM segments should therefore support descriptive targeting rather than be presented as validated campaign lift.

## Late-delivery model

- Selected model: Logistic Regression
- Chronological test rows: 19,294
- Test PR-AUC: 0.117
- Test ROC-AUC: 0.721
- Test precision: 0.104
- Test recall: 0.450
- Test F1: 0.169
- Decision threshold selected on the validation period: 0.62

The model uses only information available at purchase or checkout. Delivery timestamps, lateness outcomes and review scores are excluded from its features. Performance is measured on the newest 20% of orders to better reflect future use.

## Recommended actions

- Route high-risk orders to proactive delivery monitoring before the promised date.
- Review category-state lanes with both high order volume and above-average delay rates.
- Compare seller performance within similar distance and product profiles before changing commercial terms.
- Run a controlled experiment before claiming that proactive notifications improve satisfaction.

## Limits

The data covers an anonymized Brazilian marketplace from 2016–2018. It does not include inventory, carrier identity, marketing exposure, margin or intervention outcomes. Model scores rank operational risk; they do not prove why an order will be late.
