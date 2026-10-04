-- Churn by segment, and the revenue it takes with it.

-- 1. Contract x tenure, with subtotals for each contract and an overall total (ROLLUP) --------------
CREATE OR REPLACE TABLE mart_contract_tenure AS
SELECT COALESCE(Contract, 'All contracts')            AS contract,
       COALESCE(tenure_band, 'All tenures')           AS tenure_band,
       MIN(tenure_band_order)                         AS band_order,
       COUNT(*)                                       AS customers,
       SUM(churned::INT)                              AS churned,
       ROUND(100 * AVG(churned::INT), 1)              AS churn_rate_pct
FROM customers
GROUP BY ROLLUP (Contract, tenure_band)
ORDER BY contract, band_order;

-- 2. One-variable churn rates, all in one long table ---------------------------------------------
CREATE OR REPLACE TABLE mart_churn_by_feature AS
WITH long AS (
    SELECT 'Contract' AS feature, Contract AS value, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Internet service', InternetService, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Payment method', PaymentMethod, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Tech support', TechSupport, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Online security', OnlineSecurity, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Paperless billing', CASE WHEN paperless_billing THEN 'Yes' ELSE 'No' END, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Senior citizen', CASE WHEN senior THEN 'Yes' ELSE 'No' END, churned, monthly_charges FROM customers
    UNION ALL SELECT 'Tenure', tenure_band, churned, monthly_charges FROM customers
)
SELECT feature, value,
       COUNT(*)                                                 AS customers,
       ROUND(100 * AVG(churned::INT), 1)                        AS churn_rate_pct,
       ROUND(AVG(monthly_charges), 2)                           AS avg_monthly_charges,
       ROUND(100 * AVG(churned::INT) / (SELECT AVG(churned::INT) FROM customers), 0) AS index_vs_average
FROM long
GROUP BY feature, value
ORDER BY feature, churn_rate_pct DESC;

-- 3. Revenue walking out the door: monthly bills of customers who left, by contract --------------
CREATE OR REPLACE TABLE mart_revenue_at_risk AS
SELECT Contract                                                         AS contract,
       COUNT(*)                                                         AS customers,
       ROUND(SUM(monthly_charges), 0)                                   AS monthly_revenue,
       ROUND(SUM(monthly_charges) FILTER (WHERE churned), 0)            AS monthly_revenue_lost,
       ROUND(100 * SUM(monthly_charges) FILTER (WHERE churned) / SUM(monthly_charges), 1) AS pct_revenue_lost,
       ROUND(100 * SUM(monthly_charges) FILTER (WHERE churned)
             / SUM(SUM(monthly_charges) FILTER (WHERE churned)) OVER (), 1)                AS share_of_all_lost_revenue
FROM customers
GROUP BY Contract
ORDER BY monthly_revenue_lost DESC;
