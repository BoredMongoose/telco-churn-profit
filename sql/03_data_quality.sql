-- Data-quality checks. src/run_sql.py stops if any check fails.

CREATE OR REPLACE TABLE dq_results AS
SELECT 'one row per customer' AS check_name, format('{} rows, {} ids', COUNT(*), COUNT(DISTINCT customer_id)) AS detail,
       COUNT(*) = COUNT(DISTINCT customer_id) AS passed FROM customers
UNION ALL
SELECT 'blank totals only for brand-new customers',
       format('{} blank totals, {} of them with tenure 0', COUNT(*) FILTER (WHERE total_was_blank),
              COUNT(*) FILTER (WHERE total_was_blank AND tenure_months = 0)),
       COUNT(*) FILTER (WHERE total_was_blank) = COUNT(*) FILTER (WHERE total_was_blank AND tenure_months = 0)
FROM customers
UNION ALL  -- total billed should be roughly months x monthly bill (prices change, so allow a wide margin)
SELECT 'total charges consistent with tenure x monthly bill',
       format('{} customers outside 0.5x-2x', COUNT(*) FILTER (WHERE total_charges NOT BETWEEN 0.5 * tenure_months * monthly_charges
                                                                          AND 2 * tenure_months * monthly_charges)),
       COUNT(*) FILTER (WHERE total_charges NOT BETWEEN 0.5 * tenure_months * monthly_charges
                                                 AND 2 * tenure_months * monthly_charges) = 0
FROM customers WHERE tenure_months > 0
UNION ALL  -- add-ons must say "No internet service" exactly when there is no internet
SELECT 'internet add-ons consistent with internet service',
       format('{} inconsistent rows', COUNT(*) FILTER (WHERE (InternetService = 'No') <> (OnlineSecurity = 'No internet service')
              OR (InternetService = 'No') <> (TechSupport = 'No internet service')
              OR (InternetService = 'No') <> (StreamingTV = 'No internet service'))),
       COUNT(*) FILTER (WHERE (InternetService = 'No') <> (OnlineSecurity = 'No internet service')
              OR (InternetService = 'No') <> (TechSupport = 'No internet service')
              OR (InternetService = 'No') <> (StreamingTV = 'No internet service')) = 0
FROM customers
UNION ALL
SELECT 'multiple lines consistent with phone service',
       format('{} inconsistent rows', COUNT(*) FILTER (WHERE (PhoneService = 'No') <> (MultipleLines = 'No phone service'))),
       COUNT(*) FILTER (WHERE (PhoneService = 'No') <> (MultipleLines = 'No phone service')) = 0
FROM customers
UNION ALL
SELECT 'charges in a plausible range', format('monthly {}-{}', MIN(monthly_charges), MAX(monthly_charges)),
       MIN(monthly_charges) > 0 AND MAX(monthly_charges) < 500 FROM customers
UNION ALL
SELECT 'rollup total matches customer count',
       format('{} vs {}', (SELECT customers FROM mart_contract_tenure WHERE contract = 'All contracts'), (SELECT COUNT(*) FROM customers)),
       (SELECT customers FROM mart_contract_tenure WHERE contract = 'All contracts') = (SELECT COUNT(*) FROM customers);
