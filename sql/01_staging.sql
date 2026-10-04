-- Load IBM's Telco customer file and add the fields the analysis uses.

CREATE OR REPLACE TABLE customers AS
SELECT
    customerID                                                     AS customer_id,
    gender, SeniorCitizen = 1 AS senior, Partner = 'Yes' AS partner, Dependents = 'Yes' AS dependents,
    tenure                                                         AS tenure_months,
    PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling = 'Yes' AS paperless_billing, PaymentMethod,
    MonthlyCharges                                                 AS monthly_charges,
    -- 11 brand-new customers (tenure 0) have a blank total: they haven't been billed yet
    COALESCE(TRY_CAST(NULLIF(trim(TotalCharges), '') AS DOUBLE), 0) AS total_charges,
    trim(TotalCharges) = ''                                         AS total_was_blank,
    Churn = 'Yes'                                                   AS churned,
    CASE WHEN tenure <= 6 THEN '0-6 months' WHEN tenure <= 12 THEN '7-12 months'
         WHEN tenure <= 24 THEN '1-2 years' WHEN tenure <= 48 THEN '2-4 years' ELSE '4-6 years' END AS tenure_band,
    CASE WHEN tenure <= 6 THEN 1 WHEN tenure <= 12 THEN 2 WHEN tenure <= 24 THEN 3 WHEN tenure <= 48 THEN 4 ELSE 5 END
                                                                    AS tenure_band_order,
    (OnlineSecurity = 'Yes')::INT + (OnlineBackup = 'Yes')::INT + (DeviceProtection = 'Yes')::INT
      + (TechSupport = 'Yes')::INT                                  AS protection_addons
FROM read_csv('data/raw/Telco-Customer-Churn.csv', header = true, all_varchar = false,
              types = {'TotalCharges': 'VARCHAR', 'SeniorCitizen': 'INTEGER'});
