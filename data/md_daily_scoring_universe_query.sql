/*
Conceptual view for the daily Missed Delivery scoring universe.

IMPORTANT:
- Replace SOURCE_DATABASE.dbo.ACTIVE_DELIVERY_OBLIGATIONS with the governed
  ERP/planning source that exists before the delivery outcome.
- Replace the conceptual status codes with the real business codes.
- Do not replace the source with Fact_OTD or any outcome table containing
  lost/delivered.
*/

CREATE OR ALTER VIEW [analytics].[vw_md_daily_scoring_universe]
AS
WITH active_obligations AS (
    SELECT
        o.obligation_id,
        CAST(o.promised_delivery_date AS DATE) AS target_date,
        o.material_id,
        CAST(o.customer_id AS VARCHAR(100)) AS customer,
        CAST(o.uat_id AS VARCHAR(100)) AS uat,
        CAST(o.destination_id AS VARCHAR(100)) AS destination,
        o.source_update_timestamp
    FROM [SOURCE_DATABASE].[dbo].[ACTIVE_DELIVERY_OBLIGATIONS] o
    WHERE o.status_code IN (
        'OPEN',
        'RELEASED',
        'CONFIRMED',
        'PARTIALLY_FULFILLED'
    )
      AND COALESCE(o.remaining_quantity, 0) > 0
      AND COALESCE(o.is_cancelled, 0) = 0
      AND o.promised_delivery_date IS NOT NULL
),

mapped_obligations AS (
    SELECT
        a.obligation_id,
        a.target_date,
        d.Grupo_raiz,
        a.customer,
        a.uat,
        a.destination,
        a.source_update_timestamp
    FROM active_obligations a
    INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
        ON d.Material = a.material_id
       AND d.Centro = 'ES00'
    WHERE d.Grupo_raiz IS NOT NULL
      AND d.Grupo_raiz <> 'GENERICO'
)

SELECT
    target_date,
    Grupo_raiz,
    MAX(customer) AS customer,
    MAX(uat) AS uat,
    MAX(destination) AS destination,
    DATEADD(DAY, -7, target_date) AS scoring_eligibility_date,
    COUNT_BIG(*) AS obligation_count,
    MAX(source_update_timestamp) AS source_last_update_timestamp
FROM mapped_obligations
GROUP BY
    target_date,
    Grupo_raiz;

/*
Daily pipeline filter (parameter supplied by orchestration):

SELECT *
FROM [analytics].[vw_md_daily_scoring_universe]
WHERE scoring_eligibility_date = @scoring_date;
*/
