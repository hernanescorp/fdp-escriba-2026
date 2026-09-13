WITH cte AS (
    SELECT
        f.DateKey,
        d.Grupo_raiz,
        MAX(f.Direccion_Entrega) AS Direccion_Entrega,
        MAX(f.Id_UAT) AS Id_UAT,
        SUM(f.Cantidad_Elementos_Perdidos) AS Cantidad_Elementos_Perdidos,
        SUM(f.Cantidad_Elementos_Enviados) AS Cantidad_Elementos_Enviados,
        SUM(f.Cantidad_Elementos_Enviados)
            + SUM(f.Cantidad_Elementos_Perdidos) AS [required]
    FROM Logistica_DW.gld_erp.Fact_OTD f
    INNER JOIN Logistica_DW.gld_erp.dim_material d
        ON f.MaterialKey = d.MaterialKey
    WHERE
        f.Centro = 'ES00'
        AND d.GrupoEstratgPlanif = 'ZA'
        AND f.DateKey = 20260909
    GROUP BY
        f.DateKey,
        d.Grupo_raiz
),
predicciones AS (
    SELECT
        *,
        1 AS Tiene_prediccion
    FROM BRZ_SLV_Lakegistica.dbo.md_predictions
    WHERE scoring_date = '2026-09-02'
)
SELECT
    p.scoring_date,
    p.target_date,
    COALESCE(p.Grupo_raiz, c.Grupo_raiz) AS Grupo_raiz,
    p.raw_score,
    p.calibrated_probability,
    p.alert_flag,
    
    c.Cantidad_Elementos_Perdidos,
    CASE
        WHEN p.Tiene_prediccion IS NULL THEN 'Sin predicción'
        WHEN c.DateKey IS NULL THEN 'Sin coincidencia en CTE'
        ELSE 'Coincide'
    END AS Estado
FROM predicciones p
FULL OUTER JOIN cte c
    ON c.Grupo_raiz = p.Grupo_raiz
WHERE
    p.Tiene_prediccion = 1
    OR c.Cantidad_Elementos_Perdidos > 0
ORDER BY Grupo_raiz DESC, c.Cantidad_Elementos_Enviados DESC;