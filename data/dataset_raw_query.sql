WITH unicos AS (
    SELECT DISTINCT
        d.Grupo_raiz
    FROM BRZ_SLV_Lakegistica.[erp].[fact_dim_production_rates_followup] f
    INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
        ON f.Referencia = d.Material
        AND d.Centro = 'ES00'
    WHERE f.Werks = 'ES00'
),

ritmos AS (
    SELECT
        f.DateRegister,
        d.Grupo_raiz,

        -- Variables absolutas
        f.FabricacionSemana,
        f.RitmoSemana,
        f.ExpedidasSemana,
        f.Demanda,
        f.StockFin,
        f.StockReal,
        f.AcumFab,
        f.AcumRitmo,
        f.AcumEnvio,
        f.AcumDemanda,
        f.EntregasPrev,
        f.DemandActual,

        -- Ratios / cumplimientos
        f.CumpFabRit,
        f.CumpExpDem,
        f.CumpStock,
        f.PorcenCump,
        f.PorcenCumpto,
        f.CumpTotal

    FROM BRZ_SLV_Lakegistica.[erp].[fact_dim_production_rates_followup] f
    INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
        ON f.Referencia = d.Material
        AND d.Centro = 'ES00'

    WHERE f.Werks = 'ES00'
      AND d.Grupo_raiz <> 'GENERICO'
)

SELECT
    CAST(fact.Fecha AS DATE) AS Fecha,
    d.Grupo_raiz,

    -- Producción
    r.FabricacionSemana,
    r.RitmoSemana,
    r.CumpFabRit,

    -- Expediciones / demanda
    r.ExpedidasSemana,
    r.Demanda,
    r.CumpExpDem,

    -- Stock
    r.StockFin,
    r.StockReal,
    r.CumpStock,

    -- Producción acumulada
    r.AcumFab,
    r.AcumRitmo,
    r.PorcenCump,

    -- Expedición acumulada
    r.AcumEnvio,
    r.AcumDemanda,
    r.PorcenCumpto,

    -- Cobertura / cumplimiento total
    r.EntregasPrev,
    r.DemandActual,
    r.CumpTotal,

    -- Información logística
    MAX(fact.Id_ClienteBaan) AS customer,
    MAX(fact.Id_UAT) AS uat,
    MAX(fact.Direccion_Entrega) AS destination,

    -- Resultado de entrega
    SUM(fact.Cantidad_Elementos_Perdidos) AS lost,
    SUM(fact.Cantidad_Elementos_Enviados) AS delivered

FROM [WH_P3-2].[dbo].[Fact_OTD] fact

INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
    ON d.Material = fact.Id_item
    AND d.Centro = 'ES00'

INNER JOIN unicos u
    ON u.Grupo_raiz = d.Grupo_raiz

INNER JOIN ritmos r
    ON CAST(r.DateRegister AS DATE) = CAST(fact.Fecha AS DATE)
    AND r.Grupo_raiz = d.Grupo_raiz

WHERE fact.Id_Planta = 1
  AND fact.Fecha > '2025-01-01'
  AND fact.Id_UAT <> '999'
  AND d.GrupoEstratgPlanif = 'ZA'

-- Para probar primero con un solo grupo:
-- AND d.Grupo_raiz = 'B015M'

GROUP BY
    CAST(fact.Fecha AS DATE),
    d.Grupo_raiz,

    r.FabricacionSemana,
    r.RitmoSemana,
    r.CumpFabRit,

    r.ExpedidasSemana,
    r.Demanda,
    r.CumpExpDem,

    r.StockFin,
    r.StockReal,
    r.CumpStock,

    r.AcumFab,
    r.AcumRitmo,
    r.PorcenCump,

    r.AcumEnvio,
    r.AcumDemanda,
    r.PorcenCumpto,

    r.EntregasPrev,
    r.DemandActual,
    r.CumpTotal

ORDER BY
    Fecha,
    d.Grupo_raiz;