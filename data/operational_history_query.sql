WITH target_groups AS (

    SELECT DISTINCT
        d.Grupo_raiz

    FROM [WH_P3-2].[dbo].[Fact_OTD] fact

    INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
        ON d.Material = fact.Id_item
        AND d.Centro = 'ES00'

    WHERE fact.Id_Planta = 1
      AND fact.Fecha > '2025-01-01'
      AND fact.Id_UAT <> '999'
      AND d.GrupoEstratgPlanif = 'ZA'
      AND d.Grupo_raiz <> 'GENERICO'
)

SELECT
    CAST(f.DateRegister AS DATE) AS Fecha,
    d.Grupo_raiz,

    f.FabricacionSemana,
    f.RitmoSemana,
    f.CumpFabRit,

    f.ExpedidasSemana,
    f.Demanda,
    f.CumpExpDem,

    f.StockFin,
    f.StockReal,
    f.CumpStock,

    f.AcumFab,
    f.AcumRitmo,
    f.PorcenCump,

    f.AcumEnvio,
    f.AcumDemanda,
    f.PorcenCumpto,

    f.EntregasPrev,
    f.DemandActual,
    f.CumpTotal

FROM BRZ_SLV_Lakegistica.[erp].[fact_dim_production_rates_followup] f

INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
    ON f.Referencia = d.Material
    AND d.Centro = 'ES00'

INNER JOIN target_groups tg
    ON tg.Grupo_raiz = d.Grupo_raiz

WHERE f.Werks = 'ES00'
  AND d.Grupo_raiz <> 'GENERICO'

ORDER BY
    d.Grupo_raiz,
    CAST(f.DateRegister AS DATE);