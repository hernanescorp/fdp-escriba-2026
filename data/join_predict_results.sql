WITH cte AS (
SELECT TOP (1000) [DateKey]
      ,[d].Grupo_raiz
      ,MAX(f.[Direccion_Entrega]) AS [Direccion_Entrega]
      ,MAX(f.[Id_UAT]) AS [Id_UAT]
      ,SUM(f.[Cantidad_Elementos_Perdidos]) AS [Cantidad_Elementos_Perdidos]
      ,SUM(f.[Cantidad_Elementos_Enviados]) AS [Cantidad_Elementos_Enviados]
      ,SUM(f.Cantidad_Elementos_Enviados) + SUM(f.Cantidad_Elementos_Perdidos) AS [required]

      
  FROM Logistica_DW.[gld_erp].[Fact_OTD] f
  INNER JOIN Logistica_DW.[gld_erp].[dim_material] d
    ON f.[MaterialKey] = d.[MaterialKey]
  WHERE 
    f.[Centro] = 'ES00'  
    AND d.GrupoEstratgPlanif = 'ZA'
    --AND d.Grupo_raiz = 'B015M'
    AND f.DateKey = 20260902
  GROUP BY [DateKey], [d].Grupo_raiz)


SELECT TOP (1000) [scoring_date]
      ,[target_date]
      ,[request]
      ,p.[Grupo_raiz]
      ,[raw_score]
      ,[calibrated_probability]
      ,[alert_flag]
      ,c.Cantidad_Elementos_Perdidos
  FROM BRZ_SLV_Lakegistica.[dbo].[md_predictions] p
    LEFT JOIN cte c
    ON c.Grupo_raiz = p.Grupo_raiz
  WHERE scoring_date = '2026-08-26' --AND alert_flag = 1
  