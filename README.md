# TFM IA

## Previsión del precio del mercado diario e intradiario (España) con ML

Proyecto del TFM orientado a predecir el precio day-ahead (DAM) y el spread con el intradiario (IDM) en el mercado eléctrico español. Integra modelos de ML/DL, variables exógenas (clima, demanda, festivos, renovables) y una estrategia simple de asignación de volumen para evaluar impacto económico.

## Qué hace

- Predice precio horario DAM y spread DAM-IDM.

- Compara modelos: XGBoost, LightGBM, CatBoost, RidgeCV, Random Forest, LSTM y stacking.

- Usa features exógenas: meteorología (AEMET), demanda/generación (REE/ESIOS), festivos ponderados por población (INE + BOE).

- Incluye backtesting (desde 2018) y métricas (MAE, RMSE, R², ROC/AUC en clasificación de spread).
