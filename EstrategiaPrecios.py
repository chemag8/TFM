import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
from pathlib import Path

# =============================
# Configuración / Parámetros
# =============================
RUTA_DATASET_PRED = Path("dataset_con_pred.csv")
RUTA_DATASET_PRECIOS = Path("dataset_modelo_spread.csv")
RUTA_MODELO = Path("lstm_spread.h5")
RUTA_SCALER_X = Path("scaler_x.joblib")
RUTA_SCALER_Y = Path("scaler_y.joblib")

FECHA_MIN = "2023-01-01"
TARGET = "spread"
TIME_STEPS = 24

capital_inicial = 10000
max_mwh_por_hora = 2      # capacidad operativa máxima por hora

# =============================
# Funciones auxiliares
# =============================

def create_sequences(X: np.ndarray, time_steps: int = 24) -> np.ndarray:
    """Convierte una matriz 2D (t, features) en secuencias 3D (t - time_steps, time_steps, features)."""
    Xs = []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps : i])
    return np.array(Xs)

def plot_capital_y_precios(df_res: pd.DataFrame, capital_inicial: float) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Eje 1: capital acumulado
    ax1.plot(df_res["datetime"], df_res["capital"], label="Capital acumulado", color="black", linewidth=2)
    ax1.scatter(
        df_res[df_res["resultado"] == "ganancia"]["datetime"],
        df_res[df_res["resultado"] == "ganancia"]["capital"],
        color="green", label="Ganancia", s=10
    )
    ax1.scatter(
        df_res[df_res["resultado"] == "perdida"]["datetime"],
        df_res[df_res["resultado"] == "perdida"]["capital"],
        color="red", label="Pérdida", s=10
    )
    ax1.axhline(capital_inicial, color="gray", linestyle="--", label="Capital inicial")
    ax1.set_ylabel("Capital (€)")
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} €'))
    ax1.set_xlabel("Fecha")
    ax1.grid(True)

    # Eje 2: precios
    ax2 = ax1.twinx()
    ax2.plot(df_res["datetime"], df_res["precio_diario"], label="Precio diario", color="blue", alpha=0.5)
    ax2.plot(df_res["datetime"], df_res["precio_intra"], label="Precio intra", color="orange", alpha=0.5)
    ax2.set_ylabel("Precio de la electricidad (€/MWh)")

    # Leyendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Capital acumulado y evolución de precios diario vs intra")
    plt.tight_layout()
    plt.show()

def plot_capital_y_precios_subplots(df_res: pd.DataFrame, capital_inicial: float) -> None:
    fig, (ax_cap, ax_pre) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Capital acumulado
    ax_cap.plot(df_res["datetime"], df_res["capital"], label="Capital acumulado", linewidth=2)
    ax_cap.axhline(capital_inicial, linestyle="--", label="Capital inicial")
    ax_cap.set_ylabel("Capital (€)")
    ax_cap.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} €'))
    ax_cap.legend(loc="upper left")
    ax_cap.grid(True)
    ax_cap.set_title("Evolución del capital")

    # Subplot 2: Precios
    ax_pre.plot(df_res["datetime"], df_res["precio_diario"], label="Precio diario", alpha=0.7, linewidth=0.3)
    ax_pre.plot(df_res["datetime"], df_res["precio_intra"], label="Precio intradiario", alpha=0.7, linewidth=0.3)
    ax_pre.set_ylabel("Precio (€/MWh)")
    ax_pre.set_xlabel("Fecha")
    ax_pre.legend(loc="upper left")
    ax_pre.grid(True)
    ax_pre.set_title("Evolución de precios diario vs intra")

    plt.tight_layout()
    plt.show()



# =============================
# 1) Cargar datos y modelos
# =============================

df = pd.read_csv(RUTA_DATASET_PRED, parse_dates=["datetime"])  # dataset con target real (spread) y features

# Filtrar desde fecha mínima
df = df[df["datetime"] >= FECHA_MIN].copy()

# Cargar scalers y modelo ANTES de usar sus metadatos
scaler_x = joblib.load(RUTA_SCALER_X)
scaler_y = joblib.load(RUTA_SCALER_Y)

# Determinar features desde el scaler si están disponibles
if hasattr(scaler_x, "feature_names_in_"):
    features = list(scaler_x.feature_names_in_)
else:
    # Fallback: intentar todas menos columnas no-numéricas y el target
    cols_excluir = {"datetime", TARGET}
    features = [c for c in df.columns if c not in cols_excluir]

# Validar que todas las features existan en df
features_presentes = [c for c in features if c in df.columns]
if len(features_presentes) != len(features):
    faltan = set(features) - set(features_presentes)
    print(f"[AVISO] Faltan columnas en el dataset para escalar: {faltan}. Se usarán solo las presentes: {features_presentes}")
    features = features_presentes

# Cargar modelo LSTM
modelo = load_model(
    RUTA_MODELO,
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)

# =============================
# 2) Predicción del spread
# =============================
X = df[features].values
X_scaled = scaler_x.transform(X)
X_seq = create_sequences(X_scaled, time_steps=TIME_STEPS)

y_pred_scaled = modelo.predict(X_seq)
preds = scaler_y.inverse_transform(y_pred_scaled).flatten()

# Alinear índices tras crear secuencias
df = df.iloc[TIME_STEPS:].reset_index(drop=True)
df["spread_pred"] = preds

# =============================
# 3) Simulación operativa
# =============================
capital = capital_inicial
mwh_acumulados = 0.0
historial = []

for _, row in df.iterrows():
    pred = row["spread_pred"]
    real = row[TARGET]
    # IMPORTANTE: debe existir la columna 'demanda' en df
    energia_operada = row.get("demanda", 0.0)

    ganancia = 0.0
    accion = "No operó"

    # Estrategia de predicción: COMPRA si se espera que intra suba
    if pred > 1:
        mwh_acumulados += energia_operada
        accion = f"Compró {energia_operada:.2f} MWh (demanda real)"
    elif pred < -1:
        accion = "No operó (predijo intra más barato)"

    # Evaluación real: VENTA si intra realmente es más caro
    if real > 0 and mwh_acumulados > 0:
        venta_mwh = min(mwh_acumulados, max_mwh_por_hora)
        ganancia = venta_mwh * real
        mwh_acumulados -= venta_mwh
        accion += f" y vendió {venta_mwh:.2f} MWh en intra (+{ganancia:.2f}€)"

    capital += ganancia
    historial.append({
        "datetime": row["datetime"],
        "spread_pred": pred,
        "spread_real": real,
        "accion": accion,
        "mwh_acumulados": mwh_acumulados,
        "ganancia": ganancia,
        "capital": capital,
    })

# Resultados de la simulación
df_resultado = pd.DataFrame(historial)

# =============================
# 4) Enriquecer con precios absolutos (diario vs intra)
# =============================

df_precios = pd.read_csv(RUTA_DATASET_PRECIOS, parse_dates=["datetime"])  # debe contener columnas 'precio_electricidad' y 'precio_intradiario'
df_precios = df_precios[["datetime", "precio_electricidad", "precio_intradiario"]]

df_resultado = df_resultado.merge(df_precios, on="datetime", how="left")

# Simulación de precios absolutos
df_resultado["precio_diario"] = df_resultado["precio_electricidad"]
df_resultado["precio_intra"] = df_resultado["precio_intradiario"]

# Etiqueta de resultado por punto temporal
def etiquetar_resultado(x: float) -> str:
    return "ganancia" if x > 0 else ("perdida" if x < 0 else "neutral")

df_resultado["resultado"] = df_resultado["ganancia"].apply(etiquetar_resultado)

# =============================
# 5) Gráficos (los mismos que en los tres códigos originales)
# =============================

plot_capital_y_precios(df_resultado, capital_inicial)
plot_capital_y_precios_subplots(df_resultado, capital_inicial)

# =============================
# 6) (Opcional) Guardar resultados
# =============================
# df_resultado.to_csv("resultado_simulacion.csv", index=False)
