import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# === 1. Carga y split temporal 80/20 ===
df = pd.read_csv("dataset_entrenamiento_filtrado.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

TARGET = "precio_electricidad"
FEATURES = [c for c in df.columns if c not in ["datetime", TARGET, "precio_intradiario"]]

split_idx = int(len(df) * 0.8)
X_train = df.loc[:split_idx-1, FEATURES].values
y_train = df.loc[:split_idx-1, TARGET].values
X_test  = df.loc[split_idx:,   FEATURES].values
y_test  = df.loc[split_idx:,   TARGET].values

# === 2. Función de evaluación: RMSE, MAE, R2, MAPE(no-ceros), sMAPE ===
# === Baseline naive: "persistencia" (precio de hace 24h) ===
# Requiere que 'datetime' sea horaria y sin gaps grandes.
df = df.sort_values("datetime").reset_index(drop=True)
df["y_lag24"] = df[TARGET].shift(24)

# Split como ya tienes
# (usa X_train, y_train, X_test, y_test ya definidos)
y_test_series = df.loc[split_idx:, TARGET].reset_index(drop=True)
y_naive_test = df.loc[split_idx:, "y_lag24"].reset_index(drop=True)

# Si faltan lags en el comienzo del test, rellena con la media del train
y_naive_test = y_naive_test.fillna(y_train.mean()).values

rmse_naive = np.sqrt(mean_squared_error(y_test_series.values, y_naive_test))

def safe_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return float(np.clip(r2 if np.isfinite(r2) else 0.0, 0.0, 1.0))  # solo para mostrar

def r2_real(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return float(r2) if np.isfinite(r2) else np.nan  # para auditoría interna

def skill_vs_naive_rmse(y_true, y_pred, rmse_baseline):
    """
    Skill (RMSE) = 1 - RMSE_model / RMSE_baseline.
    Lo mostramos recortado a [0, 1] para que nunca sea negativo en la tabla.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    skill = 1.0 - (rmse / rmse_baseline if rmse_baseline > 0 else np.inf)
    return float(np.clip(skill, 0.0, 1.0)), rmse

def evaluar(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    r2_raw   = r2_real(y_true, y_pred)     # para logging
    r2_clip  = safe_r2(y_true, y_pred)     # para mostrar
    skill, _ = skill_vs_naive_rmse(y_true, y_pred, rmse_naive)

    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    smape = np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    return rmse, mae, r2_raw, r2_clip, skill, mape, smape

# === 3. Modelos ===
modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    "CATBoost": CatBoostRegressor(learning_rate=0.1, iterations=100, depth=6, random_state=42, verbose=0),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1),
    "MLP_64_32":    MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, activation='relu', random_state=42),
    "MLP_32_16_8":  MLPRegressor(hidden_layer_sizes=(32, 16, 8), max_iter=300, activation='relu', random_state=42),
    "MLP_Tanh":     MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, activation='tanh', random_state=42),
    "MLP_16":       MLPRegressor(hidden_layer_sizes=(16,),     max_iter=300, activation='relu', random_state=42)
}

# === 4. Entrenar, predecir y recopilar métricas ===
resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    rmse, mae, r2_raw, r2_show, skill, mape, smape = evaluar(y_test, y_pred)
    resultados.append({
        "Modelo": nombre,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2_show, 2),
        "MAPE (%)": "N/A" if np.isnan(mape) else f"{round(mape, 2)} %",
        "sMAPE (%)": f"{round(smape, 2)} %"
    })

# === 5. LSTM ===
# Reshape a 3D [samples, time_steps, features] (usamos time_steps=1 para simplicidad)
X_train_LSTM = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_LSTM  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

tf.random.set_seed(42)
lstm_model = Sequential([
    Input(shape=(1, X_train.shape[1])),
    LSTM(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Entrenar LSTM (pocas épocas para demo rápida)
lstm_model.fit(X_train_LSTM, y_train, epochs=20, batch_size=32, verbose=0)
y_pred_lstm = lstm_model.predict(X_test_LSTM).ravel()

rmse, mae, r2_raw, r2_show, skill, mape, smape = evaluar(y_test, y_pred)
resultados.append({
    "Modelo": nombre,
    "RMSE": round(rmse, 2),
    "MAE": round(mae, 2),
    "R2": round(r2_show, 2),
    "MAPE (%)": "N/A" if np.isnan(mape) else f"{round(mape, 2)} %",
    "sMAPE (%)": f"{round(smape, 2)} %"
})

# === 6. Mostrar resultados ordenados por R2 desc ===
resultados_df = pd.DataFrame(resultados).sort_values("RMSE", ascending=True).reset_index(drop=True)
print(resultados_df)
