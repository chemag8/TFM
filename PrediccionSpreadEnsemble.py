import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense

# Parámetros del ensemble
N_MODELS   = 5
TIME_STEPS = 24
BATCH_SIZE = 64
EPOCHS     = 100
PATIENCE   = 20

# 1) Carga y escalado
df = pd.read_csv("dataset_modelo_spread.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
features = [c for c in df.columns if c not in ["datetime","spread","precio_electricidad","precio_intradiario"]]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[["spread"]])

# 2) Crear secuencias
def create_sequences(X, y, ts):
    xs, ys = [], []
    for i in range(ts, len(X)):
        xs.append(X[i-ts:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, TIME_STEPS)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# 3) Función para construir un modelo
def build_model(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, activation="tanh"),
        BatchNormalization(), Dropout(0.1),
        LSTM(32, activation="tanh"),
        BatchNormalization(), Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

# 4) Entrenar ensemble y recolectar predicciones
preds = []
for i in range(N_MODELS):
    seed = 1234 + i
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model((TIME_STEPS, X_train.shape[2]))
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0
    )
    preds.append(model.predict(X_test))

# 5) Ensemble y evaluación
y_pred_ens_scaled = np.mean(np.stack(preds, axis=0), axis=0)
y_pred = scaler_y.inverse_transform(y_pred_ens_scaled)
y_true = scaler_y.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE Ensemble ({N_MODELS} modelos): {rmse:.4f}")

# === 5.b Métricas de signo binario (>=0 vs <0) + Matriz 2x2 pintada ===
def binary_sign_metrics(y_true, y_pred, eps=0.0):
    """
    Clasifica en:
      0 -> spread >= eps  (no-negativo)
      1 -> spread <  -eps (negativo)
    eps actúa como banda muerta alrededor de 0.
    Devuelve (accuracy, confusion_matrix_2x2)
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    yt_bin = (yt < -eps).astype(int)  # 1 si negativo, 0 si >=0
    yp_bin = (yp < -eps).astype(int)
    acc = (yt_bin == yp_bin).mean()
    cm = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
    return acc, cm

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")

    # Números encima
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# Completo (test)
eps = 0.0  # prueba 0.05–0.10 si quieres ignorar micro-ruido
acc_bin, cm_bin = binary_sign_metrics(y_true, y_pred, eps=eps)
print(f"Directional Accuracy binario (>=0 vs <0, eps={eps}): {acc_bin*100:.2f}%")
print("Matriz 2x2 (test):\n", cm_bin)

plot_confusion_matrix(
    cm_bin,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (test completo)\nAcc: {acc_bin*100:.2f}%"
)

# 6) Gráficos
# 6.1 Completo
plt.figure(figsize=(16,6))
plt.plot(y_true, label="Real", linewidth=1)
plt.plot(y_pred, label="Ensemble predicho", linewidth=1, alpha=0.8)
plt.title(f"Ensemble LSTM - Spread (completo)\nRMSE: {rmse:.4f}")
plt.xlabel("Timestep")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()

# 6.2 Últimos 1000 pasos + matriz 2x2 para ese tramo
n = 1000
y_t_last = y_true[-n:]
y_p_last = y_pred[-n:]
rmse_last = np.sqrt(mean_squared_error(y_t_last, y_p_last))
print(f"RMSE últimos {n} pasos: {rmse_last:.4f}")

acc_last, cm_last = binary_sign_metrics(y_t_last, y_p_last, eps=eps)
print(f"Directional Accuracy binario últimos {n}: {acc_last*100:.2f}%")
print("Matriz 2x2 (últimos n):\n", cm_last)

plot_confusion_matrix(
    cm_last,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (últimos {n})\nAcc: {acc_last*100:.2f}%"
)
