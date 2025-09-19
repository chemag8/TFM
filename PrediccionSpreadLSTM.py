import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import tensorflow as tf

# --- 1. Cargar y preparar datos (asumes que ya trae hour/dow/month) ---
df = pd.read_csv("dataset_modelo_spread.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

target = "spread"
exclude = ["datetime", target, "precio_electricidad", "precio_intradiario"]
features = [c for c in df.columns if c not in exclude]

# --- 2. Escalar datos ---
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]])

# --- 3. Crear secuencias para LSTM ---
def create_sequences(X, y, time_steps=24):
    xs, ys = [], []
    for i in range(time_steps, len(X)):
        xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps=time_steps)

# --- 4. División train/test ---
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# --- 5. Modelo: 2 LSTM + BatchNorm + Dropout + capa extra + callbacks ---
model = Sequential([
    Input(shape=(time_steps, X_train.shape[2])),

    LSTM(64, return_sequences=True, activation="tanh"),
    BatchNormalization(),
    Dropout(0.1),

    LSTM(32, activation="tanh"),
    BatchNormalization(),
    Dropout(0.1),

    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse", metrics=["mae"])

# Callbacks
es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-4)
mc = ModelCheckpoint("best_lstm_spread.keras", monitor="val_loss", save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es, rl, mc],
    verbose=1
)

# Carga el mejor modelo
model = load_model("best_lstm_spread.keras")

# --- 6. Predicción y evaluación ---
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE (test set): {rmse:.4f}")

# === 6.b Métricas de ACIERTO DE SIGNO binario (>=0 vs <0) ===
def binary_sign_metrics(y_true, y_pred, eps=0.0):
    """
    Clasifica el spread en:
      0 = spread >= eps (no-negativo)
      1 = spread < -eps (negativo)
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    yt_bin = (yt < -eps).astype(int)  # 1 si negativo, 0 si >=0
    yp_bin = (yp < -eps).astype(int)

    acc = (yt_bin == yp_bin).mean()
    cm = confusion_matrix(yt_bin, yp_bin, labels=[0,1])
    return acc, cm

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")

    # Texto con números
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

# Cálculo en test completo
eps = 0.0
acc_bin, cm_bin = binary_sign_metrics(y_true, y_pred, eps=eps)
print(f"Directional Accuracy binario (>=0 vs <0, eps={eps}): {acc_bin*100:.2f}%")
print("Matriz de confusión 2x2:\n", cm_bin)

plot_confusion_matrix(
    cm_bin,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (test completo)\nAcc: {acc_bin*100:.2f}%"
)

# --- 7. Guardar escalers ---
joblib.dump(scaler_x, "scaler_x.joblib")
joblib.dump(scaler_y, "scaler_y.joblib")

# --- 8. Gráficos de resultados ---

# 8.1 Gráfico completo
plt.figure(figsize=(16,6))
plt.plot(y_true,   label="Real",    linewidth=1)
plt.plot(y_pred,   label="Predicho",linewidth=1, alpha=0.8)
plt.title(f"Predicción del spread horario (completo)\nRMSE: {rmse:.4f}")
plt.xlabel("Timestep")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()

# 8.2 Gráfico últimos 1000 pasos + metrics de signo para ese tramo
n = 1000
y_t_last = y_true[-n:]
y_p_last = y_pred[-n:]
rmse_last = np.sqrt(mean_squared_error(y_t_last, y_p_last))
print(f"RMSE (últimos {n} pasos): {rmse_last:.4f}")

acc_last, cm_last = binary_sign_metrics(y_t_last, y_p_last, eps=eps)
print(f"Directional Accuracy binario últimos {n}: {acc_last*100:.2f}%")
print("Matriz 2x2 últimos n:\n", cm_last)

plot_confusion_matrix(
    cm_last,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (últimos {n})\nAcc: {acc_last*100:.2f}%"
)

plt.figure(figsize=(16,6))
plt.plot(y_t_last, label="Real",    linewidth=1)
plt.plot(y_p_last, label="Predicho",linewidth=1, alpha=0.8)
plt.title(f"Predicción del spread horario (últimos {n} pasos)\nRMSE: {rmse_last:.4f}")
plt.xlabel("Últimos pasos")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()
