import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Espacios de búsqueda
TIME_STEPS_LIST = [12, 24, 48]
UNITS_LIST      = [32, 64]
LR_LIST         = [1e-2, 1e-3]
BATCH_SIZE      = 64
EPOCHS          = 75

# 1) Carga y escalado
df = pd.read_csv("dataset_modelo_spread.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
features = [c for c in df.columns if c not in ["datetime","spread","precio_electricidad","precio_intradiario"]]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(df[features])
y = scaler_y.fit_transform(df[["spread"]])

def create_sequences(X, y, ts):
    xs, ys = [], []
    for i in range(ts, len(X)):
        xs.append(X[i-ts:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

# === Métricas de ACIERTO DE SIGNO binario (>=0 vs <0) ===
def binary_sign_metrics(y_true, y_pred, eps=0.0):
    """
    Clasifica en:
      0 -> spread >= eps (no-negativo)
      1 -> spread < -eps (negativo)
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    yt_bin = (yt < -eps).astype(int)
    yp_bin = (yp < -eps).astype(int)
    acc = (yt_bin == yp_bin).mean()
    cm = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
    return acc, cm

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")

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

best_rmse = float("inf")
best_cfg  = None

# 2) Grid search
for ts, units, lr in itertools.product(TIME_STEPS_LIST, UNITS_LIST, LR_LIST):
    X_seq, y_seq = create_sequences(X, y, ts)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    tf.random.set_seed(42); np.random.seed(42)
    model = Sequential([
        Input(shape=(ts, X_train.shape[2])),
        LSTM(units, activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred = scaler_y.inverse_transform(model.predict(X_test))
    y_true = scaler_y.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"ts={ts}, units={units}, lr={lr:.1e} → RMSE={rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_cfg = (ts, units, lr)

print("\n>>> Mejor configuración")
print(f"Time steps={best_cfg[0]}, Units={best_cfg[1]}, LR={best_cfg[2]:.1e} → RMSE={best_rmse:.4f}")

# 3) Volver a entrenar con la mejor configuración y graficar
ts, units, lr = best_cfg
X_seq, y_seq = create_sequences(X, y, ts)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential([
    Input(shape=(ts, X_train.shape[2])),
    LSTM(units, activation="tanh"),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

y_pred = scaler_y.inverse_transform(model.predict(X_test))
y_true = scaler_y.inverse_transform(y_test)

rmse_best_refit = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE (reentrenado con mejor config): {rmse_best_refit:.4f}")

# === 3.b Acierto de signo binario ===
eps = 0.0
acc_bin, cm_bin = binary_sign_metrics(y_true, y_pred, eps=eps)
print(f"Directional Accuracy binario (>=0 vs <0, eps={eps}): {acc_bin*100:.2f}%")
print("Matriz de confusión 2x2 (test):\n", cm_bin)

plot_confusion_matrix(
    cm_bin,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (test completo)\nAcc: {acc_bin*100:.2f}%"
)

# 4) Gráficos de la mejor configuración
# 4.1 Completo
plt.figure(figsize=(16,6))
plt.plot(y_true, label="Real", linewidth=1)
plt.plot(y_pred, label="Predicho", linewidth=1, alpha=0.8)
plt.title(f"Mejor config (ts={ts}, units={units}, lr={lr:.1e})\n"
          f"RMSE grid={best_rmse:.4f} | RMSE refit={rmse_best_refit:.4f} | "
          f"DA bin={acc_bin*100:.2f}%")
plt.xlabel("Timestep")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()

# 4.2 Últimos 1000 pasos
n = 1000
y_t_last = y_true[-n:]
y_p_last = y_pred[-n:]
rmse_last = np.sqrt(mean_squared_error(y_t_last, y_p_last))

acc_last, cm_last = binary_sign_metrics(y_t_last, y_p_last, eps=eps)
print(f"RMSE últimos {n} pasos: {rmse_last:.4f}")
print(f"Directional Accuracy binario últimos {n}: {acc_last*100:.2f}%")
print("Matriz 2x2 últimos n:\n", cm_last)

plot_confusion_matrix(
    cm_last,
    labels=["≥0","<0"],
    title=f"Matriz de confusión (últimos {n})\nAcc: {acc_last*100:.2f}%"
)

plt.figure(figsize=(16,6))
plt.plot(y_t_last, label="Real", linewidth=1)
plt.plot(y_p_last, label="Predicho", linewidth=1, alpha=0.8)
plt.title(f"Últimos {n} pasos "
          f"(RMSE: {rmse_last:.4f} | DA bin: {acc_last*100:.2f}%)")
plt.xlabel("Últimos pasos")
plt.ylabel("Spread")
plt.legend()
plt.tight_layout()
plt.show()
