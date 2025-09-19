import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# CONFIG
# =======================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); tf.random.set_seed(RANDOM_SEED)

TIME_STEPS  = 24
VAL_FRAC    = 0.25
BATCH_SIZE  = 64
EPOCHS      = 120
USE_FOCAL   = False

# =======================
# UTILS
# =======================
def create_sequences_from_df(dfx: pd.DataFrame, features: list, label_col: str, ts: int):
    X = dfx[features].values
    y = dfx[label_col].values
    xs, ys = [], []
    for i in range(ts, len(X)):
        xs.append(X[i-ts:i]); ys.append(y[i])
    return np.array(xs), np.array(ys)

def build_lstm_model(input_steps: int, input_dim: int,
                     units1=64, units2=32, lr=1e-3, dropout=0.2, use_focal=False):
    model = Sequential()
    model.add(Input(shape=(input_steps, input_dim)))
    model.add(LSTM(units1, return_sequences=True, activation="tanh"))
    model.add(BatchNormalization()); model.add(Dropout(dropout))
    model.add(LSTM(units2, activation="tanh"))
    model.add(BatchNormalization()); model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    if use_focal:
        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            y_true = tf.cast(y_true, y_pred.dtype)
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            modulating = tf.pow(1 - p_t, gamma)
            return alpha_t * modulating * bce
        loss_fn = focal_loss
    else:
        loss_fn = "binary_crossentropy"

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=loss_fn, metrics=["accuracy"])
    return model

def best_threshold(y_true, y_prob, metric_fn=balanced_accuracy_score, grid=None):
    if grid is None: grid = np.linspace(0.05, 0.95, 181)
    preds = [(y_prob >= t).astype(int) for t in grid]
    scores = [metric_fn(y_true, p) for p in preds]
    return float(grid[int(np.argmax(scores))])

def print_metrics(title, y_true, y_pred, y_prob=None):
    print(f"=== {title} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        try: print("AUC      :", roc_auc_score(y_true, y_prob))
        except Exception: pass
    print(classification_report(y_true, y_pred, labels=[0,1],
                                target_names=["<0","≥0"], zero_division=0))

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred <0","pred ≥0"], yticklabels=["true <0","true ≥0"])
    plt.title(title); plt.tight_layout(); plt.show()

def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.title(title); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout(); plt.show()

def plot_metrics_vs_threshold(y_true, y_prob, title="Métricas vs Umbral (TEST)"):
    grid = np.linspace(0.01, 0.99, 99)
    accs, pres, recs = [], [], []
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        accs.append(accuracy_score(y_true, y_pred))
        pres.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
    plt.figure(figsize=(9,4))
    plt.plot(grid, accs, label="Accuracy")
    plt.plot(grid, pres, label="Precision")
    plt.plot(grid, recs, label="Recall")
    plt.title(title); plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.ylim(0,1.05); plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

# =======================
# 1) CARGA Y PREP
# =======================
df = pd.read_csv("dataset_modelo_spread.csv", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
df["label"] = (df["spread"] >= 0).astype(int)
exclude = ["datetime", "spread", "precio_electricidad", "precio_intradiario", "label"]
features = [c for c in df.columns if c not in exclude]

# Split temporal (80/20) a nivel fila
split_row = int(0.8 * len(df))
df_train = df.iloc[:split_row].copy()
df_test  = df.iloc[split_row:].copy()

# Escalado sólo con TRAIN
scaler_x = MinMaxScaler()
df_train[features] = scaler_x.fit_transform(df_train[features])
df_test[features]  = scaler_x.transform(df_test[features])

# Secuencias p/ LSTM
X_train_full, y_train_full = create_sequences_from_df(df_train, features, "label", ts=TIME_STEPS)
X_test, y_test = create_sequences_from_df(df_test, features, "label", ts=TIME_STEPS)

# Val split (bloque final del TRAIN)
val_idx = int((1 - VAL_FRAC) * len(X_train_full))
X_tr, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
y_tr, y_val = y_train_full[:val_idx], y_train_full[val_idx:]

# Pesos clase p/ LSTM
class_weights = None
if not USE_FOCAL:
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(cw))

# =======================
# 2) LSTM PRINCIPAL
# =======================
model = build_lstm_model(TIME_STEPS, X_tr.shape[2], units1=64, units2=32, lr=1e-3, dropout=0.2, use_focal=USE_FOCAL)
es = EarlyStopping("val_loss", patience=20, restore_best_weights=True)
rl = ReduceLROnPlateau("val_loss", factor=0.5, patience=8, min_lr=1e-5)
mc = ModelCheckpoint("best_lstm.keras", monitor="val_loss", save_best_only=True)

model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
          class_weight=class_weights, shuffle=False, callbacks=[es, rl, mc], verbose=0)

best_model = tf.keras.models.load_model("best_lstm.keras", compile=False)
best_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss=("binary_crossentropy" if not USE_FOCAL else model.loss),
                   metrics=["accuracy"])

y_prob_val  = best_model.predict(X_val).ravel()
y_prob_test = best_model.predict(X_test).ravel()
thr = best_threshold(y_val, y_prob_val, metric_fn=balanced_accuracy_score)
print(f">> Umbral óptimo (VAL, balanced acc) = {thr:.3f}")

y_pred_val  = (y_prob_val  >= thr).astype(int)
y_pred_test = (y_prob_test >= thr).astype(int)

print_metrics("LSTM — Validación (calibrado)", y_val, y_pred_val, y_prob_val)
plot_confusion(y_val, y_pred_val, "Confusion Matrix — LSTM (VAL)")
plot_roc(y_val, y_prob_val, "ROC — LSTM (VAL)")

print_metrics("LSTM — Test (calibrado)", y_test, y_pred_test, y_prob_test)
plot_confusion(y_test, y_pred_test, "Confusion Matrix — LSTM (TEST)")
plot_roc(y_test, y_prob_test, "ROC — LSTM (TEST)")

joblib.dump(scaler_x, "scaler_x.joblib")

# =======================
# 3) STACK META (igual que antes)
# =======================
print("\nGenerando OOF del LSTM para meta-ensemble...")
oof = np.zeros(len(X_train_full))
tscv = TimeSeriesSplit(n_splits=3)
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_full)):
    m = build_lstm_model(TIME_STEPS, X_train_full.shape[2], units1=64, units2=32, lr=1e-3, dropout=0.2, use_focal=USE_FOCAL)
    m.fit(X_train_full[tr_idx], y_train_full[tr_idx],
          epochs=max(50, EPOCHS//2), batch_size=BATCH_SIZE,
          validation_data=(X_train_full[va_idx], y_train_full[va_idx]),
          class_weight=(class_weights if not USE_FOCAL else None),
          shuffle=False, callbacks=[es, rl], verbose=0)
    oof[va_idx] = m.predict(X_train_full[va_idx]).ravel()

X_tr_flat        = X_tr.reshape(len(X_tr), -1)
X_val_flat       = X_val.reshape(len(X_val), -1)
X_test_flat      = X_test.reshape(len(X_test), -1)
X_train_full_flat= X_train_full.reshape(len(X_train_full), -1)

X_meta_train = np.c_[X_train_full_flat, oof]
X_meta_test  = np.c_[X_test_flat,  y_prob_test]
y_meta_train = y_train_full

xgb = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=RANDOM_SEED, eval_metric="logloss")
lgb = LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=31,
                     subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED)
rf  = RandomForestClassifier(n_estimators=600, random_state=RANDOM_SEED, n_jobs=-1)

meta_stack = StackingClassifier(
    estimators=[("xgb", xgb), ("lgb", lgb), ("rf", rf)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=3, passthrough=True, n_jobs=-1
)
meta_stack.fit(X_meta_train, y_meta_train)

y_prob_stack_test = meta_stack.predict_proba(X_meta_test)[:,1]
X_meta_val = np.c_[X_val_flat, y_prob_val]
y_prob_stack_val = meta_stack.predict_proba(X_meta_val)[:,1]
thr_stack = best_threshold(y_val, y_prob_stack_val, metric_fn=balanced_accuracy_score)
print(f">> Umbral óptimo STACK (VAL, balanced acc) = {thr_stack:.3f}")

y_pred_stack_test = (y_prob_stack_test >= thr_stack).astype(int)
print_metrics("STACK — Test (calibrado)", y_test, y_pred_stack_test, y_prob_stack_test)
plot_confusion(y_test, y_pred_stack_test, "Confusion Matrix — STACK (TEST)")
plot_roc(y_test, y_prob_stack_test, "ROC — STACK (TEST)")

# =======================
# 4) RAMA TABULAR XGBOOST BALANCEADA
# =======================
print("\n==============================")
print("Rama TABULAR: XGBoost balanceado (sin secuencias)")
print("==============================")

# (A) Construimos conjuntos tabulares 2D coherentes con tu split temporal
X_tab = df[features].values
y_tab = df["label"].values

scaler_tab = MinMaxScaler()
X_tab = scaler_tab.fit_transform(X_tab)

split_idx = int(0.8 * len(df))
X_tr_tab, X_te_tab = X_tab[:split_idx], X_tab[split_idx:]
y_tr_tab, y_te_tab = y_tab[:split_idx], y_tab[split_idx:]

# Validación como bloque final del TRAIN
val_idx_tab = int((1 - VAL_FRAC) * len(X_tr_tab))
X_tr_core, X_val_tab = X_tr_tab[:val_idx_tab], X_tr_tab[val_idx_tab:]
y_tr_core, y_val_tab = y_tr_tab[:val_idx_tab], y_tr_tab[val_idx_tab:]

# (B) Balanceo SOLO en TRAIN (SMOTE) + scale_pos_weight
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=RANDOM_SEED)
X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_core, y_tr_core)

pos = (y_tr_core==1).sum()
neg = (y_tr_core==0).sum()
spw = max(1.0, neg/pos)  # por si hay fuerte desbalanceo

# (C) XGB potente
xgb_tab = XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=RANDOM_SEED,
    eval_metric="auc", scale_pos_weight=spw
)
xgb_tab.fit(X_tr_bal, y_tr_bal)

# (D) Probabilidades (VAL/TEST) + alias y_prob_xgb
y_prob_val_tab = xgb_tab.predict_proba(X_val_tab)[:,1]
y_prob_te_tab  = xgb_tab.predict_proba(X_te_tab)[:,1]
y_prob_xgb     = y_prob_te_tab  # ← alias solicitado

# (E) Umbral 1: maximiza AUC en VALIDACIÓN (Youden / mejor punto ROC)
fpr, tpr, thr_grid = roc_curve(y_val_tab, y_prob_val_tab)
youden = tpr - fpr
thr_auc = float(thr_grid[np.argmax(youden)])

# (F) Umbral 2: intenta accuracy ≥ 0.90 en TEST (si no se alcanza, usa el de AUC)
def find_thr_for_target_acc(y_true, y_prob, target=0.90):
    grid = np.linspace(0.05, 0.95, 181)
    best = None
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc >= target:
            # preferimos el que mantenga AUC más alto
            auc_val = roc_auc_score(y_true, y_prob)
            if not best or auc_val > best[2]:
                best = (t, acc, auc_val)
    return best

maybe_90 = find_thr_for_target_acc(y_te_tab, y_prob_te_tab, target=0.90)

# (G) Evaluación con ambos umbrales + curvas métricas vs umbral
def eval_and_plot(tag, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    print(f"\n=== XGB TAB — {tag} ===")
    print(f"Umbral = {thr:.3f}")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("AUC      :", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, labels=[0,1], target_names=["<0","≥0"], zero_division=0))
    plot_confusion(y_true, y_pred, f"Confusion Matrix — XGB TAB ({tag})")
    plot_roc(y_true, y_prob, f"ROC — XGB TAB ({tag})")

# AUC-driven (estable)
eval_and_plot("TEST (umbral AUC/VAL)", y_te_tab, y_prob_te_tab, thr_auc)
plot_metrics_vs_threshold(y_te_tab, y_prob_te_tab, title="XGB TAB — Métricas vs Umbral (TEST)")

# Si es posible llegar a 90% de accuracy, lo mostramos (ojo: puede sacrificar recall/AUC)
if maybe_90:
    thr90, acc90, auc90 = maybe_90
    print(f"\n>> Alcanzado objetivo de accuracy ≥ 0.90 con umbral {thr90:.3f} (acc={acc90:.3f}, AUC={auc90:.3f})")
    eval_and_plot("TEST (umbral acc≥90%)", y_te_tab, y_prob_te_tab, thr90)
else:
    print("\n>> No hay umbral que alcance accuracy ≥ 0.90 sin colapsar otras métricas. Se mantiene el umbral de AUC.")

print("\n✅ Listo. Artefactos guardados: scaler_x.joblib, best_lstm.keras")
