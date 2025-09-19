import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ========= Config =========
TRAIN_RATIO   = 0.80
N_ITER_SEARCH = 25
N_SPLITS_CV   = 5
RANDOM_STATE  = 42
PLOT_MARKERS  = False
N_LAST_STEPS  = 1000

# ========= 1) Carga =========
df = pd.read_csv("dataset_entrenamiento_filtrado.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# ========= 2) Features y target =========
X = df.drop(columns=["datetime", "precio_electricidad", "precio_intradiario"])
y = df["precio_electricidad"].astype(float).values

# ========= 3) Split temporal 80/20 =========
n_total  = len(df)
n_train  = int(n_total * TRAIN_RATIO)
n_hold   = n_total - n_train
X_train, X_holdout = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_holdout = y[:n_train], y[n_train:]
print(f"Split temporal → Train: {n_train} | Holdout: {n_hold}")

# ========= 4) Escalado =========
scaler = StandardScaler()
X_train_scaled   = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

# ========= 5) CV temporal =========
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

# ========= 6) Modelos + espacios =========
models = {
    "XGB": XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
    "LGBM": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "CAT": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
}

param_grids = {
    "XGB": {
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0]
    },
    "LGBM": {
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [-1, 6, 12],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.0, 5.0, 10.0]
    },
    "CAT": {
        "iterations": [300, 600, 1000],
        "learning_rate": [0.01, 0.03, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 5.0]
    }
}

# ========= Helpers =========
def print_metrics(y_true, y_pred, title=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {title}")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²  : {r2:.4f}")
    return mae, rmse, r2

def plot_series(y_true, y_pred, title):
    plt.figure(figsize=(14, 4))
    if PLOT_MARKERS:
        plt.plot(y_true, label="Real", linewidth=1, marker='o', markersize=3)
        plt.plot(y_pred, label="Predicho", linewidth=1, marker='x', markersize=3)
    else:
        plt.plot(y_true, label="Real", linewidth=1)
        plt.plot(y_pred, label="Predicho", linewidth=1, alpha=0.85)
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_feature_importance(model, feature_names, title):
    try:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[order], align="center")
        plt.xticks(range(len(importances)), feature_names[order], rotation=90)
        plt.title(title); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"(No se pudieron obtener importancias: {e})")

# ========= 7) Tuning + evaluación =========
best_models = {}
feature_names = np.array(X_train.columns)

for name, model in models.items():
    print(f"\n================= {name} =================")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],
        n_iter=N_ITER_SEARCH,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    search.fit(X_train_scaled, y_train)
    best_models[name] = search.best_estimator_
    print(f"> Mejores params {name}: {search.best_params_}")

    y_pred_holdout = best_models[name].predict(X_holdout_scaled)
    print_metrics(y_holdout, y_pred_holdout, f"{name} • HOLDOUT completo")
    plot_series(y_holdout, y_pred_holdout, f"{name} — Holdout completo")
    plot_feature_importance(best_models[name], feature_names, f"Importancia de variables — {name}")

    # últimos N
    if N_LAST_STEPS < len(y_holdout):
        y_true_last = y_holdout[-N_LAST_STEPS:]
        y_pred_last = y_pred_holdout[-N_LAST_STEPS:]
        print_metrics(y_true_last, y_pred_last, f"{name} • Últimos {N_LAST_STEPS}")
        plot_series(y_true_last, y_pred_last, f"{name} — Últimos {N_LAST_STEPS}")

# ========= 8) Stacking =========
# predicciones base en holdout
hold_base_preds = np.column_stack([best_models[m].predict(X_holdout_scaled) for m in best_models])
meta_scaler = StandardScaler()
hold_meta = meta_scaler.fit_transform(hold_base_preds)

meta = RidgeCV(alphas=(0.1, 1.0, 10.0))
meta.fit(hold_meta, y_holdout)  # meta entrenado directamente en holdout (simplificado)
y_pred_stack = meta.predict(hold_meta)

print_metrics(y_holdout, y_pred_stack, f"STACK • HOLDOUT completo")
plot_series(y_holdout, y_pred_stack, f"STACK — Holdout completo")

if N_LAST_STEPS < len(y_holdout):
    y_true_last = y_holdout[-N_LAST_STEPS:]
    y_pred_last = y_pred_stack[-N_LAST_STEPS:]
    print_metrics(y_true_last, y_pred_last, f"STACK • Últimos {N_LAST_STEPS}")
    plot_series(y_true_last, y_pred_last, f"STACK — Últimos {N_LAST_STEPS}")

plt.figure(figsize=(6,4))
plt.bar(best_models.keys(), meta.coef_)
plt.title("Pesos del meta (RidgeCV)")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# ========= 9) Tabla comparativa =========
def metrics_dict(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred)
    }

rows = []
for name in best_models:
    m_full = metrics_dict(y_holdout, best_models[name].predict(X_holdout_scaled))
    m_last = metrics_dict(
        y_holdout[-N_LAST_STEPS:], 
        best_models[name].predict(X_holdout_scaled)[-N_LAST_STEPS:]
    ) if N_LAST_STEPS < len(y_holdout) else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    rows.append((name, m_full["MAE"], m_full["RMSE"], m_full["R2"],
                 m_last["MAE"], m_last["RMSE"], m_last["R2"]))

m_full = metrics_dict(y_holdout, y_pred_stack)
m_last = metrics_dict(
    y_holdout[-N_LAST_STEPS:], 
    y_pred_stack[-N_LAST_STEPS:]
) if N_LAST_STEPS < len(y_holdout) else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
rows.append(("STACK", m_full["MAE"], m_full["RMSE"], m_full["R2"],
             m_last["MAE"], m_last["RMSE"], m_last["R2"]))

cols = pd.MultiIndex.from_product(
    [["Holdout completo", f"Últimos {N_LAST_STEPS}"], ["MAE", "RMSE", "R2"]],
    names=["Tramo", "Métrica"]
)

index = [r[0] for r in rows]
data  = [r[1:] for r in rows]

metrics_table = pd.DataFrame(data, index=index, columns=cols)

# ordenar por MAE del holdout (menor es mejor)
metrics_table = metrics_table.sort_values(("Holdout completo", "MAE"))

print("\n==================== RESUMEN DE MÉTRICAS ====================")
print(metrics_table)
