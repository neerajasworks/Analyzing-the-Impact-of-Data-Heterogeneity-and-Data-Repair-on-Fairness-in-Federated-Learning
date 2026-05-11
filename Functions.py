import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import random
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import ot
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow_federated as tff

# =============================================================================
# DATA LOADING AND PRE-PROCESSING
# =============================================================================

def cargar_datos(ruta_train, ruta_test):
    cols = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
        "Martial Status", "Occupation", "Relationship", "OrigEthn",
        "Sex", "Capital Gain", "Capital Loss", "Hours per week",
        "Country", "Target",
    ]
    df_train = pd.read_csv(
        ruta_train, names=cols, sep=r"\s*,\s*",
        engine="python", na_values="?"
    )
    df_test = pd.read_csv(
        ruta_test, names=cols, sep=r"\s*,\s*",
        engine="python", na_values="?"
    )
    df = pd.concat([df_test, df_train])
    df.reset_index(inplace=True, drop=True)
    return df


def preprocesar_datos(df, variable_sensible):
    data = df.copy()
    data["Child"]    = np.where(data["Relationship"] == "Own-child", "ChildYes", "ChildNo")
    data["OrigEthn"] = np.where(data["OrigEthn"] == "White", "CaucYes", "CaucNo")
    data = data.drop(columns=["fnlwgt", "Relationship", "Country", "Education"])
    data = data.replace("<=50K.", "<=50K").replace(">50K.", ">50K")

    data_ohe = data.copy()
    data_ohe["Target"]   = np.where(data_ohe["Target"]   == ">50K",    1., 0.)
    data_ohe["OrigEthn"] = np.where(data_ohe["OrigEthn"] == "CaucYes", 1., 0.)
    data_ohe["Sex"]      = np.where(data_ohe["Sex"]      == "Male",    1., 0.)

    for col in ["Workclass", "Martial Status", "Occupation", "Child"]:
        if len(set(data_ohe[col].dropna())) == 2:
            val = data_ohe[col].dropna().iloc[0]
            data_ohe[col] = np.where(data_ohe[col] == val, 1., 0.)
        else:
            data_ohe = pd.get_dummies(data_ohe, prefix=[col], columns=[col])

    y = data_ohe["Target"].values.reshape(-1, 1)
    data_ohe_wo_target = data_ohe.drop(columns=["Target"])
    X_col_names = list(data_ohe_wo_target.columns)
    X = data_ohe_wo_target.astype(np.float32).values
    y = y.astype(np.float32)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    num_cols = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]
    scaler = StandardScaler()
    X[:, num_cols] = scaler.fit_transform(X[:, num_cols])

    return X, y, X_col_names


# =============================================================================
# DATA PARTITIONING ACROSS CLIENTS
# =============================================================================

def repartir_datos(X, y, variable_sensible, modo_reparto, num_clientes,
                   porc_min=0.05, porc_max=0.95, X_col_names=None):
    n = X.shape[0]
    repartos = []
    indices_usados = set()

    if modo_reparto == 1:
        idx_sensitive = X_col_names.index(variable_sensible)
        major_fracs   = np.linspace(porc_min, porc_max, num_clientes)
        idx_group1    = np.where(X[:, idx_sensitive] == 1)[0]
        idx_group0    = np.where(X[:, idx_sensitive] == 0)[0]
        np.random.shuffle(idx_group1)
        np.random.shuffle(idx_group0)

        max_per_client = int(
            min(len(idx_group0) / np.sum(major_fracs),
                len(idx_group1) / np.sum(1 - major_fracs))
        )
        pos1 = pos0 = 0
        for frac0 in major_fracs:
            n0 = int(round(max_per_client * frac0))
            n1 = max_per_client - n0
            idx0 = idx_group0[pos0:pos0 + n0]
            idx1 = idx_group1[pos1:pos1 + n1]
            pos0 += n0
            pos1 += n1
            indices = np.concatenate([idx0, idx1])
            indices_usados.update(indices)
            np.random.shuffle(indices)
            repartos.append({"X": X[indices], "y": y[indices]})
        resto = np.array(list(set(range(n)) - indices_usados))
        if len(resto) > 0:
            repartos.append({"X": X[resto], "y": y[resto]})
    else:
        indices = np.random.permutation(n)
        size = n // num_clientes
        for i in range(num_clientes):
            start = i * size
            end   = (i + 1) * size if i < num_clientes - 1 else n
            idx   = indices[start:end]
            indices_usados.update(idx)
            repartos.append({"X": X[idx], "y": y[idx]})
        resto = np.array(list(set(range(n)) - indices_usados))
        if len(resto) > 0:
            repartos.append({"X": X[resto], "y": y[resto]})

    return [cl for cl in repartos if cl["X"].shape[0] > 0]




# =============================================================================
# CLIENT SELECTION
# =============================================================================

def seleccionar_clientes_sesgados(repartos, variable_sensible, m,
                                   X_col_names, modo_sesgo="mayoritario"):
    porcentajes = []
    if variable_sensible == "Target":
        for i, cl in enumerate(repartos):
            y_cl  = cl["y"].flatten()
            total = y_cl.shape[0]
            prop  = np.sum(y_cl == 1) / total if total > 0 else 0
            if 0 < prop < 1:
                porcentajes.append((i, prop))
    else:
        idx = X_col_names.index(variable_sensible)
        for i, cl in enumerate(repartos):
            total = cl["X"].shape[0]
            prop  = np.sum(cl["X"][:, idx]) / total if total > 0 else 0
            if 0 < prop < 1:
                porcentajes.append((i, prop))

    if not porcentajes:
        return repartos

    if modo_sesgo == "mayoritario":
        sel = sorted(porcentajes, key=lambda x: x[1])[:m]
    elif modo_sesgo == "minoritario":
        sel = sorted(porcentajes, key=lambda x: -x[1])[:m]
    else:
        validos = list(range(len(repartos)))
        if m >= len(validos):
            sel = [(i, 0) for i in validos]
        else:
            escogidos = np.random.choice(validos, size=m, replace=False)
            sel = [(i, 0) for i in escogidos]

    return [repartos[i] for i, _ in sel]

def repartir_datos_noiid_target_dirichlet(X, y, num_clientes,
                                          beta=0.5, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    repartos = []
    indices_usados = set()

    y_flat = y.flatten()
    clases = np.unique(y_flat)
    min_size = 0
    min_requerido = 10

    while min_size < min_requerido:
        idx_batch = [[] for _ in range(num_clientes)]

        for k in clases:
            idx_k = np.where(y_flat == k)[0]
            np.random.shuffle(idx_k)

            # Dirichlet + avoid extreme 0/1 proportions
            proporciones = np.random.dirichlet(np.repeat(beta, num_clientes))
            proporciones = np.clip(proporciones, 0.05, None)
            proporciones = proporciones / proporciones.sum()

            puntos_corte = (np.cumsum(proporciones) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, puntos_corte))
            ]

        min_size = min(len(idx_j) for idx_j in idx_batch)

    for j in range(num_clientes):
        np.random.shuffle(idx_batch[j])
        indices = np.array(idx_batch[j])
        indices_usados.update(indices)
        repartos.append({"X": X[indices], "y": y[indices]})

    resto = np.array(list(set(range(n)) - indices_usados))
    if len(resto) > 0:
        repartos.append({"X": X[resto], "y": y[resto]})

    return [cl for cl in repartos if cl["X"].shape[0] > 0]
# =============================================================================
# MODEL DEFINITION
# =============================================================================

def crear_modelo(tipo_modelo, input_shape):
    if tipo_modelo == "simple":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation="sigmoid",
                                  input_shape=(input_shape,))
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu",
                                  input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    return model


def df_to_tf_dataset(X_np, y_np, batch_size=16, num_local_epochs=1):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_np.astype("float32"), y_np.astype("float32"))
    )
    return ds.repeat(num_local_epochs).shuffle(len(X_np)).batch(batch_size)


# =============================================================================
# TRAINING: FEDERATED AND CENTRALISED
# =============================================================================

def entrenar_federado(reparto, variable_sensible, rounds, c,
                       X_col_names, modo_sesgo="mayoritario"):
    reparto = [cl for cl in reparto if cl["X"].shape[0] > 0]
    federated_data = [df_to_tf_dataset(r["X"], r["y"]) for r in reparto]

    def model_fn():
        model = crear_modelo("simple", reparto[0]["X"].shape[1])
        return tff.learning.models.from_keras_model(
            keras_model=model,
            input_spec=federated_data[0].element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    )

    state = iterative_process.initialize()
    history = []
    num_clients = len(federated_data)
    m = min(max(1, round(c * num_clients)), num_clients)

    for r in range(1, rounds + 1):
        sel = seleccionar_clientes_sesgados(
            reparto, variable_sensible, m, X_col_names,
            modo_sesgo=modo_sesgo
        )
        selected_clients = [df_to_tf_dataset(cl["X"], cl["y"]) for cl in sel]
        result = iterative_process.next(state, selected_clients)
        state  = result.state
        m_data = result.metrics["client_work"]["train"]
        history.append({
            "Ronda":           r,
            "Loss":            m_data["loss"],
            "Binary Accuracy": m_data["binary_accuracy"],
            "Num Examples":    m_data["num_examples"],
        })

    keras_model = crear_modelo("simple", reparto[0]["X"].shape[1])
    fed_weights  = iterative_process.get_model_weights(state)
    fed_weights.assign_weights_to(keras_model)
    return {"modelo": keras_model, "hist": history}


def entrenar_centralizado(X_train, y_train, modelo, epochs=5):
    modelo.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    return {"modelo": modelo}




# =============================================================================
# FAIRNESS METRICS
# =============================================================================

def _grad_h(x):
    return np.array([
         x[3] / (x[1] * x[2]),
        -x[0] * x[3] / (x[1]**2 * x[2]),
        -x[0] * x[3] / (x[2]**2 * x[1]),
         x[0] / (x[1] * x[2]),
    ])


def disparate(X, S, Y, alpha=0.05):
    n    = X.shape[0]
    pi_1 = np.mean(X[:, S])
    pi_0 = 1 - pi_1
    p_1  = np.mean(X[:, S] * X[:, Y])
    p_0  = np.mean((1 - X[:, S]) * X[:, Y])

    if any(v == 0 for v in [pi_0, pi_1, p_0, p_1]):
        return np.nan, np.nan, np.nan, np.nan

    Tn   = p_0 * pi_1 / (p_1 * pi_0)
    grad = _grad_h(np.array([p_0, p_1, pi_0, pi_1]))

    Cov = np.zeros((4, 4))
    Cov[0, 1] = -p_0 * p_1
    Cov[0, 2] =  pi_1 * p_0
    Cov[0, 3] = -pi_1 * p_0
    Cov[1, 2] = -pi_0 * p_1
    Cov[1, 3] =  pi_0 * p_1
    Cov[2, 3] = -pi_0 * pi_1
    Cov      += Cov.T + np.diag([
        p_0 * (1 - p_0),
        p_1 * (1 - p_1),
        pi_0 * pi_1,
        pi_0 * pi_1,
    ])

    sigma = np.sqrt(grad @ Cov @ grad.T)
    z     = norm.ppf(1 - alpha / 2)
    lower = Tn - sigma * z / np.sqrt(n)
    upper = Tn + sigma * z / np.sqrt(n)
    BER   = 0.5 * (p_0 / pi_0 + 1 - p_1 / pi_1)
    return lower, Tn, upper, BER


def evaluar_modelo(modelo, X_test, y_test, variable_sensible,
                    X_col_names, alpha=0.05):
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    y_pred = (modelo.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_true = y_test.flatten()
    acc    = np.mean(y_pred == y_true)

    idx_s = (X_col_names.index("Sex")
             if variable_sensible == "Target"
             else X_col_names.index(variable_sensible))
    S = X_test[:, idx_s].ravel()

    X_ext = np.concatenate([X_test, y_pred.reshape(-1, 1)], axis=1)
    Y_idx = X_ext.shape[1] - 1
    lower, di_hat, upper, _ = disparate(X_ext, idx_s, Y_idx, alpha)

    spd, eod = calcular_spd_eod(y_true, y_pred, S)
    return acc, di_hat, spd, eod, (lower, di_hat, upper)


def calcular_intervalo_confianza(arr):
    return stats.t.interval(
        confidence=0.95,
        df=len(arr) - 1,
        loc=np.mean(arr),
        scale=stats.sem(arr),
    )


# =============================================================================
# WASSERSTEIN DISTANCE WITH CONFIDENCE INTERVAL
# =============================================================================

def wasserstein_ci(scores0, scores1, p=2, delta0=0.5, alpha=0.05):
    scores0 = np.asarray(scores0, dtype=float)
    scores1 = np.asarray(scores1, dtype=float)
    n0, n1  = len(scores0), len(scores1)
    x0_sort = np.sort(scores0)
    x1_sort = np.sort(scores1)

    grid = np.unique(np.concatenate([
        np.linspace(0, 1, n0 + 1),
        np.linspace(0, 1, n1 + 1),
    ]))
    q0 = np.quantile(scores0, grid)
    q1 = np.quantile(scores1, grid)
    Wp_p = float(np.sum(np.diff(grid) * np.abs(q0[:-1] - q1[:-1]) ** p))

    prob0   = np.arange(1, n0) / n0
    q1_at_0 = np.quantile(scores1, prob0)
    incr1   = (np.abs(x0_sort[1:]  - q1_at_0) ** p
             - np.abs(x0_sort[:-1] - q1_at_0) ** p)
    d1      = np.cumsum(incr1)
    s1_sq   = float(np.mean(d1 ** 2) - np.mean(d1) ** 2)

    prob1   = np.arange(1, n1) / n1
    q0_at_1 = np.quantile(scores0, prob1)
    incr2   = (np.abs(x1_sort[1:]  - q0_at_1) ** p
             - np.abs(x1_sort[:-1] - q0_at_1) ** p)
    d2      = np.cumsum(incr2)
    s2_sq   = float(np.mean(d2 ** 2) - np.mean(d2) ** 2)

    sigma_hat = np.sqrt(0.5 * s1_sq + 0.5 * s2_sq)
    n_eff     = (n0 * n1) / (n0 + n1)
    se        = sigma_hat / np.sqrt(n_eff)
    z_ci      = norm.ppf(1 - alpha / 2)
    z_test    = norm.ppf(alpha)

    lower = Wp_p - z_ci * se
    upper = Wp_p + z_ci * se

    test_stat = ((np.sqrt(n_eff) / sigma_hat) * (Wp_p - delta0 ** p)
                 if sigma_hat > 0 else np.inf)
    reject = int(test_stat < z_test)
    return lower, Wp_p, upper, reject


def wasserstein_from_model(modelo, X_test, X_col_names,
                            sensitive_name="Sex", p=2,
                            delta0=0.5, alpha=0.05):
    X_test = np.asarray(X_test, dtype=np.float32)
    idx_s  = X_col_names.index(sensitive_name)
    S      = X_test[:, idx_s].astype(int)
    scores = modelo.predict(X_test, verbose=0).flatten()
    s0, s1 = scores[S == 0], scores[S == 1]
    if len(s0) == 0 or len(s1) == 0:
        return np.nan, np.nan, np.nan, np.nan
    return wasserstein_ci(s0, s1, p=p, delta0=delta0, alpha=alpha)


# =============================================================================
# Wasserstein distance computed directly on feature matrix X
# =============================================================================

def wasserstein_features(X, X_col_names, sensitive_name="Sex", p=2):

    X = np.asarray(X, dtype=float)
    idx_s = X_col_names.index(sensitive_name)
    S = X[:, idx_s].astype(int)
    mask0 = S == 0
    mask1 = S == 1
    X0 = X[mask0]
    X1 = X[mask1]

    if len(X0) == 0 or len(X1) == 0:
        return np.nan, {}

    per_col = {}
    for j, col in enumerate(X_col_names):
        if col == sensitive_name:
            continue
        a = X0[:, j]
        b = X1[:, j]
        # 1-D Wasserstein via sorted quantile integral
        n_a, n_b = len(a), len(b)
        grid = np.unique(np.concatenate([
            np.linspace(0, 1, n_a + 1),
            np.linspace(0, 1, n_b + 1),
        ]))
        qa = np.quantile(a, grid)
        qb = np.quantile(b, grid)
        wp = float(np.sum(np.diff(grid) * np.abs(qa[:-1] - qb[:-1]) ** p))
        per_col[col] = wp

    mean_wp = float(np.mean(list(per_col.values()))) if per_col else np.nan
    return mean_wp, per_col

# =============================================================================
# RANDOM REPAIR  (del Barrio et al. 2019, Section 4.2)
# =============================================================================

def random_repair_X(X, X_col_names, sensitive_name="Sex", lam=0.5, p=2):
    """
    Apply Random Repair to a SINGLE dataset array X.
    λ=0  → no repair (original data kept).
    λ=1  → full repair (every point moved to its OT barycenter).
    """
    X     = np.asarray(X, dtype=np.float32)
    X_rep = X.copy()

    # λ=0 short-circuit: no repair needed
    if lam == 0.0:
        return X_rep

    idx_s       = X_col_names.index(sensitive_name)
    S           = X[:, idx_s].astype(int)
    idx0_global = np.where(S == 0)[0]
    idx1_global = np.where(S == 1)[0]
    X0, X1      = X[idx0_global], X[idx1_global]
    n0, n1      = len(idx0_global), len(idx1_global)

    if n0 == 0 or n1 == 0:
        return X_rep

    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)

    C  = cdist(X0, X1, metric="euclidean") ** p
    T  = ot.emd(np.ones(n0) / n0, np.ones(n1) / n1, C)

    nz_i, nz_j = np.nonzero(T)
    edges0 = {i: [] for i in range(n0)}
    edges1 = {j: [] for j in range(n1)}

    for k in range(len(nz_i)):
        loc_i = int(nz_i[k])
        loc_j = int(nz_j[k])
        w     = float(T[loc_i, loc_j])
        bary  = (1 - lam) * X0[loc_i] + lam * X1[loc_j]
        edges0[loc_i].append((w, bary))
        edges1[loc_j].append((w, bary))

    B0 = np.random.binomial(1, lam, size=n0)
    B1 = np.random.binomial(1, lam, size=n1)

    for loc_i, flag in enumerate(B0):
        if flag == 1 and edges0[loc_i]:
            partners = edges0[loc_i]
            if len(partners) == 1:
                bary = partners[0][1]
            else:
                ws = np.array([w for w, _ in partners])
                k  = np.random.choice(len(partners), p=ws / ws.sum())
                bary = partners[k][1]
            X_rep[idx0_global[loc_i], :] = bary

    for loc_j, flag in enumerate(B1):
        if flag == 1 and edges1[loc_j]:
            partners = edges1[loc_j]
            if len(partners) == 1:
                bary = partners[0][1]
            else:
                ws = np.array([w for w, _ in partners])
                k  = np.random.choice(len(partners), p=ws / ws.sum())
                bary = partners[k][1]
            X_rep[idx1_global[loc_j], :] = bary

    return X_rep


def repair_per_client(repartos, X_col_names, sensitive_name, lam, p=2):
    """
    Apply random repair to each client shard independently (FL-private).
    """
    repaired = []
    for cl in repartos:
        X_cl_rep = random_repair_X(
            cl["X"], X_col_names,
            sensitive_name=sensitive_name,
            lam=lam, p=p
        )
        repaired.append({"X": X_cl_rep, "y": cl["y"]})
    return repaired


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def plot_box_metrics(accs_fed, di_fed, spd_fed, eod_fed,
                     accs_cent, di_cent, spd_cent, eod_cent,
                     accs_fed_rr, di_fed_rr,
                     accs_cent_rr, di_cent_rr,
                     n_reps, nombre_exp):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'Fed':      '#1f77b4',
        'Cent':     '#ff7f0e',
        'Fed+RR':   '#1f77b4',
        'Cent+RR':  '#ff7f0e',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{nombre_exp}  –  {n_reps} repetitions", fontsize=14, fontweight='normal')

    subplots_config = {
        "Accuracy": ([accs_fed, accs_cent, accs_fed_rr, accs_cent_rr],
                     ["Fed", "Cent", "Fed+RR", "Cent+RR"]),
        "DI":       ([di_fed,   di_cent,   di_fed_rr,   di_cent_rr],
                     ["Fed", "Cent", "Fed+RR", "Cent+RR"]),
        "SPD":      ([spd_fed,  spd_cent,  [],          []],
                     ["Fed", "Cent", "",     ""]),
        "EOD":      ([eod_fed,  eod_cent,  [],          []],
                     ["Fed", "Cent", "",     ""]),
    }

    for ax, (title, (data, labels)) in zip(axes.flatten(), subplots_config.items()):
        clean = [(d, l) for d, l in zip(data, labels) if len(d) > 0]
        if not clean:
            ax.set_visible(False)
            continue

        bp = ax.boxplot([d for d, _ in clean],
                        labels=[l for _, l in clean],
                        patch_artist=True,
                        showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.6))

        for patch, label in zip(bp['boxes'], [l for _, l in clean]):
            patch.set_facecolor(colors.get(label, '#cccccc'))
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)

        ax.set_title(title, fontsize=12, fontweight='normal')
        ax.set_ylabel(title if title != "Accuracy" else None)
        ax.tick_params(axis='x', labelsize=10, rotation=0)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.6, color='#888888', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor('white')

    plt.tight_layout()
    os.makedirs("Figuras", exist_ok=True)
    ruta = os.path.join("Figuras", f"Resultados_{nombre_exp}.png")
    plt.savefig(ruta, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close()
    return ruta


def plot_loss_convergence(hist_losses, nombre_exp):
    plt.figure(figsize=(8, 4))
    for i, losses in enumerate(hist_losses):
        plt.plot(range(1, len(losses) + 1), losses, alpha=0.4,
                 label=f"Rep {i+1}")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss convergence - {nombre_exp}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Figuras", exist_ok=True)
    ruta = os.path.join("Figuras", f"Convergencia_{nombre_exp}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    return ruta


# =============================================================================
#  Plot 1: λ vs Accuracy & Fairness (tradeoff curve)
# =============================================================================
def plot_lambda_tradeoff(sweep_results, lambdas, nombre_exp):
    """
    sweep_results : list of dicts, one per lambda value.

    Now plots:
      1) Accuracy
      2) Disparate Impact
      3) Feature-space Wasserstein distance
    """

    def mean_ci(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2:
            return np.nan, 0.0
        lo, hi = calcular_intervalo_confianza(vals)
        return np.mean(vals), (hi - lo) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"λ Sweep — Fairness–Accuracy Tradeoff  ({nombre_exp})",
        fontsize=13
    )

    # ------------------------------------------------------------------
    # 1) Accuracy
    # ------------------------------------------------------------------
    ax = axes[0]

    m_fed, e_fed = zip(*[
        mean_ci(sr["acc_fed_rr"]) for sr in sweep_results
    ])
    m_cent, e_cent = zip(*[
        mean_ci(sr["acc_cent_rr"]) for sr in sweep_results
    ])

    ax.errorbar(
        lambdas, m_fed, yerr=e_fed, fmt='-o',
        color='#1f77b4', label='Fed+RR',
        capsize=4, linewidth=2
    )
    ax.errorbar(
        lambdas, m_cent, yerr=e_cent, fmt='-s',
        color='#ff7f0e', label='Cent+RR',
        capsize=4, linewidth=2
    )

    ax.set_xlabel("λ (repair intensity)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy", fontsize=11)
    ax.set_xticks(lambdas)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')

    # ------------------------------------------------------------------
    # 2) Disparate Impact
    # ------------------------------------------------------------------
    ax = axes[1]

    m_fed, e_fed = zip(*[
        mean_ci(sr["di_fed_rr"]) for sr in sweep_results
    ])
    m_cent, e_cent = zip(*[
        mean_ci(sr["di_cent_rr"]) for sr in sweep_results
    ])

    ax.errorbar(
        lambdas, m_fed, yerr=e_fed, fmt='-o',
        color='#1f77b4', label='Fed+RR',
        capsize=4, linewidth=2
    )
    ax.errorbar(
        lambdas, m_cent, yerr=e_cent, fmt='-s',
        color='#ff7f0e', label='Cent+RR',
        capsize=4, linewidth=2
    )

    ax.set_xlabel("λ (repair intensity)", fontsize=11)
    ax.set_ylabel("DI", fontsize=11)
    ax.set_title("Disparate Impact (→1)", fontsize=11)
    ax.set_xticks(lambdas)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')

    # ------------------------------------------------------------------
    # 3) Wasserstein on features
    # ------------------------------------------------------------------
    ax = axes[2]

    before_vals = [
        v for sr in sweep_results
        for v in sr["wfeat_before"]
        if not np.isnan(v)
    ]

    if before_vals:
        ax.axhline(
            np.mean(before_vals),
            linestyle=':',
            color='gray',
            linewidth=1.8,
            label='Before repair (λ=0)'
        )

    m_fed, e_fed = zip(*[
        mean_ci(sr["wfeat_fed_rr"]) for sr in sweep_results
    ])
    m_cent, e_cent = zip(*[
        mean_ci(sr["wfeat_cent_rr"]) for sr in sweep_results
    ])

    ax.errorbar(
        lambdas, m_fed, yerr=e_fed, fmt='-o',
        color='#1f77b4', label='Fed+RR',
        capsize=4, linewidth=2
    )
    ax.errorbar(
        lambdas, m_cent, yerr=e_cent, fmt='-s',
        color='#ff7f0e', label='Cent+RR',
        capsize=4, linewidth=2
    )

    ax.set_xlabel("λ (repair intensity)", fontsize=11)
    ax.set_ylabel("Mean Wp feature distance", fontsize=11)
    ax.set_title("Feature Wasserstein", fontsize=11)
    ax.set_xticks(lambdas)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')

    plt.tight_layout()
    os.makedirs("Figuras", exist_ok=True)

    ruta = os.path.join(
        "Figuras",
        f"Lambda_Tradeoff_{nombre_exp}.png"
    )

    plt.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight",
        facecolor='white'
    )
    plt.close()

    return ruta



# =============================================================================
#  Plot 2: λ vs Wasserstein distance on X (feature distribution)
# =============================================================================

def plot_lambda_wasserstein_features(sweep_results, lambdas, nombre_exp):
    """
    sweep_results : list of dicts, each with lists
                    'wfeat_before' and 'wfeat_fed_rr' and 'wfeat_cent_rr'
    """
    def mean_ci(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2:
            return np.nan, 0.0
        lo, hi = calcular_intervalo_confianza(vals)
        return np.mean(vals), (hi - lo) / 2

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"λ vs Feature Distribution Alignment  ({nombre_exp})",
                 fontsize=13)

    # "before" is the same for every λ (unrepaired), so plot as horizontal
    before_vals = [v for sr in sweep_results for v in sr["wfeat_before"]
                   if not np.isnan(v)]
    if before_vals:
        ax.axhline(np.mean(before_vals), linestyle=':', color='gray',
                   linewidth=1.8, label='Before repair (λ=0)')

    m_fed,  e_fed  = zip(*[mean_ci(sr["wfeat_fed_rr"])  for sr in sweep_results])
    m_cent, e_cent = zip(*[mean_ci(sr["wfeat_cent_rr"]) for sr in sweep_results])

    ax.errorbar(lambdas, m_fed,  yerr=e_fed,  fmt='-o',
                color='#1f77b4', label='Fed+RR',  capsize=4, linewidth=2)
    ax.errorbar(lambdas, m_cent, yerr=e_cent, fmt='-s',
                color='#ff7f0e', label='Cent+RR', capsize=4, linewidth=2)

    ax.set_xlabel("λ (repair intensity)", fontsize=11)
    ax.set_ylabel("Mean Wp feature distance", fontsize=11)
    ax.set_xticks(lambdas)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')

    plt.tight_layout()
    os.makedirs("Figuras", exist_ok=True)
    ruta = os.path.join("Figuras", f"Lambda_WassFeatures_{nombre_exp}.png")
    plt.savefig(ruta, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close()
    return ruta


# =============================================================================
# MAIN PIPELINE
# =============================================================================

BINARIZACIONES = {
    "Sex":      {"nombre_grupo1": "Male",    "nombre_grupo0": "Female"},
    "OrigEthn": {"nombre_grupo1": "CaucYes", "nombre_grupo0": "CaucNo"},
    "Target":   {"nombre_grupo1": ">50K",    "nombre_grupo0": "<=50K"},
}

# NEW: canonical set of λ values to sweep over.
# λ=0  → no repair (baseline).
# λ=1  → full repair.
# The intermediate values trace the fairness–accuracy tradeoff curve.
LAMBDA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]

def algoritmo_principal(
    VARIABLE_SENSIBLE,
    MODELO,
    PORCENTAJE_TEST,
    N_REPETICIONES,
    NUM_CLIENTES,
    N_ROUNDS,
    MAX_DATOS,
    MODO_SESGO_CLIENTES,
    C_MIN,
    C_MAX,
    MODO_REPARTO,
    BETA=0.5,
    REPAIR_P=2,
    FICHERO_PROPORCIONES="proporciones_clientes.csv",
    nombre_exp="Exp",
    lambda_sweep=None,
):
    if lambda_sweep is None:
        lambda_sweep = LAMBDA_SWEEP

    # 1. Load data
    df = cargar_datos(
        "./adult_dataset/adult.data.csv",
        "./adult_dataset/adult.test.csv",
    )

    # 2. Pre-process
    if MODO_REPARTO == 1 and VARIABLE_SENSIBLE == "Sex":
        df_m   = df[df["Sex"] == "Male"].sample(n=5000, random_state=0)
        df_f   = df[df["Sex"] == "Female"].sample(n=5000, random_state=0)
        df_use = pd.concat([df_m, df_f]).sample(frac=1, random_state=0).reset_index(drop=True)
    elif MODO_REPARTO == 1 and VARIABLE_SENSIBLE == "OrigEthn":
        df_w   = df[df["OrigEthn"] == "White"].sample(n=5000, random_state=0)
        df_nw  = df[df["OrigEthn"] != "White"].sample(n=5000, random_state=0)
        df_use = pd.concat([df_w, df_nw]).sample(frac=1, random_state=0).reset_index(drop=True)
    elif MODO_REPARTO == 2:
        df_hi  = df[df["Target"].isin([">50K", ">50K."])].sample(n=5000, random_state=0)
        df_lo  = df[df["Target"].isin(["<=50K", "<=50K."])].sample(n=5000, random_state=0)
        df_use = pd.concat([df_hi, df_lo]).sample(frac=1, random_state=0).reset_index(drop=True)
    elif MODO_REPARTO == 3:                                          # ← NEW
        df_hi  = df[df["Target"].isin([">50K", ">50K."])].sample(n=5000, random_state=0)
        df_lo  = df[df["Target"].isin(["<=50K", "<=50K."])].sample(n=5000, random_state=0)
        df_use = pd.concat([df_hi, df_lo]).sample(frac=1, random_state=0).reset_index(drop=True)
    else:
        df_use = df

    vs_preproc = "Sex" if MODO_REPARTO in (2, 3) else VARIABLE_SENSIBLE  # ← MODIFIED (added 3)
    X, y, X_col_names = preprocesar_datos(df_use, vs_preproc)

    # 3. Initialise CSV
    with open(FICHERO_PROPORCIONES, mode="w", newline="") as f:
        csv.writer(f).writerow(["repeticion", "cliente", "tipo", "proporcion"])

    sweep_results = [
        {
            "acc_fed_rr":    [],
            "di_fed_rr":     [],
            "spd_fed_rr":    [],
            "acc_cent_rr":   [],
            "di_cent_rr":    [],
            "spd_cent_rr":   [],
            "wfeat_before":  [],
            "wfeat_fed_rr":  [],
            "wfeat_cent_rr": [],
            "spred_before":  [],
            "spred_fed_rr":  [],
            "spred_cent_rr": [],
        }
        for _ in lambda_sweep
    ]

    resultados  = []
    hist_losses = []

    # 4. Repetition loop
    for rep in range(N_REPETICIONES):
        np.random.seed(rep)
        print(f"\n{'='*60}")
        print(f"  {nombre_exp}  -  repetition {rep+1}/{N_REPETICIONES}")
        print(f"{'='*60}")

        idx   = np.random.choice(len(X), size=min(MAX_DATOS, len(X)), replace=False)
        X_rep = X[idx]
        y_rep = y[idx]

        C = random.uniform(C_MIN, C_MAX)

        # ── Partitioning ──────────────────────────────────────────────────
        if MODO_REPARTO in (0, 1):
            repartos = repartir_datos(
                X_rep, y_rep, VARIABLE_SENSIBLE,
                modo_reparto=MODO_REPARTO,
                num_clientes=NUM_CLIENTES,
                porc_min=0.01, porc_max=0.99,
                X_col_names=X_col_names,
            )
        elif MODO_REPARTO == 2:
            repartos = repartir_datos_noiid_target(
                X_rep, y_rep,
                num_clientes=NUM_CLIENTES,
                porc_min=0.05, porc_max=0.95,
            )
        elif MODO_REPARTO == 3:
            repartos = repartir_datos_noiid_target_dirichlet(
                X_rep, y_rep,
                num_clientes=NUM_CLIENTES,
                beta=BETA,
            )




        # ── Log proportions ───────────────────────────────────────────────
        with open(FICHERO_PROPORCIONES, mode="a", newline="") as f:
            writer = csv.writer(f)
            if MODO_REPARTO in (2, 3):                               # ← MODIFIED (added 3)
                for i, cl in enumerate(repartos):
                    prop = float(np.sum(cl["y"])) / cl["y"].shape[0]
                    writer.writerow([rep + 1, i, "Y=1", f"{prop:.4f}"])
            else:
                idx_vs = X_col_names.index(VARIABLE_SENSIBLE)
                g1     = BINARIZACIONES[VARIABLE_SENSIBLE]["nombre_grupo1"]
                for i, cl in enumerate(repartos):
                    prop = float(np.sum(cl["X"][:, idx_vs])) / cl["X"].shape[0]
                    writer.writerow([rep + 1, i, g1, f"{prop:.4f}"])

        if len(repartos) > NUM_CLIENTES:
            repartos = repartos[:NUM_CLIENTES]

        X_total = np.vstack([cl["X"] for cl in repartos])
        y_total = np.vstack([cl["y"] for cl in repartos])
        X_train, X_test, y_train, y_test = train_test_split(
            X_total, y_total, test_size=PORCENTAJE_TEST
        )
        X_test = X_test.astype(np.float32)

        # A. Standard federated model (no repair)
        print("  Training federated model...")
        dict_fed = entrenar_federado(
            repartos, VARIABLE_SENSIBLE,
            rounds=N_ROUNDS, c=C,
            X_col_names=X_col_names,
            modo_sesgo=MODO_SESGO_CLIENTES,
        )
        hist_losses.append([h["Loss"] for h in dict_fed["hist"]])
        acc_fed, di_fed, spd_fed, eod_fed, ic_di_fed = evaluar_modelo(
            dict_fed["modelo"], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
        )
        W_fed = {}
        for p_order in [1, 2, 3]:
            W_fed[p_order] = wasserstein_from_model(
                dict_fed["modelo"], X_test, X_col_names,
                sensitive_name="Sex", p=p_order, delta0=0.5,
            )
        print(f"  Fed   Acc:{acc_fed:.3f}  DI:{di_fed:.3f}")

        # B. Standard centralised model (no repair)
        print("  Training centralised model...")
        modelo_cent = crear_modelo(MODELO, X_train.shape[1])
        dict_cent   = entrenar_centralizado(X_train, y_train, modelo_cent, epochs=64)
        acc_cent, di_cent, spd_cent, eod_cent, ic_di_cent = evaluar_modelo(
            dict_cent["modelo"], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
        )
        W_cent = {}
        for p_order in [1, 2, 3]:
            W_cent[p_order] = wasserstein_from_model(
                dict_cent["modelo"], X_test, X_col_names,
                sensitive_name="Sex", p=p_order, delta0=0.5,
            )
        print(f"  Cent  Acc:{acc_cent:.3f}  DI:{di_cent:.3f}")

        # ── λ sweep ───────────────────────────────────────────────────────
        wfeat_before, _ = wasserstein_features(
            X_train, X_col_names,
            sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
        )
        spred_before = s_predictability(
            X_train, X_test, X_col_names,
            sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
        )

        for lam_idx, lam in enumerate(lambda_sweep):
            print(f"  λ={lam:.2f}: running repair variants...")

            # Centralised + repair
            X_train_rr = random_repair_X(
                X_train, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE,
                lam=lam, p=REPAIR_P,
            )
            modelo_cent_rr = crear_modelo(MODELO, X_train_rr.shape[1])
            dict_cent_rr   = entrenar_centralizado(X_train_rr, y_train, modelo_cent_rr, epochs=64)
            acc_cent_rr, di_cent_rr, spd_cent_rr, eod_cent_rr, _ = evaluar_modelo(
                dict_cent_rr["modelo"], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
            )
            wfeat_cent_rr, _ = wasserstein_features(
                X_train_rr, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
            )
            spred_cent_rr = s_predictability(
                X_train_rr, X_test, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
            )

            # Federated + repair
            repartos_rr = repair_per_client(
                repartos, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE,
                lam=lam, p=REPAIR_P,
            )
            if len(repartos_rr) > NUM_CLIENTES:
                repartos_rr = repartos_rr[:NUM_CLIENTES]

            dict_fed_rr = entrenar_federado(
                repartos_rr, VARIABLE_SENSIBLE,
                rounds=N_ROUNDS, c=C,
                X_col_names=X_col_names,
                modo_sesgo=MODO_SESGO_CLIENTES,
            )
            acc_fed_rr, di_fed_rr, spd_fed_rr, eod_fed_rr, _ = evaluar_modelo(
                dict_fed_rr["modelo"], X_test, y_test, VARIABLE_SENSIBLE, X_col_names
            )
            X_clients_rr = np.vstack([cl["X"] for cl in repartos_rr])
            wfeat_fed_rr, _ = wasserstein_features(
                X_clients_rr, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
            )
            spred_fed_rr = s_predictability(
                X_clients_rr, X_test, X_col_names,
                sensitive_name=VARIABLE_SENSIBLE if VARIABLE_SENSIBLE != "Target" else "Sex"
            )

            sr = sweep_results[lam_idx]
            sr["acc_fed_rr"].append(acc_fed_rr)
            sr["di_fed_rr"].append(di_fed_rr)
            sr["spd_fed_rr"].append(spd_fed_rr)
            sr["acc_cent_rr"].append(acc_cent_rr)
            sr["di_cent_rr"].append(di_cent_rr)
            sr["spd_cent_rr"].append(spd_cent_rr)
            sr["wfeat_before"].append(wfeat_before)
            sr["wfeat_fed_rr"].append(wfeat_fed_rr)
            sr["wfeat_cent_rr"].append(wfeat_cent_rr)
            sr["spred_before"].append(spred_before)
            sr["spred_fed_rr"].append(spred_fed_rr)
            sr["spred_cent_rr"].append(spred_cent_rr)

            print(f"    Fed+RR  Acc:{acc_fed_rr:.3f} DI:{di_fed_rr:.3f} "
                  f"WFeat:{wfeat_fed_rr:.4f} SPred:{spred_fed_rr:.3f}")
            print(f"    Cent+RR Acc:{acc_cent_rr:.3f} DI:{di_cent_rr:.3f} "
                  f"WFeat:{wfeat_cent_rr:.4f} SPred:{spred_cent_rr:.3f}")

        mid_idx = min(range(len(lambda_sweep)),
                      key=lambda i: abs(lambda_sweep[i] - 0.5))
        sr_mid  = sweep_results[mid_idx]

        resultados.append({
            "acc_fed":     acc_fed,
            "di_fed":      di_fed,
            "spd_fed":     spd_fed,
            "eod_fed":     eod_fed,
            "W1_fed":      W_fed[1][1],
            "W2_fed":      W_fed[2][1],
            "W3_fed":      W_fed[3][1],
            "W2_fed_lower":  W_fed[2][0],
            "W2_fed_upper":  W_fed[2][2],
            "W2_fed_reject": W_fed[2][3],
            "acc_cent":    acc_cent,
            "di_cent":     di_cent,
            "spd_cent":    spd_cent,
            "eod_cent":    eod_cent,
            "W1_cent":     W_cent[1][1],
            "W2_cent":     W_cent[2][1],
            "W3_cent":     W_cent[3][1],
            "W2_cent_lower":  W_cent[2][0],
            "W2_cent_upper":  W_cent[2][2],
            "W2_cent_reject": W_cent[2][3],
            "acc_cent_rr":  sr_mid["acc_cent_rr"][-1],
            "di_cent_rr":   sr_mid["di_cent_rr"][-1],
            "spd_cent_rr":  sr_mid["spd_cent_rr"][-1],
            "eod_cent_rr":  0.0,
            "acc_fed_rr":   sr_mid["acc_fed_rr"][-1],
            "di_fed_rr":    sr_mid["di_fed_rr"][-1],
            "spd_fed_rr":   sr_mid["spd_fed_rr"][-1],
            "eod_fed_rr":   0.0,
            "W2_cent_rr":   np.nan,
            "W2_cent_rr_lower": np.nan,
            "W2_cent_rr_upper": np.nan,
            "W2_fed_rr":    np.nan,
            "W2_fed_rr_lower":  np.nan,
            "W2_fed_rr_upper":  np.nan,
        })

    # 5. Summary
    def ci(key):
        vals = [r[key] for r in resultados if not np.isnan(r[key])]
        return calcular_intervalo_confianza(vals) if len(vals) > 1 else (np.nan, np.nan)

    print("\n" + "="*60)
    print(f"  SUMMARY  -  {nombre_exp}")
    print("="*60)
    for label, ak, dk in [
        ("Federated",        "acc_fed",     "di_fed"),
        ("Centralised",      "acc_cent",    "di_cent"),
        ("Centralised + RR", "acc_cent_rr", "di_cent_rr"),
        ("Federated   + RR", "acc_fed_rr",  "di_fed_rr"),
    ]:
        print(f"  {label:<20}  Acc:{ci(ak)}  DI:{ci(dk)}")

    print("\n  λ sweep summary:")
    for lam, sr in zip(lambda_sweep, sweep_results):
        def m(k):
            v = [x for x in sr[k] if not np.isnan(x)]
            return np.mean(v) if v else float('nan')
        print(f"    λ={lam:.2f}  Fed+RR: Acc={m('acc_fed_rr'):.3f} "
              f"DI={m('di_fed_rr'):.3f} WFeat={m('wfeat_fed_rr'):.4f} "
              f"SPred={m('spred_fed_rr'):.3f}")

    # 6. Plots

    ruta_box  = plot_box_metrics(
        [r["acc_fed"]     for r in resultados],
        [r["di_fed"]      for r in resultados],
        [r["spd_fed"]     for r in resultados],
        [r["eod_fed"]     for r in resultados],
        [r["acc_cent"]    for r in resultados],
        [r["di_cent"]     for r in resultados],
        [r["spd_cent"]    for r in resultados],
        [r["eod_cent"]    for r in resultados],
        [r["acc_fed_rr"]  for r in resultados],
        [r["di_fed_rr"]   for r in resultados],
        [r["acc_cent_rr"] for r in resultados],
        [r["di_cent_rr"]  for r in resultados],
        N_REPETICIONES, nombre_exp,
    )
    ruta_loss     = plot_loss_convergence(hist_losses, nombre_exp)
    ruta_tradeoff = plot_lambda_tradeoff(sweep_results, lambda_sweep, nombre_exp)
    #ruta_wfeat    = plot_lambda_wasserstein_features(sweep_results, lambda_sweep, nombre_exp)
    ruta_spred    = plot_lambda_s_predictability(sweep_results, lambda_sweep, nombre_exp)
    ruta_bar      = plot_model_comparison_bar(resultados, nombre_exp)

    print(f"\n  Figures saved:")
    print(f"    {ruta_box}")
    print(f"    {ruta_loss}")
    print(f"    {ruta_tradeoff}")
    print(f"    {ruta_wfeat}")
    print(f"    {ruta_spred}")
    print(f"    {ruta_bar}")

    return {
        "resultados":     resultados,
        "sweep_results":  sweep_results,
        "lambda_sweep":   lambda_sweep,
        "ci_acc_fed":     ci("acc_fed"),
        "ci_di_fed":      ci("di_fed"),
        "ci_acc_cent":    ci("acc_cent"),
        "ci_di_cent":     ci("di_cent"),
        "ci_acc_cent_rr": ci("acc_cent_rr"),
        "ci_di_cent_rr":  ci("di_cent_rr"),
        "ci_acc_fed_rr":  ci("acc_fed_rr"),
        "ci_di_fed_rr":   ci("di_fed_rr"),
    }
