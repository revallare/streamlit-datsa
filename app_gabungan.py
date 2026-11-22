# =============================================================================
#  APLIKASI STREAMLIT TERPADU - ANALISIS PERFORMA SISWA (student-mat)
#  Satu file berisi: pemuatan data -> pelatihan model regresi & clustering
#  (train-test split, pemilihan algoritma, tuning, evaluasi, interpretasi) -> aplikasi.
#  Model dilatih otomatis saat aplikasi pertama dibuka lalu di-cache,
#  jadi TIDAK perlu menjalankan kode Colab terpisah.
#
#  Struktur repo untuk deploy (Streamlit Community Cloud / lokal):
#      app.py                 <- file ini
#      student-mat.xlsx       <- dataset (jika tidak ada, aplikasi meminta upload)
#      requirements.txt       <- lihat catatan di bawah
#
#  requirements.txt:
#      streamlit
#      pandas
#      numpy
#      scipy
#      scikit-learn
#      matplotlib
#      seaborn
#      joblib
#      openpyxl
#
#  Jalankan lokal:  streamlit run app.py
# =============================================================================
import io
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import kruskal

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_absolute_error, mean_squared_error, median_absolute_error,
                             r2_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (RandomizedSearchCV, RepeatedKFold, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Analisis Performa Siswa", page_icon="🎓", layout="wide")

BASE = Path(__file__).resolve().parent
TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PASS_THRESHOLD = 10

# Pengaturan kecepatan pelatihan. "Cepat" cocok untuk server gratis, "Lengkap" = setara kode Colab.
MODE_CFG = {
    "Cepat": dict(cv_repeats=1, n_iter=8, n_tune=2, perm_repeats=10, boot=10, learning_curve=False,
                  trees=150, km_init=10, gmm_init=2, final_init=20),
    "Lengkap": dict(cv_repeats=2, n_iter=25, n_tune=3, perm_repeats=30, boot=30, learning_curve=True,
                    trees=300, km_init=20, gmm_init=5, final_init=50),
}

# -----------------------------------------------------------------------------
# METADATA FITUR (label, kelompok, bantuan) - sesuai dokumentasi dataset UCI
# -----------------------------------------------------------------------------
LABELS = {
    "school": "Sekolah", "sex": "Jenis kelamin", "age": "Usia", "address": "Tempat tinggal",
    "famsize": "Ukuran keluarga", "Pstatus": "Status orang tua", "Medu": "Pendidikan ibu",
    "Fedu": "Pendidikan ayah", "Mjob": "Pekerjaan ibu", "Fjob": "Pekerjaan ayah",
    "reason": "Alasan memilih sekolah", "guardian": "Wali", "traveltime": "Waktu tempuh ke sekolah",
    "studytime": "Waktu belajar mingguan", "failures": "Jumlah kegagalan kelas sebelumnya",
    "schoolsup": "Bimbingan tambahan sekolah", "famsup": "Dukungan belajar keluarga",
    "paid": "Les berbayar (Matematika)", "activities": "Kegiatan ekstrakurikuler",
    "nursery": "Pernah ke taman kanak-kanak", "higher": "Ingin kuliah", "internet": "Akses internet di rumah",
    "romantic": "Menjalin hubungan romantis", "famrel": "Kualitas hubungan keluarga",
    "freetime": "Waktu luang setelah sekolah", "goout": "Frekuensi keluar bersama teman",
    "Dalc": "Konsumsi alkohol hari kerja", "Walc": "Konsumsi alkohol akhir pekan",
    "health": "Kondisi kesehatan", "absences": "Jumlah absen", "G1": "Nilai periode 1 (G1)",
    "G2": "Nilai periode 2 (G2)", "G3": "Nilai akhir (G3)",
}
HELP = {
    "Medu": "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi",
    "Fedu": "0 = tidak sekolah, 1 = SD, 2 = SMP, 3 = SMA, 4 = pendidikan tinggi",
    "traveltime": "1 = <15 menit, 2 = 15-30 menit, 3 = 30-60 menit, 4 = >60 menit",
    "studytime": "1 = <2 jam, 2 = 2-5 jam, 3 = 5-10 jam, 4 = >10 jam per minggu",
    "failures": "Jumlah kegagalan kelas sebelumnya (3 berarti 3 atau lebih)",
    "famrel": "1 = sangat buruk ... 5 = sangat baik", "freetime": "1 = sangat sedikit ... 5 = sangat banyak",
    "goout": "1 = sangat jarang ... 5 = sangat sering", "Dalc": "1 = sangat rendah ... 5 = sangat tinggi",
    "Walc": "1 = sangat rendah ... 5 = sangat tinggi", "health": "1 = sangat buruk ... 5 = sangat baik",
    "G1": "Skala 0-20", "G2": "Skala 0-20",
    "G3": "Skala 0-20. Hanya dipakai model clustering (bila fitur nilai diaktifkan), bukan model regresi.",
    "school": "GP = Gabriel Pereira, MS = Mousinho da Silveira",
}
VALUE_LABELS = {
    "sex": {"F": "Perempuan", "M": "Laki-laki"},
    "address": {"U": "Perkotaan", "R": "Pedesaan"},
    "famsize": {"LE3": "3 anggota atau kurang", "GT3": "Lebih dari 3 anggota"},
    "Pstatus": {"T": "Tinggal bersama", "A": "Terpisah"},
    "guardian": {"mother": "Ibu", "father": "Ayah", "other": "Lainnya"},
    "reason": {"home": "Dekat rumah", "reputation": "Reputasi sekolah", "course": "Pilihan program", "other": "Lainnya"},
    "Mjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Pelayanan publik", "at_home": "Di rumah", "other": "Lainnya"},
    "Fjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Pelayanan publik", "at_home": "Di rumah", "other": "Lainnya"},
}
for _c in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
    VALUE_LABELS[_c] = {"yes": "Ya", "no": "Tidak"}

GROUPS = {
    "👤 Demografi": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "👨‍👩‍👧 Keluarga": ["Medu", "Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"],
    "📚 Akademik": ["reason", "traveltime", "studytime", "failures", "schoolsup", "paid",
                    "higher", "nursery", "absences", "G1", "G2", "G3"],
    "🎯 Gaya hidup": ["activities", "internet", "romantic", "freetime", "goout", "Dalc", "Walc", "health"],
}


# =============================================================================
#  BAGIAN A - DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def read_data(content: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")  # CSV asli UCI memakai ';'
    else:
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name="student-mat")
        except ValueError:
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
    return df.drop_duplicates().reset_index(drop=True)


def get_raw_data():
    """Cari dataset di folder app; jika tidak ada minta upload. Mengembalikan (bytes, nama)."""
    for fname in ("student-mat.xlsx", "student-mat.csv"):
        p = BASE / fname
        if p.exists():
            return p.read_bytes(), fname
    st.warning("File `student-mat.xlsx` tidak ditemukan di folder aplikasi. Unggah dataset di bawah ini.")
    up = st.file_uploader("Unggah student-mat.xlsx (atau .csv)", type=["xlsx", "csv"], key="upload_dataset")
    if up is None:
        st.stop()
    return up.getvalue(), up.name


raw_bytes, raw_name = get_raw_data()
df_raw = read_data(raw_bytes, raw_name)
missing_cols = [c for c in LABELS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Dataset tidak valid, kolom hilang: {', '.join(missing_cols)}")
    st.stop()
DATA_KEY = f"{raw_name}-{df_raw.shape}-{int(pd.util.hash_pandas_object(df_raw, index=True).sum() % 10**12)}"


# =============================================================================
#  BAGIAN B - FUNGSI BANTU MODEL
# =============================================================================
def split_columns(X: pd.DataFrame):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=np.number).columns.tolist()
    return num, cat


def make_preprocessor(num_cols, cat_cols, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                                sparse_output=False))]), cat_cols),
    ])


def close_fig(fig):
    plt.close(fig)
    return fig


def show_fig(fig):
    st.pyplot(fig, clear_figure=False)


def regression_report(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
        "Adj. R2": 1 - (1 - r2) * (n - 1) / (n - n_features - 1),
        "Akurasi ±1 poin": float(np.mean(np.abs(np.asarray(y_true) - y_pred) <= 1)),
        "Akurasi ±2 poin": float(np.mean(np.abs(np.asarray(y_true) - y_pred) <= 2)),
    }


def eval_labels(Z, labels):
    if len(set(labels)) < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan)
    return dict(silhouette=silhouette_score(Z, labels),
                calinski=calinski_harabasz_score(Z, labels),
                davies=davies_bouldin_score(Z, labels))


# =============================================================================
#  BAGIAN C - PELATIHAN MODEL REGRESI (di-cache)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_regression(_df, data_key, use_prev_grades, mode):
    cfg = MODE_CFG[mode]
    drop_cols = [TARGET] + ([] if use_prev_grades else ["G1", "G2"])
    X = _df.drop(columns=drop_cols).copy()
    y = _df[TARGET].astype(float)
    num_cols, cat_cols = split_columns(X)
    X[num_cols] = X[num_cols].astype(float)

    # 1) train-test split (stratified menurut kuantil target)
    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins)

    # 2) pemilihan algoritma: benchmark dengan repeated k-fold CV
    def P(scale):
        return make_preprocessor(num_cols, cat_cols, scale)
    models = {
        "Baseline (rata-rata)": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("pre", P(True)), ("m", LinearRegression())]),
        "Ridge": Pipeline([("pre", P(True)), ("m", Ridge(alpha=10))]),
        "Lasso": Pipeline([("pre", P(True)), ("m", Lasso(alpha=0.05))]),
        "ElasticNet": Pipeline([("pre", P(True)), ("m", ElasticNet(alpha=0.05, l1_ratio=0.5))]),
        "KNN": Pipeline([("pre", P(True)), ("m", KNeighborsRegressor(n_neighbors=10))]),
        "SVR (RBF)": Pipeline([("pre", P(True)), ("m", SVR(C=5, epsilon=0.5))]),
        "Random Forest": Pipeline([("pre", P(False)), ("m", RandomForestRegressor(
            n_estimators=cfg["trees"], random_state=RANDOM_STATE, n_jobs=-1))]),
        "Extra Trees": Pipeline([("pre", P(False)), ("m", ExtraTreesRegressor(
            n_estimators=cfg["trees"], random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("pre", P(False)), ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("pre", P(False)), ("m", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]),
    }
    try:  # XGBoost dipakai otomatis bila terpasang
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([("pre", P(False)), ("m", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1))])
    except Exception:
        pass

    cv = RepeatedKFold(n_splits=5, n_repeats=cfg["cv_repeats"], random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows = []
    for name, mdl in models.items():
        r = cross_validate(mdl, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"Model": name, "CV R2": r["test_r2"].mean(), "CV R2 std": r["test_r2"].std(),
                     "CV MAE": -r["test_mae"].mean(), "CV RMSE": -r["test_rmse"].mean()})
    bench = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)

    # 3) pelatihan: tuning hyperparameter pada kandidat terbaik
    param_spaces = {
        "Ridge": {"m__alpha": np.logspace(-2, 3, 30)},
        "Lasso": {"m__alpha": np.logspace(-3, 0.5, 30)},
        "ElasticNet": {"m__alpha": np.logspace(-3, 0.5, 30), "m__l1_ratio": np.linspace(0.1, 0.9, 9)},
        "KNN": {"m__n_neighbors": list(range(3, 40)), "m__weights": ["uniform", "distance"], "m__p": [1, 2]},
        "SVR (RBF)": {"m__C": np.logspace(-1, 2, 20), "m__epsilon": [0.1, 0.3, 0.5, 1.0],
                      "m__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
        "Random Forest": {"m__n_estimators": [150, 300, 500], "m__max_depth": [None, 4, 6, 8, 12],
                          "m__min_samples_leaf": [1, 2, 4, 8], "m__max_features": ["sqrt", 0.3, 0.5, 1.0]},
        "Extra Trees": {"m__n_estimators": [150, 300, 500], "m__max_depth": [None, 4, 6, 8, 12],
                        "m__min_samples_leaf": [1, 2, 4, 8], "m__max_features": ["sqrt", 0.3, 0.5, 1.0]},
        "Gradient Boosting": {"m__n_estimators": [100, 200, 300, 500], "m__learning_rate": [0.01, 0.03, 0.05, 0.1],
                              "m__max_depth": [2, 3, 4], "m__subsample": [0.6, 0.8, 1.0],
                              "m__min_samples_leaf": [1, 3, 5, 10]},
        "HistGradientBoosting": {"m__learning_rate": [0.01, 0.03, 0.05, 0.1], "m__max_iter": [100, 200, 400],
                                 "m__max_depth": [None, 3, 5], "m__min_samples_leaf": [5, 10, 20],
                                 "m__l2_regularization": [0, 0.1, 1.0]},
        "XGBoost": {"m__n_estimators": [200, 300, 500], "m__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "m__max_depth": [2, 3, 4, 5], "m__subsample": [0.6, 0.8, 1.0],
                    "m__colsample_bytree": [0.5, 0.8, 1.0]},
    }
    tunable = [m for m in bench["Model"] if m in param_spaces][:cfg["n_tune"]]
    tuned, tune_rows = {}, []
    for name in tunable:
        search = RandomizedSearchCV(models[name], param_spaces[name], n_iter=cfg["n_iter"],
                                    scoring="neg_root_mean_squared_error", cv=5,
                                    random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        tune_rows.append({"Model": name, "CV RMSE (tuned)": -search.best_score_,
                          "Best params": str({k.replace("m__", ""): (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in search.best_params_.items()})})
    tune_df = pd.DataFrame(tune_rows).sort_values("CV RMSE (tuned)").reset_index(drop=True)
    best_name = tune_df.loc[0, "Model"]
    best_model = tuned[best_name]

    # 4) evaluasi pada data test
    pred_train = np.clip(best_model.predict(X_train), 0, 20)
    pred_test = np.clip(best_model.predict(X_test), 0, 20)
    rep = pd.DataFrame({
        "Train": regression_report(y_train, pred_train, X.shape[1]),
        "Test": regression_report(y_test, pred_test, X.shape[1]),
        "Baseline (Test)": regression_report(y_test, np.full(len(y_test), y_train.mean()), X.shape[1]),
    }).T
    resid = y_test.values - pred_test

    act_pass, pred_pass = (y_test >= PASS_THRESHOLD), (pred_test >= PASS_THRESHOLD)
    tp = int(((~act_pass) & (~pred_pass)).sum()); fn = int(((~act_pass) & pred_pass).sum())
    fp = int((act_pass & (~pred_pass)).sum());    tn = int((act_pass & pred_pass).sum())
    risk = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1)}

    # Gambar diagnostik
    n_panels = 4 if cfg["learning_curve"] else 3
    fig_diag, ax = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4))
    ax[0].scatter(y_test, pred_test, alpha=.7, c=np.abs(resid), cmap="coolwarm", edgecolor="k", lw=.3)
    ax[0].plot([0, 20], [0, 20], "k--"); ax[0].set_xlabel("Aktual"); ax[0].set_ylabel("Prediksi")
    ax[0].set_title(f"Aktual vs Prediksi ({best_name})")
    ax[1].scatter(pred_test, resid, alpha=.7, edgecolor="k", lw=.3, color="#4C72B0")
    ax[1].axhline(0, color="r", ls="--"); ax[1].set_xlabel("Prediksi"); ax[1].set_ylabel("Residual")
    ax[1].set_title("Plot residual (pola acak = baik)")
    sns.histplot(resid, kde=True, ax=ax[2], color="#55A868"); ax[2].set_title("Distribusi residual")
    if cfg["learning_curve"]:
        sizes, tr_s, va_s = learning_curve(best_model, X_train, y_train, cv=5, n_jobs=-1,
                                           scoring="neg_root_mean_squared_error",
                                           train_sizes=np.linspace(.2, 1, 6), random_state=RANDOM_STATE)
        ax[3].plot(sizes, -tr_s.mean(1), "o-", label="Train RMSE")
        ax[3].plot(sizes, -va_s.mean(1), "o-", label="Validasi RMSE")
        ax[3].set_xlabel("Jumlah data latih"); ax[3].set_ylabel("RMSE"); ax[3].legend()
        ax[3].set_title("Learning curve")
    fig_diag.tight_layout(); close_fig(fig_diag)

    fig_bench, axb = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=bench, y="Model", x="CV RMSE", palette="mako", ax=axb)
    axb.set_title("Perbandingan algoritma (RMSE lebih kecil = lebih baik)")
    fig_bench.tight_layout(); close_fig(fig_bench)

    # 5) interpretasi: permutation importance, PDP, koefisien
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=cfg["perm_repeats"],
                                  random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = (pd.DataFrame({"Fitur": X.columns, "Kenaikan RMSE bila diacak": perm.importances_mean,
                         "Std": perm.importances_std})
           .sort_values("Kenaikan RMSE bila diacak", ascending=False).reset_index(drop=True))
    top = imp.head(15).iloc[::-1]
    fig_imp, axi = plt.subplots(figsize=(8, 5.5))
    axi.barh(top["Fitur"], top["Kenaikan RMSE bila diacak"], xerr=top["Std"], color="#4C72B0")
    axi.set_xlabel("Kenaikan RMSE saat fitur diacak (makin besar = makin penting)")
    axi.set_title("Permutation Importance (data test)")
    fig_imp.tight_layout(); close_fig(fig_imp)

    top_num = [f for f in imp["Fitur"] if f in num_cols][:3]
    fig_pdp = None
    if top_num:
        fig_pdp, axp = plt.subplots(1, len(top_num), figsize=(4.6 * len(top_num), 3.8))
        axp = np.atleast_1d(axp)
        PartialDependenceDisplay.from_estimator(best_model, X_train, top_num, ax=axp, grid_resolution=20)
        fig_pdp.suptitle("Partial Dependence: pengaruh rata-rata fitur terhadap prediksi G3")
        fig_pdp.tight_layout(); close_fig(fig_pdp)

    coef = None
    final_est = best_model.named_steps["m"]
    if hasattr(final_est, "coef_"):
        names = best_model.named_steps["pre"].get_feature_names_out()
        coef = pd.Series(np.ravel(final_est.coef_), index=names).sort_values(key=np.abs, ascending=False).head(10)

    mask0 = (y_test.values == 0)
    mae_zero = float(np.abs(resid[mask0]).mean()) if mask0.sum() else None
    mae_other = float(np.abs(resid[~mask0]).mean()) if (~mask0).sum() else None

    schema = {
        "model_name": best_name, "target": TARGET, "use_prev_grades": bool(use_prev_grades),
        "feature_order": X.columns.tolist(),
        "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
                    for c in num_cols},
        "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "metrics_test": {k: float(v) for k, v in rep.loc["Test"].items()},
        "top_features": imp.head(10)["Fitur"].tolist(), "pass_threshold": PASS_THRESHOLD,
    }
    return dict(model=best_model, schema=schema, bench=bench, tune=tune_df, report=rep, importance=imp,
                risk=risk, coef=coef, fig_diag=fig_diag, fig_bench=fig_bench, fig_imp=fig_imp, fig_pdp=fig_pdp,
                n_train=len(X_train), n_test=len(X_test), n_features=X.shape[1], num_cols=num_cols,
                cat_cols=cat_cols, mae_zero=mae_zero, mae_other=mae_other, tunable=tunable,
                gap_r2=float(rep.loc["Train", "R2"] - rep.loc["Test", "R2"]))


# =============================================================================
#  BAGIAN D - PELATIHAN MODEL CLUSTERING (di-cache)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_clustering(_df, data_key, include_grades, min_k_final, mode):
    cfg = MODE_CFG[mode]
    grade_cols = ["G1", "G2", "G3"]
    feat_df = (_df if include_grades else _df.drop(columns=grade_cols)).copy()
    num_cols, cat_cols = split_columns(feat_df)
    feat_df[num_cols] = feat_df[num_cols].astype(float)

    preprocessor = make_preprocessor(num_cols, cat_cols, scale=True)
    transformer = Pipeline([("pre", preprocessor), ("pca", PCA(n_components=0.80, random_state=RANDOM_STATE))])

    # train-test split: cluster dipelajari dari train, kualitas diuji pada test
    df_train, df_test = train_test_split(feat_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Z_train = transformer.fit_transform(df_train)
    Z_test = transformer.transform(df_test)

    # pemilihan algoritma & k
    rows = []
    for k in range(2, 9):
        algos = {
            "KMeans": KMeans(n_clusters=k, n_init=cfg["km_init"], random_state=RANDOM_STATE),
            "GMM": GaussianMixture(n_components=k, covariance_type="diag", n_init=cfg["gmm_init"],
                                   random_state=RANDOM_STATE),
            "Agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        }
        for name, mdl in algos.items():
            lab = mdl.fit_predict(Z_train)
            m = eval_labels(Z_train, lab)
            m.update(Algoritma=name, k=k, inertia=mdl.inertia_ if name == "KMeans" else np.nan,
                     bic=mdl.bic(Z_train) if name == "GMM" else np.nan)
            rows.append(m)
    sel = pd.DataFrame(rows)
    sel["rank_score"] = (sel["silhouette"].rank(ascending=False) + sel["calinski"].rank(ascending=False)
                         + sel["davies"].rank(ascending=True))

    fig_sel, ax = plt.subplots(1, 4, figsize=(21, 4))
    for name, g in sel.groupby("Algoritma"):
        ax[0].plot(g["k"], g["silhouette"], "o-", label=name)
        ax[1].plot(g["k"], g["calinski"], "o-", label=name)
        ax[2].plot(g["k"], g["davies"], "o-", label=name)
    ax[0].set_title("Silhouette (makin besar makin baik)")
    ax[1].set_title("Calinski-Harabasz (makin besar makin baik)")
    ax[2].set_title("Davies-Bouldin (makin kecil makin baik)")
    for a in ax[:3]:
        a.set_xlabel("k"); a.legend()
    km = sel[sel.Algoritma == "KMeans"]
    ax[3].plot(km["k"], km["inertia"], "o-", color="#C44E52")
    ax[3].set_title("Elbow KMeans (inertia)"); ax[3].set_xlabel("k")
    fig_sel.tight_layout(); close_fig(fig_sel)

    # DBSCAN (pembanding)
    nn = NearestNeighbors(n_neighbors=10).fit(Z_train)
    kdist = np.sort(nn.kneighbors(Z_train)[0][:, -1])
    db_rows = []
    for eps in np.quantile(kdist, [.5, .6, .7, .8, .9]):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z_train)
        mask = lab != -1
        n_cl = len(set(lab[mask]))
        sil = silhouette_score(Z_train[mask], lab[mask]) if n_cl >= 2 and mask.sum() > n_cl else np.nan
        db_rows.append({"eps": round(float(eps), 3), "n_cluster": n_cl,
                        "noise (%)": round(100 * float((~mask).mean()), 1), "silhouette (non-noise)": sil})
    dbscan_df = pd.DataFrame(db_rows)

    # model final: hanya algoritma dengan .predict() yang layak untuk deployment
    deployable = sel[sel.Algoritma.isin(["KMeans", "GMM"]) & (sel.k >= min_k_final)]
    best = deployable.sort_values("rank_score").iloc[0]
    best_algo, best_k = best["Algoritma"], int(best["k"])

    def build_model(algo, k):
        if algo == "KMeans":
            return KMeans(n_clusters=k, n_init=cfg["final_init"], random_state=RANDOM_STATE)
        return GaussianMixture(n_components=k, covariance_type="diag", n_init=max(cfg["gmm_init"], 5),
                               random_state=RANDOM_STATE)

    final_model = build_model(best_algo, best_k).fit(Z_train)
    final_pipeline = Pipeline(transformer.steps + [("cluster", final_model)])

    # evaluasi train vs test + stabilitas bootstrap
    lab_train = final_pipeline.predict(df_train)
    lab_test = final_pipeline.predict(df_test)
    ev = pd.DataFrame({"Train": eval_labels(Z_train, lab_train), "Test": eval_labels(Z_test, lab_test)}).T
    rng = np.random.RandomState(RANDOM_STATE)
    ari = []
    for _ in range(cfg["boot"]):
        idx = rng.choice(len(Z_train), size=len(Z_train), replace=True)
        mdl_b = build_model(best_algo, best_k).fit(Z_train[idx])
        ari.append(adjusted_rand_score(lab_train, mdl_b.predict(Z_train)))
    ari = np.array(ari)
    rand_sil = float(np.mean([silhouette_score(Z_train, rng.randint(0, best_k, len(Z_train))) for _ in range(10)]))

    # interpretasi: label diurutkan menurut rata-rata G3
    df_all = _df.copy()
    Z_all = transformer.transform(feat_df)
    df_all["cluster"] = final_pipeline.predict(feat_df)
    order = df_all.groupby("cluster")["G3"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    df_all["cluster"] = df_all["cluster"].map(remap)
    size = df_all["cluster"].value_counts().sort_index()

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    P2 = pca2.fit_transform(preprocessor.transform(feat_df))
    sample_sil = silhouette_samples(Z_all, df_all["cluster"])
    pal = sns.color_palette("Set2")
    fig_pca, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    for c in range(best_k):
        m = df_all["cluster"].values == c
        axs[0].scatter(P2[m, 0], P2[m, 1], s=35, color=pal[c % 8], edgecolor="k", lw=.3, label=f"Cluster {c}")
    axs[0].legend(); axs[0].set_title(f"Sebaran cluster ({best_algo}, k={best_k}) pada PCA 2D")
    axs[0].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
    axs[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    y0 = 0
    for c in range(best_k):
        v = np.sort(sample_sil[df_all["cluster"].values == c])
        axs[1].fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=.8, color=pal[c % 8])
        axs[1].text(-0.08, y0 + len(v) / 2, f"C{c}")
        y0 += len(v) + 10
    axs[1].axvline(sample_sil.mean(), color="r", ls="--")
    axs[1].set_title("Silhouette per observasi"); axs[1].set_xlabel("Nilai silhouette"); axs[1].set_yticks([])
    fig_pca.tight_layout(); close_fig(fig_pca)

    prof_cols = _df.select_dtypes(include=np.number).columns.tolist()
    prof = df_all.groupby("cluster")[prof_cols].mean()
    z_prof = (prof - df_all[prof_cols].mean()) / df_all[prof_cols].std()
    fig_heat, axh = plt.subplots(figsize=(7, 8))
    sns.heatmap(z_prof.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=.5, ax=axh)
    axh.set_title("Deviasi rata-rata fitur tiap cluster (z-score)\nmerah = di atas rata-rata, biru = di bawah")
    fig_heat.tight_layout(); close_fig(fig_heat)

    # kategori menonjol per cluster
    cat_report = []
    for c in cat_cols:
        ct = pd.crosstab(df_all["cluster"], df_all[c], normalize="index")
        overall = df_all[c].value_counts(normalize=True)
        for cl in ct.index:
            for cat in ct.columns:
                cat_report.append({"cluster": cl, "fitur": c, "kategori": cat,
                                   "proporsi_cluster": ct.loc[cl, cat], "proporsi_populasi": overall[cat],
                                   "lift": ct.loc[cl, cat] / overall[cat]})
    cat_report = pd.DataFrame(cat_report)
    notable = cat_report[(cat_report["lift"] >= 1.3) & (cat_report["proporsi_cluster"] >= 0.20)]
    notable = notable.sort_values(["cluster", "lift"], ascending=[True, False]).groupby("cluster").head(5)

    H, p = kruskal(*[g["G3"].values for _, g in df_all.groupby("cluster")])

    def label_cluster(cl):
        z = z_prof.loc[cl]
        traits = []
        if z.get("G3", 0) > 0.5:
            traits.append("nilai tinggi")
        elif z.get("G3", 0) < -0.5:
            traits.append("nilai rendah")
        else:
            traits.append("nilai menengah")
        if z.get("failures", 0) > 0.5: traits.append("riwayat gagal banyak")
        if z.get("absences", 0) > 0.5: traits.append("absen tinggi")
        if z.get("studytime", 0) > 0.5: traits.append("belajar rajin")
        if z.get("Dalc", 0) + z.get("Walc", 0) > 1.0: traits.append("konsumsi alkohol tinggi")
        if z.get("goout", 0) > 0.5: traits.append("sering bersosialisasi")
        return ", ".join(traits)

    cluster_names = {int(cl): f"Cluster {cl}: {label_cluster(cl)}" for cl in prof.index}
    schema = {
        "algorithm": best_algo, "k": best_k, "include_grades": bool(include_grades),
        "feature_order": feat_df.columns.tolist(),
        "numeric": {c: {"min": float(feat_df[c].min()), "max": float(feat_df[c].max()),
                        "median": float(feat_df[c].median())} for c in num_cols},
        "categorical": {c: sorted(feat_df[c].dropna().unique().tolist()) for c in cat_cols},
        "label_map": {int(k): int(v) for k, v in remap.items()},
        "cluster_names": cluster_names, "cluster_sizes": {int(k): int(v) for k, v in size.items()},
        "metrics": {"silhouette_train": float(ev.loc["Train", "silhouette"]),
                    "silhouette_test": float(ev.loc["Test", "silhouette"]),
                    "davies_bouldin_test": float(ev.loc["Test", "davies"]),
                    "calinski_test": float(ev.loc["Test", "calinski"]),
                    "stability_ari": float(ari.mean())},
    }
    return dict(model=final_pipeline, schema=schema, sel=sel, dbscan=dbscan_df, ev=ev, ari=ari,
                rand_sil=rand_sil, df_all=df_all, prof=prof, size=size, notable=notable,
                kruskal_H=float(H), kruskal_p=float(p), fig_sel=fig_sel, fig_pca=fig_pca, fig_heat=fig_heat,
                n_train=len(df_train), n_test=len(df_test), n_pc=Z_train.shape[1],
                var_pc=float(transformer.named_steps["pca"].explained_variance_ratio_.sum()))


@st.cache_resource(show_spinner=False)
def make_eda_figure(_df, data_key):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    sns.histplot(_df[TARGET], bins=21, kde=True, ax=ax[0], color="#4C72B0"); ax[0].set_title(f"Distribusi {TARGET}")
    corr = _df.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET).sort_values()
    corr.plot(kind="barh", ax=ax[1], color=np.where(corr > 0, "#55A868", "#C44E52"))
    ax[1].set_title(f"Korelasi numerik dengan {TARGET}")
    sns.boxplot(data=_df, x="failures", y=TARGET, ax=ax[2], palette="viridis"); ax[2].set_title(f"{TARGET} vs failures")
    fig.tight_layout()
    return close_fig(fig)


# =============================================================================
#  BAGIAN E - FUNGSI BANTU APLIKASI
# =============================================================================
def is_whole(x) -> bool:
    return float(x).is_integer()


def merged_feature_specs(schemas):
    numeric, categorical, order = {}, {}, []
    for sch in schemas:
        for c in sch["feature_order"]:
            if c not in order:
                order.append(c)
        for c, v in sch["numeric"].items():
            numeric.setdefault(c, v)
        for c, opts in sch["categorical"].items():
            categorical.setdefault(c, opts)
    return order, numeric, categorical


def render_input_form(order, numeric, categorical) -> dict:
    inputs, placed = {}, set()

    def widget(col):
        label, help_txt = LABELS.get(col, col), HELP.get(col)
        if col in numeric:
            v = numeric[col]
            lo, hi, med = v["min"], v["max"], v["median"]
            if is_whole(lo) and is_whole(hi):
                lo, hi = int(lo), int(hi)
                if hi <= lo:
                    hi = lo + 1
                return st.slider(label, lo, hi, int(min(max(round(med), lo), hi)), key=f"in_{col}", help=help_txt)
            if hi <= lo:
                hi = lo + 1.0
            return st.slider(label, float(lo), float(hi), float(med), key=f"in_{col}", help=help_txt)
        fmt = VALUE_LABELS.get(col, {})
        return st.selectbox(label, categorical[col], key=f"in_{col}", help=help_txt,
                            format_func=lambda x, f=fmt: f.get(x, str(x)))

    for gname, cols in GROUPS.items():
        present = [c for c in cols if c in order]
        if not present:
            continue
        with st.expander(gname, expanded=(gname == "📚 Akademik")):
            for c in present:
                inputs[c] = widget(c)
                placed.add(c)
    rest = [c for c in order if c not in placed]
    if rest:
        with st.expander("Lainnya"):
            for c in rest:
                inputs[c] = widget(c)
    return inputs


def to_frame(values: dict, feature_order, numeric_cols) -> pd.DataFrame:
    row = pd.DataFrame([values])[feature_order].copy()
    for c in numeric_cols:
        row[c] = pd.to_numeric(row[c], errors="coerce").astype(float)
    return row


def clean_batch(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = df_in[schema["feature_order"]].copy()
    for c in schema["numeric"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def predict_g3(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0, 20)


def predict_cluster(model, schema: dict, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict(X)
    lm = schema["label_map"]
    return np.array([lm[int(r)] if int(r) in lm else lm[str(int(r))] for r in raw])


def model_bytes(model) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


# =============================================================================
#  BAGIAN F - SIDEBAR: PENGATURAN MODEL
# =============================================================================
st.title("🎓 Analisis Performa Siswa")
st.caption("Pelatihan model, prediksi nilai akhir (regresi), dan segmentasi profil siswa (clustering) "
           "dalam satu aplikasi. Dataset: student-mat.")

with st.sidebar:
    with st.expander("⚙️ Pengaturan model", expanded=False):
        with st.form("model_settings"):
            f_prev = st.checkbox("Regresi: pakai G1 & G2 sebagai fitur", value=True,
                                 help="Nonaktifkan untuk model peringatan dini (lebih sulit, akurasi lebih rendah).")
            f_inc = st.checkbox("Clustering: pakai G1-G3 sebagai fitur", value=True)
            f_mink = st.slider("Clustering: k minimal model final", 2, 5, 3,
                               help="k = 2 hanya membelah siswa menjadi 'baik' dan 'buruk'.")
            f_mode = st.radio("Mode pelatihan", ["Cepat", "Lengkap"], index=0,
                              help="Cepat: tuning ringan, cocok untuk server gratis. "
                                   "Lengkap: setara kode Colab, lebih lama.")
            st.form_submit_button("Terapkan & latih ulang")
    use_prev_grades, include_grades, min_k_final, train_mode = f_prev, f_inc, f_mink, f_mode


# =============================================================================
#  BAGIAN G - PELATIHAN (otomatis, hanya sekali per konfigurasi)
# =============================================================================
with st.spinner(f"Melatih model regresi (mode {train_mode}) ... hanya sekali per konfigurasi."):
    RES_R = train_regression(df_raw, DATA_KEY, use_prev_grades, train_mode)
with st.spinner(f"Melatih model clustering (mode {train_mode}) ..."):
    RES_C = train_clustering(df_raw, DATA_KEY, include_grades, min_k_final, train_mode)

reg_model, reg_schema = RES_R["model"], RES_R["schema"]
clu_model, clu_schema = RES_C["model"], RES_C["schema"]

with st.sidebar:
    st.header("Profil siswa")
    st.caption("Isi data siswa; hasilnya diperbarui otomatis pada tab di sebelah kanan.")
    order, numeric_specs, categorical_specs = merged_feature_specs([reg_schema, clu_schema])
    user_inputs = render_input_form(order, numeric_specs, categorical_specs)
    st.divider()
    st.caption("Model aktif")
    st.write(f"✅ Regresi: {reg_schema['model_name']}")
    st.write(f"✅ Clustering: {clu_schema['algorithm']}, k = {clu_schema['k']}")

tab_reg, tab_clu, tab_batch, tab_evr, tab_evc, tab_data = st.tabs([
    "📈 Prediksi Nilai", "🧩 Segmentasi Siswa", "📂 Prediksi Massal",
    "🔬 Evaluasi Regresi", "🔬 Evaluasi Clustering", "🗂️ Data & Unduhan"])


# =============================================================================
#  TAB 1 - PREDIKSI NILAI
# =============================================================================
with tab_reg:
    st.subheader("Prediksi nilai akhir (G3)")
    X_row = to_frame(user_inputs, reg_schema["feature_order"], reg_schema["numeric"].keys())
    pred = float(predict_g3(reg_model, X_row)[0])
    thr = reg_schema["pass_threshold"]
    rmse = reg_schema["metrics_test"]["RMSE"]
    lo_b, hi_b = max(0.0, pred - rmse), min(20.0, pred + rmse)
    pop_mean = float(df_raw[TARGET].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediksi nilai akhir", f"{pred:.1f} / 20", delta=f"{pred - pop_mean:+.1f} dari rata-rata siswa")
    c2.metric("Perkiraan rentang", f"{lo_b:.1f} - {hi_b:.1f}", help="Prediksi ± 1 RMSE pada data test.")
    c3.metric("Status", "Berisiko" if pred < thr else "Aman", help=f"Berisiko bila prediksi G3 < {thr}.")
    st.progress(min(max(pred / 20.0, 0.0), 1.0))

    if pred < thr:
        st.warning(f"Prediksi nilai akhir berada di bawah batas lulus ({thr}). "
                   "Siswa ini layak mendapat perhatian atau pendampingan lebih awal.")
    else:
        st.success(f"Prediksi nilai akhir berada di atas batas lulus ({thr}).")

    m = reg_schema["metrics_test"]
    st.caption(f"Model: {reg_schema['model_name']} | pada data test: MAE {m['MAE']:.2f}, RMSE {m['RMSE']:.2f}, "
               f"R² {m['R2']:.2f}. Prediksi adalah perkiraan statistik, bukan kepastian.")
    if reg_schema["use_prev_grades"]:
        st.info("Model ini memakai nilai G1 dan G2, yang sangat menentukan hasil prediksi. "
                "Untuk skenario peringatan dini, nonaktifkan opsi G1 & G2 di Pengaturan model.")

    st.markdown("#### Simulasi what-if")
    st.caption("Ubah satu fitur sambil menahan fitur lain sesuai isian sidebar, lalu lihat dampaknya pada prediksi.")
    num_feats = [c for c in reg_schema["feature_order"] if c in reg_schema["numeric"]]
    top_pref = [c for c in reg_schema["top_features"] if c in num_feats]
    options = top_pref + [c for c in num_feats if c not in top_pref]
    feat = st.selectbox("Fitur yang diubah", options, format_func=lambda c: LABELS.get(c, c))
    spec = reg_schema["numeric"][feat]
    if is_whole(spec["min"]) and is_whole(spec["max"]):
        grid = np.arange(int(spec["min"]), int(spec["max"]) + 1, dtype=float)
    else:
        grid = np.linspace(spec["min"], spec["max"], 30)
    X_var = X_row.loc[X_row.index.repeat(len(grid))].reset_index(drop=True)
    X_var[feat] = grid
    sim = pd.DataFrame({"Prediksi G3": predict_g3(reg_model, X_var)}, index=pd.Index(grid, name=LABELS.get(feat, feat)))
    st.line_chart(sim)
    best_val = grid[int(sim["Prediksi G3"].values.argmax())]
    st.caption(f"Dengan fitur lain tetap, prediksi tertinggi ({sim['Prediksi G3'].max():.1f}) dicapai saat "
               f"{LABELS.get(feat, feat)} = {best_val:g}. Ini pola asosiasi dalam model, bukan bukti sebab-akibat.")


# =============================================================================
#  TAB 2 - SEGMENTASI
# =============================================================================
with tab_clu:
    st.subheader("Segmentasi profil siswa")
    X_c = to_frame(user_inputs, clu_schema["feature_order"], clu_schema["numeric"].keys())
    cl = int(predict_cluster(clu_model, clu_schema, X_c)[0])
    names = clu_schema["cluster_names"]
    st.success(f"Siswa ini masuk **{names.get(cl, f'Cluster {cl}')}**")

    mt = clu_schema["metrics"]
    st.caption(f"Model: {clu_schema['algorithm']} (k = {clu_schema['k']}) | silhouette test {mt['silhouette_test']:.2f} | "
               f"stabilitas ARI {mt['stability_ari']:.2f}. Silhouette di bawah 0,25 berarti cluster saling "
               "tumpang-tindih, jadi baca hasil ini sebagai kecenderungan, bukan batas yang tegas.")

    if hasattr(clu_model, "predict_proba"):
        proba = clu_model.predict_proba(X_c)[0]
        lm = clu_schema["label_map"]
        pr = pd.Series({f"Cluster {lm[i]}": p for i, p in enumerate(proba)}).sort_index()
        st.markdown("**Probabilitas keanggotaan**")
        st.bar_chart(pr)

    prof_df, df_all = RES_C["prof"], RES_C["df_all"]
    st.markdown("#### Perbandingan dengan rata-rata cluster")
    key_cols = [c for c in ["G1", "G2", "G3", "absences", "failures", "studytime", "goout",
                            "Dalc", "Walc", "Medu", "Fedu", "age"] if c in prof_df.columns and c in user_inputs]
    cmp_df = pd.DataFrame({
        "Siswa ini": [user_inputs[c] for c in key_cols],
        f"Rata-rata Cluster {cl}": [prof_df.loc[cl, c] for c in key_cols],
        "Rata-rata semua siswa": [df_all[c].mean() for c in key_cols],
    }, index=[LABELS.get(c, c) for c in key_cols]).astype(float).round(2)
    st.dataframe(cmp_df)

    st.markdown("#### Profil seluruh cluster")
    prof_show = prof_df.copy()
    prof_show.index = [names.get(i, f"Cluster {i}") for i in prof_show.index]
    prof_show.columns = [LABELS.get(c, c) for c in prof_show.columns]
    st.dataframe(prof_show.T.round(2))

    sizes = RES_C["size"].copy()
    sizes.index = [f"Cluster {i}" for i in sizes.index]
    st.markdown("**Jumlah siswa per cluster**")
    st.bar_chart(sizes)


# =============================================================================
#  TAB 3 - PREDIKSI MASSAL
# =============================================================================
with tab_batch:
    st.subheader("Prediksi massal dari file")
    st.caption("Unggah CSV atau Excel berisi data banyak siswa dengan nama kolom seperti dataset asli.")
    needed = []
    for sch in (reg_schema, clu_schema):
        needed += [c for c in sch["feature_order"] if c not in needed]
    template = pd.DataFrame([user_inputs])[[c for c in needed if c in user_inputs]]
    st.download_button("Unduh template (1 baris contoh)", template.to_csv(index=False).encode("utf-8"),
                       file_name="template_siswa.csv", mime="text/csv")

    up = st.file_uploader("Unggah file data siswa", type=["csv", "xlsx"], key="upload_batch")
    if up is not None:
        try:
            df_up = (pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv")
                     else pd.read_excel(up))
        except Exception as e:  # noqa: BLE001
            st.error(f"File tidak dapat dibaca: {e}")
            df_up = None
        if df_up is not None:
            st.write(f"{len(df_up)} baris, {df_up.shape[1]} kolom terbaca.")
            result = df_up.copy()
            miss = [c for c in reg_schema["feature_order"] if c not in df_up.columns]
            if miss:
                st.warning(f"Prediksi G3 dilewati, kolom hilang: {', '.join(miss)}")
            else:
                result["Prediksi_G3"] = predict_g3(reg_model, clean_batch(df_up, reg_schema)).round(2)
                result["Status_risiko"] = np.where(result["Prediksi_G3"] < reg_schema["pass_threshold"],
                                                   "Berisiko", "Aman")
            miss = [c for c in clu_schema["feature_order"] if c not in df_up.columns]
            if miss:
                st.warning(f"Clustering dilewati, kolom hilang: {', '.join(miss)}")
            else:
                labs = predict_cluster(clu_model, clu_schema, clean_batch(df_up, clu_schema))
                result["Cluster"] = labs
                result["Nama_cluster"] = [clu_schema["cluster_names"].get(int(i), f"Cluster {i}") for i in labs]
            st.dataframe(result)
            if "Status_risiko" in result:
                n_risk = int((result["Status_risiko"] == "Berisiko").sum())
                st.info(f"{n_risk} dari {len(result)} siswa ({n_risk / len(result):.0%}) diprediksi berisiko.")
            st.download_button("Unduh hasil (CSV)", result.to_csv(index=False).encode("utf-8"),
                               file_name="hasil_prediksi_siswa.csv", mime="text/csv")


# =============================================================================
#  TAB 4 - EVALUASI REGRESI
# =============================================================================
with tab_evr:
    R = RES_R
    st.subheader("Evaluasi model regresi")
    st.write(f"Data dibagi **{R['n_train']} train / {R['n_test']} test** (stratified menurut kuantil G3). "
             f"Jumlah fitur: {R['n_features']}.")

    rep = R["report"]
    r2_t, mae_t, rmse_t = rep.loc["Test", "R2"], rep.loc["Test", "MAE"], rep.loc["Test", "RMSE"]
    top3 = R["importance"]["Fitur"].head(3).tolist()
    st.markdown("#### Ringkasan interpretasi")
    lines = [
        f"Model terbaik: **{R['schema']['model_name']}** ({'dengan' if use_prev_grades else 'tanpa'} fitur G1/G2).",
        f"Model menjelaskan sekitar **{r2_t:.1%}** variasi nilai akhir pada data test (R² = {r2_t:.3f}).",
        f"Rata-rata kesalahan {mae_t:.2f} poin (MAE), RMSE {rmse_t:.2f} pada skala 0-20; "
        f"{rep.loc['Test', 'Akurasi ±2 poin']:.0%} prediksi meleset paling banyak 2 poin.",
        f"Fitur paling berpengaruh: {', '.join(top3)}.",
    ]
    if use_prev_grades:
        lines.append("G1 dan G2 sangat dominan karena nilai akhir berkorelasi kuat dengan nilai periode sebelumnya. "
                     "Nonaktifkan opsi G1 & G2 untuk melihat faktor non-akademik.")
    if R["mae_zero"] is not None and R["mae_other"] is not None:
        lines.append(f"MAE siswa dengan G3 = 0 (drop out): {R['mae_zero']:.2f} poin, siswa lain: {R['mae_other']:.2f} poin.")
    lines.append("Selisih R² train-test = %.3f, %s" % (
        R["gap_r2"], "indikasi overfitting, pertimbangkan regularisasi lebih kuat." if R["gap_r2"] > 0.15
        else "generalisasi baik (tidak overfitting berat)."))
    st.markdown("\n".join(f"- {l}" for l in lines))

    st.markdown("#### Metrik pada data train, test, dan baseline")
    st.dataframe(rep.round(3))

    st.markdown("#### Pemilihan algoritma (cross-validation pada data train)")
    st.dataframe(R["bench"].round(3))
    show_fig(R["fig_bench"])

    st.markdown(f"#### Hasil tuning ({', '.join(R['tunable'])})")
    st.dataframe(R["tune"].round(3))

    st.markdown("#### Diagnostik prediksi")
    show_fig(R["fig_diag"])

    rk = R["risk"]
    st.markdown(f"#### Deteksi siswa berisiko (G3 < {PASS_THRESHOLD}) dari hasil regresi")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Terdeteksi benar", rk["tp"]); k2.metric("Terlewat", rk["fn"])
    k3.metric("Salah alarm", rk["fp"]);      k4.metric("Benar lulus", rk["tn"])
    st.caption(f"Recall siswa berisiko {rk['recall']:.0%}, precision {rk['precision']:.0%}.")

    st.markdown("#### Interpretasi fitur")
    cl_a, cl_b = st.columns(2)
    with cl_a:
        show_fig(R["fig_imp"])
    with cl_b:
        st.dataframe(R["importance"].head(10).round(4))
        if R["coef"] is not None:
            st.markdown("Koefisien terbesar (fitur terstandarisasi)")
            st.dataframe(R["coef"].round(3).rename("koefisien").to_frame())
    if R["fig_pdp"] is not None:
        show_fig(R["fig_pdp"])


# =============================================================================
#  TAB 5 - EVALUASI CLUSTERING
# =============================================================================
with tab_evc:
    C = RES_C
    st.subheader("Evaluasi model clustering")
    st.write(f"Data dibagi **{C['n_train']} train / {C['n_test']} test**. PCA mempertahankan {C['n_pc']} komponen "
             f"({C['var_pc']:.0%} varians).")

    ev = C["ev"]
    sil_t, ari_m = ev.loc["Test", "silhouette"], C["ari"].mean()
    sil_txt = "kuat" if sil_t > 0.5 else ("sedang" if sil_t >= 0.25 else "lemah (cluster tumpang-tindih)")
    stab_txt = "sangat stabil" if ari_m > 0.8 else ("cukup stabil" if ari_m >= 0.6 else "kurang stabil")
    st.markdown("#### Ringkasan interpretasi")
    lines = [
        f"Algoritma terpilih: **{clu_schema['algorithm']}**, k = {clu_schema['k']} "
        f"({'dengan' if include_grades else 'tanpa'} G1-G3 sebagai fitur).",
        f"Silhouette test = {sil_t:.3f} (label acak: {C['rand_sil']:.3f}); stabilitas ARI = {ari_m:.2f}.",
        f"Struktur cluster tergolong {sil_txt}; hasil pengelompokan {stab_txt} terhadap perubahan sampel.",
        "Acuan silhouette: > 0,5 kuat, 0,25-0,5 sedang, < 0,25 lemah. Pada data survei campuran, cluster sebaiknya "
        "dibaca sebagai segmentasi deskriptif (kecenderungan), bukan batas yang tegas.",
        f"Uji Kruskal-Wallis untuk G3 antar cluster: H = {C['kruskal_H']:.1f}, p = {C['kruskal_p']:.1e} "
        + ("(perbedaan signifikan)." if C["kruskal_p"] < 0.05 else "(tidak signifikan)."),
    ]
    for c_id in C["prof"].index:
        lines.append(f"{clu_schema['cluster_names'][int(c_id)]} | n = {int(C['size'][c_id])}, "
                     f"rata-rata G3 = {C['prof'].loc[c_id, 'G3']:.1f}, absen = {C['prof'].loc[c_id, 'absences']:.1f}, "
                     f"failures = {C['prof'].loc[c_id, 'failures']:.2f}")
    st.markdown("\n".join(f"- {l}" for l in lines))

    st.markdown("#### Metrik evaluasi (train vs test)")
    st.dataframe(ev.round(3))

    st.markdown("#### Pemilihan algoritma dan jumlah cluster (data train)")
    show_fig(C["fig_sel"])
    st.dataframe(C["sel"].sort_values("rank_score").head(10)[["Algoritma", "k", "silhouette", "calinski", "davies"]].round(3))
    with st.expander("DBSCAN (hanya pembanding, tidak bisa memprediksi data baru)"):
        st.dataframe(C["dbscan"].round(3))

    st.markdown("#### Visualisasi cluster")
    show_fig(C["fig_pca"])

    st.markdown("#### Profil cluster")
    show_fig(C["fig_heat"])
    if len(C["notable"]):
        st.markdown("**Kategori yang menonjol per cluster** (lift ≥ 1,3 dan proporsi ≥ 20%)")
        st.dataframe(C["notable"].round(2))


# =============================================================================
#  TAB 6 - DATA & UNDUHAN
# =============================================================================
with tab_data:
    st.subheader("Data")
    st.write(f"{df_raw.shape[0]} baris, {df_raw.shape[1]} kolom (sumber: {raw_name}). "
             f"Missing value: {int(df_raw.isna().sum().sum())}.")
    show_fig(make_eda_figure(df_raw, DATA_KEY))
    n_zero = int((df_raw[TARGET] == 0).sum())
    st.caption(f"{n_zero} siswa ({n_zero / len(df_raw):.1%}) bernilai G3 = 0 (kemungkinan putus ujian). "
               "Ini dipertahankan karena bukan kesalahan data.")
    st.dataframe(df_raw.head(50))

    st.subheader("Unduh model & hasil")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Model regresi (.joblib)", model_bytes(reg_model), file_name="model_regresi.joblib")
        st.download_button("Skema regresi (.json)", json.dumps(reg_schema, indent=2).encode("utf-8"),
                           file_name="schema_regresi.json", mime="application/json")
    with d2:
        st.download_button("Model clustering (.joblib)", model_bytes(clu_model), file_name="model_cluster.joblib")
        st.download_button("Skema clustering (.json)", json.dumps(clu_schema, indent=2).encode("utf-8"),
                           file_name="schema_cluster.json", mime="application/json")
    st.download_button("Data dengan label cluster (.csv)", RES_C["df_all"].to_csv(index=False).encode("utf-8"),
                       file_name="data_dengan_cluster.csv", mime="text/csv")
    st.caption("File .joblib hanya dapat dimuat dengan versi scikit-learn yang sama dengan server aplikasi ini.")
