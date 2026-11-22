import html
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Prediksi Nilai Akhir Matematika",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PASS_THRESHOLD = 10

COLORS = {
    "bg": "#0E1117",
    "surface": "#161B22",
    "border": "rgba(148, 163, 184, 0.16)",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "accent": "#38BDF8",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
}


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: #0E1117;
            color: #E6EDF3;
        }

        section[data-testid="stSidebar"] {
            background-color: #11161E;
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        .app-header {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.25rem;
        }
        .app-title {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #F0F6FC;
            margin: 0 0 0.3rem 0;
            line-height: 1.25;
        }
        .app-subtitle {
            font-size: 0.9rem;
            color: #8B949E;
            margin: 0 0 0.6rem 0;
            line-height: 1.45;
        }
        .app-meta {
            font-size: 0.8rem;
            color: #7DD3FC;
            margin: 0;
        }

        .kpi-card {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 0.95rem 1.1rem;
            margin-bottom: 0.75rem;
        }
        .kpi-accent-sky { border-top: 3px solid #38BDF8; }
        .kpi-accent-emerald { border-top: 3px solid #10B981; }
        .kpi-accent-amber { border-top: 3px solid #F59E0B; }
        .kpi-accent-rose { border-top: 3px solid #F43F5E; }

        .kpi-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8B949E;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.5rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: #F0F6FC;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: #8B949E;
            line-height: 1.4;
        }
        .delta-pos { color: #34D399; font-weight: 600; }
        .delta-neg { color: #FB7185; font-weight: 600; }
        .delta-neu { color: #7DD3FC; font-weight: 600; }

        .info-panel {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-left: 4px solid #38BDF8;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.75rem 0 1.1rem 0;
        }
        .info-panel-emerald { border-left-color: #10B981; }
        .info-panel-amber { border-left-color: #F59E0B; }
        .info-panel-rose { border-left-color: #F43F5E; }
        .info-title {
            font-size: 0.94rem;
            font-weight: 600;
            color: #F0F6FC;
            margin-bottom: 0.4rem;
        }
        .info-body {
            font-size: 0.86rem;
            color: #C9D1D9;
            line-height: 1.55;
        }
        .info-body ul {
            margin: 0.25rem 0 0 1.1rem;
            padding: 0;
        }
        .info-body li {
            margin-bottom: 0.3rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: #F0F6FC;
            margin: 1.15rem 0 0.45rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }

        .sb-card {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.9rem;
        }
        .sb-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #F0F6FC;
            margin: 0;
        }
        .sb-card-sub {
            font-size: 0.76rem;
            color: #8B949E;
            margin-top: 0.2rem;
        }
        .sb-status {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            font-size: 0.79rem;
            color: #C9D1D9;
            line-height: 1.6;
        }
        .sb-status strong { color: #F0F6FC; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background-color: #161B22;
            padding: 0.35rem;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 0.85rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.45rem 0.95rem;
            font-weight: 600;
            font-size: 0.86rem;
            color: #8B949E;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(56, 189, 248, 0.15) !important;
            color: #F0F6FC !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        div[data-testid="stExpander"] {
            background-color: #161B22;
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 10px;
            margin-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", accent: str = "sky") -> str:
    return (
        f'<div class="kpi-card kpi-accent-{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def info_box(title: str, body_html: str, tone: str = "sky") -> str:
    cls = f"info-panel info-panel-{tone}" if tone != "sky" else "info-panel"
    return (
        f'<div class="{cls}">'
        f'<div class="info-title">{html.escape(title)}</div>'
        f'<div class="info-body">{body_html}</div>'
        f"</div>"
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_plotly(fig: go.Figure, title: str = "", height: int = 360, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F0F6FC", family="Inter, sans-serif"), x=0.01, xanchor="left")
        if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161B22",
        font=dict(family="Inter, sans-serif", color="#C9D1D9", size=12),
        margin=dict(l=48, r=24, t=52 if title else 24, b=44),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(22, 27, 34, 0.85)",
            bordercolor="rgba(148, 163, 184, 0.18)",
            borderwidth=1,
            font=dict(size=11, color="#E6EDF3"),
        ),
        hoverlabel=dict(bgcolor="#0E1117", bordercolor="#38BDF8", font=dict(color="#F0F6FC", size=12)),
    )
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color="#8B949E"),
        title_font=dict(color="#C9D1D9"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color="#8B949E"),
        title_font=dict(color="#C9D1D9"),
    )
    return fig


inject_custom_css()

# -----------------------------------------------------------------------------
# METADATA FITUR SESUAI DOKUMENTASI UCI & BRIEF PROYEK
# -----------------------------------------------------------------------------
LABELS = {
    "school": "Sekolah",
    "sex": "Jenis kelamin",
    "age": "Usia (tahun)",
    "address": "Tempat tinggal",
    "famsize": "Ukuran keluarga",
    "Pstatus": "Status orang tua",
    "Medu": "Pendidikan ibu",
    "Fedu": "Pendidikan ayah",
    "Mjob": "Pekerjaan ibu",
    "Fjob": "Pekerjaan ayah",
    "reason": "Alasan memilih sekolah",
    "guardian": "Wali siswa",
    "traveltime": "Waktu tempuh ke sekolah",
    "studytime": "Waktu belajar mingguan",
    "failures": "Riwayat gagal kelas",
    "schoolsup": "Bimbingan tambahan sekolah",
    "famsup": "Dukungan belajar keluarga",
    "paid": "Les berbayar (Matematika)",
    "activities": "Kegiatan ekstrakurikuler",
    "nursery": "Pernah prasekolah (TK)",
    "higher": "Ingin melanjutkan kuliah",
    "internet": "Akses internet di rumah",
    "romantic": "Hubungan romantis",
    "famrel": "Kualitas hubungan keluarga",
    "freetime": "Waktu luang setelah sekolah",
    "goout": "Intensitas keluar dgn teman",
    "Dalc": "Konsumsi alkohol hari kerja",
    "Walc": "Konsumsi alkohol akhir pekan",
    "health": "Kondisi kesehatan",
    "absences": "Jumlah ketidakhadiran (absen)",
    "G1": "Nilai Periode 1 (G1)",
    "G2": "Nilai Periode 2 (G2)",
    "G3": "Nilai Akhir (G3)",
}

HELP = {
    "school": "GP = Gabriel Pereira, MS = Mousinho da Silveira (Portugal).",
    "Medu": "Kode ordinal 0-4: 0 tidak bersekolah s.d. 4 pendidikan tinggi.",
    "Fedu": "Kode ordinal 0-4: 0 tidak bersekolah s.d. 4 pendidikan tinggi.",
    "traveltime": "Kode kategori durasi perjalanan ke sekolah, bukan angka menit langsung.",
    "studytime": "Kode kategori durasi belajar per minggu (contoh: kode 2 berarti 2-5 jam/minggu).",
    "failures": "Jumlah kegagalan kelas sebelumnya pada data aktual CSV (0, 1, 2, 3).",
    "famrel": "Skala 1 (sangat buruk) sampai 5 (sangat baik).",
    "freetime": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "goout": "Skala 1 (sangat jarang) sampai 5 (sangat sering).",
    "Dalc": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "Walc": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "health": "Skala 1 (sangat buruk) sampai 5 (sangat baik).",
    "absences": "Total ketidakhadiran (0-93). Dikeluarkan dari regresi default karena cakupan waktunya tidak terinci.",
    "G1": "Nilai periode pertama Matematika (skala 0-20).",
    "G2": "Nilai periode kedua Matematika (skala 0-20).",
    "G3": "Nilai akhir Matematika (skala 0-20). Target prediksi regresi.",
}

ORDINAL_LABELS = {
    "Medu": {0: "0 - Tidak sekolah", 1: "1 - SD (4 thn)", 2: "2 - SMP (5-9)", 3: "3 - Menengah", 4: "4 - Pend. Tinggi"},
    "Fedu": {0: "0 - Tidak sekolah", 1: "1 - SD (4 thn)", 2: "2 - SMP (5-9)", 3: "3 - Menengah", 4: "4 - Pend. Tinggi"},
    "traveltime": {1: "1 - <15 mnt", 2: "2 - 15-30 mnt", 3: "3 - 30-60 mnt", 4: "4 - >60 mnt"},
    "studytime": {1: "1 - <2 jam/mgg", 2: "2 - 2-5 jam/mgg", 3: "3 - 5-10 jam/mgg", 4: "4 - >10 jam/mgg"},
    "failures": {0: "0 - Tidak pernah", 1: "1 - 1 kali", 2: "2 - 2 kali", 3: "3 - 3 kali"},
    "famrel": {1: "1 - Sangat buruk", 2: "2 - Buruk", 3: "3 - Cukup", 4: "4 - Baik", 5: "5 - Sangat baik"},
    "health": {1: "1 - Sangat buruk", 2: "2 - Buruk", 3: "3 - Cukup", 4: "4 - Baik", 5: "5 - Sangat baik"},
    "freetime": {1: "1 - Sangat rendah", 2: "2 - Rendah", 3: "3 - Sedang", 4: "4 - Tinggi", 5: "5 - Sangat tinggi"},
    "goout": {1: "1 - Sangat jarang", 2: "2 - Jarang", 3: "3 - Sedang", 4: "4 - Sering", 5: "5 - Sangat sering"},
    "Dalc": {1: "1 - Sangat rendah", 2: "2 - Rendah", 3: "3 - Sedang", 4: "4 - Tinggi", 5: "5 - Sangat tinggi"},
    "Walc": {1: "1 - Sangat rendah", 2: "2 - Rendah", 3: "3 - Sedang", 4: "4 - Tinggi", 5: "5 - Sangat tinggi"},
}

VALUE_LABELS = {
    "school": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"},
    "sex": {"F": "Perempuan", "M": "Laki-laki"},
    "address": {"U": "Perkotaan", "R": "Pedesaan"},
    "famsize": {"LE3": "<= 3 anggota", "GT3": "> 3 anggota"},
    "Pstatus": {"T": "Tinggal bersama", "A": "Terpisah"},
    "guardian": {"mother": "Ibu", "father": "Ayah", "other": "Lainnya"},
    "reason": {"home": "Dekat rumah", "reputation": "Reputasi sekolah", "course": "Pilihan program", "other": "Lainnya"},
    "Mjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Layanan publik", "at_home": "Di rumah", "other": "Lainnya"},
    "Fjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Layanan publik", "at_home": "Di rumah", "other": "Lainnya"},
}
BINARY_YES_NO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
for _c in BINARY_YES_NO:
    VALUE_LABELS[_c] = {"yes": "Ya", "no": "Tidak"}

PRIMARY_FEATURES = ["G1", "G2", "failures", "studytime", "higher", "Medu", "goout"]

SIDEBAR_GROUPS = {
    "Akademik & Dukungan": ["reason", "traveltime", "schoolsup", "paid", "nursery", "absences"],
    "Latar Belakang Keluarga": ["Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"],
    "Demografi Siswa": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "Gaya Hidup & Kesehatan": ["activities", "internet", "romantic", "freetime", "Dalc", "Walc", "health"],
}

STUDENT_PRESETS = {
    "berprestasi": {
        "G1": 16, "G2": 17, "failures": 0, "studytime": 3, "higher": "yes", "absences": 2,
        "Medu": 4, "Fedu": 4, "goout": 2, "Dalc": 1, "Walc": 1, "schoolsup": "no",
        "famsup": "yes", "internet": "yes", "famrel": 5, "age": 16, "address": "U",
    },
    "berisiko": {
        "G1": 7, "G2": 7, "failures": 2, "studytime": 1, "higher": "no", "absences": 14,
        "Medu": 1, "Fedu": 1, "goout": 4, "Dalc": 3, "Walc": 4, "schoolsup": "yes",
        "famsup": "no", "internet": "no", "famrel": 2, "age": 18, "address": "R",
    },
}


# -----------------------------------------------------------------------------
# PEMUATAN DATA
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_data(content: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    else:
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name="student-mat")
        except ValueError:
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
    return df.drop_duplicates().reset_index(drop=True)


def get_raw_data():
    for folder in (BASE, Path.cwd()):
        for fname in ("student-mat.xlsx", "student-mat.csv"):
            p = folder / fname
            if p.exists():
                return p.read_bytes(), fname
    st.warning("File student-mat.xlsx tidak ditemukan di folder aplikasi. Silakan unggah dataset.")
    up = st.file_uploader("Unggah student-mat.xlsx atau student-mat.csv", type=["xlsx", "csv"], key="upload_dataset")
    if up is None:
        st.stop()
    return up.getvalue(), up.name


# -----------------------------------------------------------------------------
# FUNGSI PEMODELAN REGRESI & CLUSTERING (RINGKAS & CEPAT)
# -----------------------------------------------------------------------------
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
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
            ]),
            cat_cols,
        ),
    ])


def regression_metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Akurasi ±2 poin": float(np.mean(np.abs(err) <= 2)),
    }


@st.cache_resource(show_spinner=False)
def train_regression(_df: pd.DataFrame, data_key: str, use_prev_grades: bool, exclude_absences: bool) -> dict:
    drop_cols = [TARGET]
    if not use_prev_grades:
        drop_cols += ["G1", "G2"]
    if exclude_absences and "absences" in _df.columns:
        drop_cols.append("absences")

    X = _df.drop(columns=drop_cols).copy()
    y = _df[TARGET].astype(float)
    num_cols, cat_cols = split_columns(X)
    X[num_cols] = X[num_cols].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    def pre(scale: bool):
        return make_preprocessor(num_cols, cat_cols, scale=scale)

    candidates = {
        "Ridge (alpha=10)": Pipeline([("pre", pre(True)), ("m", Ridge(alpha=10.0))]),
        "Ridge (alpha=50)": Pipeline([("pre", pre(True)), ("m", Ridge(alpha=50.0))]),
        "Lasso (alpha=0.05)": Pipeline([("pre", pre(True)), ("m", Lasso(alpha=0.05))]),
        "Random Forest": Pipeline([
            ("pre", pre(False)),
            ("m", RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", pre(False)),
            ("m", GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("pre", pre(False)),
            ("m", HistGradientBoostingRegressor(
                learning_rate=0.05, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE
            )),
        ]),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_rmse, best_pipe = None, float("inf"), None
    for name, pipe in candidates.items():
        res = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        rmse_cv = float(-res["test_score"].mean())
        if rmse_cv < best_rmse:
            best_rmse = rmse_cv
            best_name = name
            best_pipe = pipe

    best_pipe.fit(X_train, y_train)
    pred_train = np.clip(best_pipe.predict(X_train), 0, 20)
    pred_test = np.clip(best_pipe.predict(X_test), 0, 20)

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pred_base = baseline.predict(X_test)

    metrics_train = regression_metrics(y_train, pred_train)
    metrics_test = regression_metrics(y_test, pred_test)
    metrics_base = regression_metrics(y_test, pred_base)

    schema = {
        "model_name": best_name,
        "target": TARGET,
        "use_prev_grades": bool(use_prev_grades),
        "exclude_absences": bool(exclude_absences),
        "feature_order": X.columns.tolist(),
        "numeric": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
            for c in num_cols
        },
        "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "metrics_baseline": metrics_base,
        "pass_threshold": PASS_THRESHOLD,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return {"model": best_pipe, "schema": schema}


@st.cache_resource(show_spinner=False)
def train_clustering(_df: pd.DataFrame, data_key: str, include_grades: bool = True) -> dict:
    grade_cols = ["G1", "G2", "G3"]
    feat_df = (_df if include_grades else _df.drop(columns=grade_cols)).copy()
    num_cols, cat_cols = split_columns(feat_df)
    feat_df[num_cols] = feat_df[num_cols].astype(float)

    preprocessor = make_preprocessor(num_cols, cat_cols, scale=True)
    transformer = Pipeline([
        ("pre", preprocessor),
        ("pca", PCA(n_components=0.80, random_state=RANDOM_STATE)),
    ])

    df_train, _ = train_test_split(feat_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Z_train = transformer.fit_transform(df_train)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE).fit(Z_train)
    final_pipeline = Pipeline(transformer.steps + [("cluster", kmeans)])

    df_all = _df.copy()
    df_all["cluster"] = final_pipeline.predict(feat_df)
    order = df_all.groupby("cluster")["G3"].mean().sort_values().index.tolist()
    remap = {int(old): int(new) for new, old in enumerate(order)}
    df_all["cluster"] = df_all["cluster"].map(remap)

    size = df_all["cluster"].value_counts().sort_index()
    pass_rates = df_all.groupby("cluster")["G3"].apply(lambda s: float((s >= PASS_THRESHOLD).mean()))
    num_prof_cols = df_all.select_dtypes(include=np.number).columns.drop("cluster", errors="ignore").tolist()
    prof = df_all.groupby("cluster")[num_prof_cols].mean()
    pop_means = df_all[num_prof_cols].mean()
    pop_stds = df_all[num_prof_cols].std().replace(0, 1.0)

    tier_titles = {0: "Performa Rendah / Perlu Perhatian", 1: "Performa Menengah", 2: "Performa Tinggi"}
    cluster_names = {}
    cluster_bullets = {}
    pop_higher_rate = float((df_all["higher"] == "yes").mean() * 100) if "higher" in df_all.columns else 0.0

    for cl in sorted(prof.index):
        cl_int = int(cl)
        cluster_names[cl_int] = f"Cluster {cl_int} - {tier_titles.get(cl_int, f'Kelompok {cl_int}')}"
        row = prof.loc[cl]
        z = (row - pop_means) / pop_stds

        bullets = [
            f"Rata-rata nilai periode: G1 = {row['G1']:.1f}, G2 = {row['G2']:.1f}, dan G3 = {row['G3']:.1f} (rata-rata kelas: {pop_means['G3']:.1f}).",
        ]
        if row["failures"] > pop_means["failures"] + 0.15:
            bullets.append(f"Riwayat gagal kelas lebih tinggi dari rata-rata ({row['failures']:.2f} vs {pop_means['failures']:.2f} kali).")
        elif row["failures"] < pop_means["failures"] - 0.10:
            bullets.append(f"Riwayat gagal kelas sangat rendah ({row['failures']:.2f} kali vs rata-rata {pop_means['failures']:.2f}).")

        if z.get("studytime", 0) > 0.25:
            bullets.append(f"Skor waktu belajar lebih tinggi dari rata-rata populasi ({row['studytime']:.2f} vs {pop_means['studytime']:.2f}).")
        elif z.get("studytime", 0) < -0.25:
            bullets.append(f"Skor waktu belajar cenderung lebih rendah ({row['studytime']:.2f} vs rata-rata {pop_means['studytime']:.2f}).")

        if z.get("goout", 0) > 0.25 or z.get("Dalc", 0) + z.get("Walc", 0) > 0.5:
            bullets.append(f"Aktivitas sosial/keluar bersama teman relatif tinggi (goout = {row['goout']:.2f} vs {pop_means['goout']:.2f}).")
        elif z.get("Medu", 0) > 0.25:
            bullets.append(f"Latar pendidikan ibu rata-rata lebih tinggi (skor {row['Medu']:.2f} vs {pop_means['Medu']:.2f}).")

        if "higher" in df_all.columns:
            cl_higher = float((df_all.loc[df_all["cluster"] == cl, "higher"] == "yes").mean() * 100)
            bullets.append(f"Minat melanjutkan kuliah: {cl_higher:.0f}% siswa di cluster ini (populasi: {pop_higher_rate:.0f}%).")

        cluster_bullets[cl_int] = bullets[:4]

    schema = {
        "algorithm": "KMeans",
        "k": 3,
        "include_grades": bool(include_grades),
        "feature_order": feat_df.columns.tolist(),
        "numeric": {
            c: {"min": float(feat_df[c].min()), "max": float(feat_df[c].max()), "median": float(feat_df[c].median())}
            for c in num_cols
        },
        "categorical": {c: sorted(feat_df[c].dropna().unique().tolist()) for c in cat_cols},
        "label_map": remap,
        "cluster_names": cluster_names,
        "cluster_sizes": {int(k): int(v) for k, v in size.items()},
        "cluster_bullets": cluster_bullets,
    }
    return {
        "model": final_pipeline,
        "schema": schema,
        "df_all": df_all,
        "prof": prof,
        "pass_rates": pass_rates,
        "size": size,
    }


# -----------------------------------------------------------------------------
# FUNGSI BANTU INPUT, PREDIKSI & INTERPRETASI
# -----------------------------------------------------------------------------
def is_whole(x) -> bool:
    return float(x).is_integer()


def merged_feature_specs(schemas):
    numeric, categorical, order = {}, {}, []
    for sch in schemas:
        for c in sch["feature_order"]:
            if c == TARGET:
                continue
            if c not in order:
                order.append(c)
        for c, v in sch["numeric"].items():
            if c != TARGET:
                numeric.setdefault(c, v)
        for c, opts in sch["categorical"].items():
            if c != TARGET:
                categorical.setdefault(c, opts)
    return order, numeric, categorical


def compute_default_inputs(df: pd.DataFrame, order, numeric, categorical) -> dict:
    defaults = {}
    for col in order:
        if col in numeric:
            v = numeric[col]
            lo, hi, med = v["min"], v["max"], v["median"]
            if is_whole(lo) and is_whole(hi):
                defaults[col] = int(min(max(round(med), int(lo)), int(hi)))
            else:
                defaults[col] = float(med)
        elif col in categorical:
            opts = categorical[col]
            mode_val = df[col].mode().iloc[0] if col in df.columns and not df[col].mode().empty else opts[0]
            defaults[col] = mode_val if mode_val in opts else opts[0]
    return defaults


def apply_preset_callback(preset_values: dict, defaults: dict, numeric: dict, categorical: dict):
    merged = dict(defaults)
    merged.update(preset_values)
    for col, val in merged.items():
        key = f"in_{col}"
        if col in numeric:
            v = numeric[col]
            lo, hi = v["min"], v["max"]
            if is_whole(lo) and is_whole(hi):
                st.session_state[key] = int(min(max(round(float(val)), int(lo)), int(hi)))
            else:
                st.session_state[key] = float(min(max(float(val), lo), hi))
        elif col in categorical and val in categorical[col]:
            st.session_state[key] = val


def ensure_session_inputs(defaults: dict, numeric: dict, categorical: dict):
    for col, def_val in defaults.items():
        key = f"in_{col}"
        if key not in st.session_state:
            st.session_state[key] = def_val
        else:
            cur = st.session_state[key]
            if col in numeric:
                v = numeric[col]
                lo, hi = v["min"], v["max"]
                if is_whole(lo) and is_whole(hi):
                    st.session_state[key] = int(min(max(round(float(cur)), int(lo)), int(hi)))
                else:
                    st.session_state[key] = float(min(max(float(cur), lo), hi))
            elif col in categorical and cur not in categorical[col]:
                st.session_state[key] = def_val


def format_input_value(col: str, val) -> str:
    if col in ORDINAL_LABELS:
        return ORDINAL_LABELS[col].get(int(round(float(val))), str(val))
    if col in VALUE_LABELS:
        return VALUE_LABELS[col].get(str(val), str(val))
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def render_input_form(order, numeric, categorical, exclude_absences: bool) -> dict:
    inputs, placed = {}, set()

    def widget(col):
        label, help_txt = LABELS.get(col, col), HELP.get(col)
        key = f"in_{col}"
        if col in numeric:
            v = numeric[col]
            lo, hi = v["min"], v["max"]
            if col in ORDINAL_LABELS and is_whole(lo) and is_whole(hi):
                opts = list(range(int(lo), int(hi) + 1))
                ord_map = ORDINAL_LABELS[col]
                return st.select_slider(
                    label, options=opts, key=key, help=help_txt,
                    format_func=lambda x, m=ord_map: m.get(int(x), str(x)),
                )
            if is_whole(lo) and is_whole(hi):
                lo_i, hi_i = int(lo), max(int(hi), int(lo) + 1)
                return st.slider(label, lo_i, hi_i, key=key, help=help_txt)
            hi_f = float(hi) if hi > lo else float(lo) + 1.0
            return st.slider(label, float(lo), hi_f, key=key, help=help_txt)

        opts = categorical[col]
        fmt = VALUE_LABELS.get(col, {})
        if col in BINARY_YES_NO and len(opts) == 2:
            return st.radio(
                label, opts, key=key, help=help_txt, horizontal=True,
                format_func=lambda x, f=fmt: f.get(x, str(x)),
            )
        return st.selectbox(
            label, opts, key=key, help=help_txt,
            format_func=lambda x, f=fmt: f.get(x, str(x)),
        )

    primary_cols = list(PRIMARY_FEATURES)
    if not exclude_absences and "absences" in order:
        primary_cols.append("absences")

    present_primary = [c for c in primary_cols if c in order]
    if present_primary:
        with st.expander("Fitur Utama", expanded=True):
            for c in present_primary:
                inputs[c] = widget(c)
                placed.add(c)

    for gname, cols in SIDEBAR_GROUPS.items():
        present = [c for c in cols if c in order and c not in placed]
        if not present:
            continue
        with st.expander(gname, expanded=False):
            for c in present:
                inputs[c] = widget(c)
                placed.add(c)

    rest = [c for c in order if c not in placed]
    if rest:
        with st.expander("Fitur Lainnya", expanded=False):
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


def generate_student_insights(inputs: dict, pred: float, thr: float, use_prev: bool) -> tuple[str, str, list[str]]:
    notes = []
    if pred < thr:
        tone = "rose"
        title = "Perlu Perhatian - Prediksi di Bawah Batas Lulus (< 10)"
    elif pred < thr + 2.0:
        tone = "amber"
        title = "Zona Waspada - Prediksi Mendekati Batas Lulus (10 - 12)"
    else:
        tone = "emerald"
        title = "Performa Aman - Prediksi di Atas Batas Lulus (>= 12)"

    if use_prev and "G1" in inputs and "G2" in inputs:
        g1, g2 = float(inputs["G1"]), float(inputs["G2"])
        if g2 == 0:
            notes.append(
                "<strong>Nilai G2 = 0</strong>: Pada data historis, seluruh siswa dengan G2 = 0 berakhir dengan G3 = 0 (indikasi tidak mengikuti ujian akhir atau putus studi)."
            )
        elif g2 < g1 - 1.5:
            notes.append(f"<strong>Tren nilai menurun</strong> dari G1 ({g1:g}) ke G2 ({g2:g}). Perlu evaluasi materi pada periode kedua.")
        elif g2 > g1 + 1.5:
            notes.append(f"<strong>Tren nilai meningkat</strong> dari G1 ({g1:g}) ke G2 ({g2:g}), mendukung proyeksi nilai akhir yang lebih baik.")

    if inputs.get("failures", 0) >= 1:
        notes.append(f"<strong>Riwayat gagal kelas ({int(inputs['failures'])} kali)</strong> berasosiasi negatif cukup kuat terhadap nilai akhir pada data historis.")
    if inputs.get("studytime", 2) <= 1:
        notes.append("<strong>Waktu belajar mingguan rendah (&lt; 2 jam/minggu)</strong>; menambah jam belajar terstruktur berpotensi membantu pemahaman materi.")
    if inputs.get("higher") == "no":
        notes.append("<strong>Tidak berencana melanjutkan kuliah</strong>: Pada data historis, kelompok ini memiliki rata-rata G3 lebih rendah dibandingkan yang berencana kuliah.")
    if inputs.get("goout", 3) >= 4:
        notes.append("<strong>Intensitas keluar bersama teman tinggi</strong>; perlu keseimbangan waktu antara aktivitas sosial dan belajar.")

    if not notes:
        notes.append("Profil belajar dan riwayat akademik berada pada kondisi yang mendukung capaian nilai akhir yang stabil.")
    notes.append("<em>Catatan metodologis: Interpretasi ini mencerminkan pola asosiasi statistik dari data historis Portugal (2005-2006), bukan hubungan sebab-akibat langsung.</em>")
    return tone, title, notes


# -----------------------------------------------------------------------------
# EKSEKUSI UTAMA: MUAT DATA, LATIH MODEL & RENDER SIDEBAR
# -----------------------------------------------------------------------------
raw_bytes, raw_name = get_raw_data()
df_raw = read_data(raw_bytes, raw_name)
missing_cols = [c for c in LABELS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Dataset tidak valid, kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
    st.stop()

DATA_KEY = f"{raw_name}-{df_raw.shape}-{int(pd.util.hash_pandas_object(df_raw, index=True).sum() % 10**12)}"

with st.sidebar:
    st.markdown(
        """
        <div class="sb-card">
            <div class="sb-card-title">Prediksi Nilai Matematika</div>
            <div class="sb-card-sub">Dataset UCI student-mat (395 siswa)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Pengaturan Model", expanded=False):
        with st.form("model_settings"):
            f_prev = st.toggle(
                "Gunakan nilai G1 dan G2",
                value=True,
                help="Nonaktifkan untuk simulasi prediksi dini sebelum nilai periode 1 dan 2 tersedia.",
            )
            f_no_abs = st.toggle(
                "Kecualikan fitur absences",
                value=True,
                help="Sesuai catatan cleaning: waktu pencatatan absences tidak terinci dan seluruh G3=0 tercatat absences=0.",
            )
            st.form_submit_button("Terapkan Pengaturan", width="stretch")

    use_prev_grades, exclude_absences = f_prev, f_no_abs

with st.spinner("Menyiapkan model prediksi dan segmentasi..."):
    RES_R = train_regression(df_raw, DATA_KEY, use_prev_grades, exclude_absences)
    RES_C = train_clustering(df_raw, DATA_KEY, include_grades=True)

reg_model, reg_schema = RES_R["model"], RES_R["schema"]
clu_model, clu_schema = RES_C["model"], RES_C["schema"]

order, numeric_specs, categorical_specs = merged_feature_specs([reg_schema, clu_schema])
default_inputs = compute_default_inputs(df_raw, order, numeric_specs, categorical_specs)
ensure_session_inputs(default_inputs, numeric_specs, categorical_specs)

with st.sidebar:
    st.markdown("**Profil Input Siswa**")
    st.caption("Sesuaikan profil siswa di bawah ini atau pilih contoh skenario.")

    p1, p2 = st.columns(2, gap="small")
    p1.button(
        "Berprestasi",
        width="stretch",
        on_click=apply_preset_callback,
        args=(STUDENT_PRESETS["berprestasi"], default_inputs, numeric_specs, categorical_specs),
    )
    p2.button(
        "Berisiko",
        width="stretch",
        on_click=apply_preset_callback,
        args=(STUDENT_PRESETS["berisiko"], default_inputs, numeric_specs, categorical_specs),
    )
    st.button(
        "Reset ke Median",
        width="stretch",
        on_click=apply_preset_callback,
        args=({}, default_inputs, numeric_specs, categorical_specs),
    )

    user_inputs = render_input_form(order, numeric_specs, categorical_specs, exclude_absences)

    st.divider()
    rm = reg_schema["metrics_test"]
    st.markdown(
        f"""
        <div class="sb-status">
            <strong>Model Regresi Aktif</strong><br/>
            Algoritma: <strong>{html.escape(reg_schema['model_name'])}</strong><br/>
            R² Uji: <strong>{rm['R2']:.3f}</strong> | RMSE: <strong>{rm['RMSE']:.2f}</strong> | MAE: <strong>{rm['MAE']:.2f}</strong><br/>
            Fitur aktif: {len(reg_schema['feature_order'])} kolom
        </div>
        """,
        unsafe_allow_html=True,
    )

X_reg_row = to_frame(user_inputs, reg_schema["feature_order"], reg_schema["numeric"].keys())
pred_g3_val = float(predict_g3(reg_model, X_reg_row)[0])

# -----------------------------------------------------------------------------
# HEADER UTAMA & NAVIGASI 4 TAB
# -----------------------------------------------------------------------------
mode_text = "Menggunakan G1 & G2" if use_prev_grades else "Mode Peringatan Dini (Tanpa G1 & G2)"
abs_text = "Absences dikecualikan" if exclude_absences else "Absences disertakan"

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Prediksi Nilai Akhir Matematika Siswa (G3)</div>
        <div class="app-subtitle">
            Aplikasi estimasi nilai akhir (skala 0-20), segmentasi profil siswa, ringkasan eksplorasi data, dan prediksi massal berbasis UCI Student Performance.
        </div>
        <div class="app-meta">
            Mode: {mode_text} &nbsp;|&nbsp; Preprocessing: {abs_text} &nbsp;|&nbsp; Model: {html.escape(reg_schema['model_name'])} (RMSE {rm['RMSE']:.2f})
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_seg, tab_eda, tab_batch = st.tabs([
    "Prediksi Nilai",
    "Segmentasi Siswa",
    "Data & EDA",
    "Prediksi Massal",
])


# =============================================================================
# TAB 1 - PREDIKSI NILAI
# =============================================================================
with tab_pred:
    thr = reg_schema["pass_threshold"]
    rmse = rm["RMSE"]
    mae = rm["MAE"]
    lo_b, hi_b = max(0.0, pred_g3_val - rmse), min(20.0, pred_g3_val + rmse)
    pop_mean = float(df_raw[TARGET].mean())
    delta_mean = pred_g3_val - pop_mean

    if pred_g3_val < thr:
        status_label, status_accent, bar_color = "Berisiko", "rose", COLORS["rose"]
        status_sub = f'<span class="delta-neg">{pred_g3_val - thr:+.1f} poin</span> dari batas lulus ({thr})'
    elif pred_g3_val < thr + 2.0:
        status_label, status_accent, bar_color = "Waspada", "amber", COLORS["amber"]
        status_sub = f'<span class="delta-neu">{pred_g3_val - thr:+.1f} poin</span> di atas batas lulus ({thr})'
    else:
        status_label, status_accent, bar_color = "Aman (Lulus)", "emerald", COLORS["emerald"]
        status_sub = f'<span class="delta-pos">{pred_g3_val - thr:+.1f} poin</span> di atas batas lulus ({thr})'

    k1, k2, k3 = st.columns(3, gap="medium")
    delta_cls = "delta-pos" if delta_mean >= 0 else "delta-neg"
    k1.markdown(
        kpi_card(
            "Prediksi Nilai Akhir (G3)",
            f"{pred_g3_val:.1f} / 20",
            f'<span class="{delta_cls}">{delta_mean:+.1f} poin</span> vs rata-rata kelas ({pop_mean:.1f})',
            accent="sky",
        ),
        unsafe_allow_html=True,
    )
    k2.markdown(
        kpi_card("Status Kelulusan", status_label, status_sub, accent=status_accent),
        unsafe_allow_html=True,
    )
    k3.markdown(
        kpi_card(
            "Rentang Perkiraan (+/- 1 RMSE)",
            f"{lo_b:.1f} - {hi_b:.1f}",
            f"RMSE uji: {rmse:.2f} | MAE uji: {mae:.2f} | R²: {rm['R2']:.2f}",
            accent="sky",
        ),
        unsafe_allow_html=True,
    )

    # Indikator horizontal skala 0-20 (bersih tanpa teks menumpuk)
    fig_scale = go.Figure()
    fig_scale.add_vrect(x0=0, x1=10, fillcolor="rgba(244, 63, 94, 0.12)", line_width=0)
    fig_scale.add_vrect(x0=10, x1=12, fillcolor="rgba(245, 158, 11, 0.12)", line_width=0)
    fig_scale.add_vrect(x0=12, x1=20, fillcolor="rgba(16, 185, 129, 0.12)", line_width=0)
    fig_scale.add_trace(go.Bar(
        x=[pred_g3_val],
        y=["Prediksi G3"],
        orientation="h",
        width=0.42,
        marker=dict(color=bar_color, line=dict(color="#F0F6FC", width=1)),
        text=[f" {pred_g3_val:.1f} / 20"],
        textposition="outside" if pred_g3_val < 17.5 else "inside",
        textfont=dict(color="#F0F6FC", size=13),
        hovertemplate="Prediksi G3: %{x:.2f} / 20<extra></extra>",
        showlegend=False,
    ))
    fig_scale.add_vline(
        x=thr,
        line_dash="dash",
        line_color="#E6EDF3",
        line_width=2,
        annotation_text=f"Batas Lulus ({thr})",
        annotation_position="top right",
        annotation_font=dict(color="#E6EDF3", size=11),
    )
    style_plotly(fig_scale, "Posisi Nilai Prediksi pada Skala 0-20", height=175, show_legend=False)
    fig_scale.update_layout(margin=dict(l=95, r=32, t=60, b=36))
    fig_scale.update_xaxes(range=[0, 20.5], dtick=2, title="Skala Nilai (0 - 20)")
    fig_scale.update_yaxes(showgrid=False, title="")
    st.plotly_chart(fig_scale, width="stretch", theme=None)

    # Interpretasi Faktor Utama
    tone, insight_title, insight_items = generate_student_insights(user_inputs, pred_g3_val, thr, use_prev_grades)
    ul_html = "<ul>" + "".join(f"<li>{item}</li>" for item in insight_items) + "</ul>"
    st.markdown(info_box(insight_title, ul_html, tone=tone), unsafe_allow_html=True)

    # Ringkasan Parameter Input (seperti pola aplikasi referensi dosen)
    section_title("Ringkasan Parameter Input Siswa")
    key_input_cols = [c for c in ["G1", "G2", "failures", "studytime", "higher", "Medu", "goout", "schoolsup"] if c in reg_schema["feature_order"]]
    if not exclude_absences and "absences" in reg_schema["feature_order"]:
        key_input_cols.append("absences")

    summary_rows = [
        {"Variabel": LABELS.get(c, c), "Kode Kolom": c, "Nilai Dipilih": format_input_value(c, user_inputs[c])}
        for c in key_input_cols
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    other_cols = [c for c in reg_schema["feature_order"] if c not in key_input_cols]
    if other_cols:
        with st.expander("Lihat seluruh parameter input lainnya", expanded=False):
            other_rows = [
                {"Variabel": LABELS.get(c, c), "Kode Kolom": c, "Nilai Dipilih": format_input_value(c, user_inputs[c])}
                for c in other_cols
            ]
            st.dataframe(pd.DataFrame(other_rows), width="stretch", hide_index=True)


# =============================================================================
# TAB 2 - SEGMENTASI SISWA
# =============================================================================
with tab_seg:
    clu_inputs = dict(user_inputs)
    sc1, sc2 = st.columns([5, 7], gap="medium")
    with sc1:
        use_manual_g3 = st.toggle(
            "Gunakan nilai G3 aktual manual untuk segmentasi",
            value=False,
            key="toggle_manual_g3",
            help="Jika nonaktif, segmentasi otomatis memakai hasil prediksi G3 dari model regresi.",
        )
    with sc2:
        if use_manual_g3:
            manual_g3 = st.slider("Nilai Akhir (G3) Aktual", 0, 20, int(round(pred_g3_val)), key="manual_g3_val")
            clu_inputs[TARGET] = float(manual_g3)
            g3_source_txt = f"nilai G3 aktual manual ({manual_g3})"
        else:
            clu_inputs[TARGET] = round(pred_g3_val, 2)
            g3_source_txt = f"hasil prediksi G3 ({pred_g3_val:.1f})"

    X_c = to_frame(clu_inputs, clu_schema["feature_order"], clu_schema["numeric"].keys())
    cl = int(predict_cluster(clu_model, clu_schema, X_c)[0])

    names = clu_schema["cluster_names"]
    prof_df, df_all = RES_C["prof"], RES_C["df_all"]
    cl_size = int(RES_C["size"].loc[cl])
    cl_pct = cl_size / len(df_all) * 100
    cl_g3 = float(prof_df.loc[cl, "G3"])
    cl_pass = float(RES_C["pass_rates"].loc[cl] * 100)
    cl_tone = "emerald" if cl_g3 >= 12 else ("amber" if cl_g3 >= 9.5 else "rose")

    st.markdown(
        info_box(
            f"Hasil Segmentasi: {names.get(cl, f'Cluster {cl}')}",
            f"Segmentasi dihitung menggunakan <strong>{html.escape(g3_source_txt)}</strong> dan seluruh profil siswa dengan algoritma KMeans (k = 3). "
            "Kelompok ini dibaca sebagai kecenderungan deskriptif profil siswa pada data historis.",
            tone=cl_tone,
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(
        kpi_card("Jumlah Anggota Cluster", f"{cl_size} siswa", f"{cl_pct:.1f}% dari total {len(df_all)} siswa", accent="sky"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("Rata-rata G3 Cluster", f"{cl_g3:.1f} / 20", f"Rata-rata seluruh kelas: {df_all['G3'].mean():.1f}", accent=cl_tone),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card("Tingkat Kelulusan Cluster", f"{cl_pass:.0f}%", f"Persentase siswa dengan G3 >= {PASS_THRESHOLD}", accent="emerald" if cl_pass >= 70 else "amber"),
        unsafe_allow_html=True,
    )

    bullets = clu_schema["cluster_bullets"].get(cl, [])
    bullets_html = "<ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in bullets) + "</ul>"
    st.markdown(
        info_box(f"Karakteristik Utama Cluster {cl}", bullets_html, tone="sky"),
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 3 - DATA & EDA
# =============================================================================
with tab_eda:
    n_zero = int((df_raw[TARGET] == 0).sum())
    n_miss = int(df_raw.isna().sum().sum())

    e_k1, e_k2, e_k3, e_k4 = st.columns(4, gap="medium")
    e_k1.markdown(kpi_card("Total Siswa", f"{len(df_raw)}", f"Sumber: {html.escape(raw_name)}", accent="sky"), unsafe_allow_html=True)
    e_k2.markdown(kpi_card("Jumlah Variabel", f"{df_raw.shape[1]}", "30 fitur profil, G1, G2, G3", accent="sky"), unsafe_allow_html=True)
    e_k3.markdown(kpi_card("Kualitas Data", f"{n_miss} Missing", "0 baris duplikat penuh", accent="emerald"), unsafe_allow_html=True)
    e_k4.markdown(kpi_card("Kasus G3 = 0", f"{n_zero} siswa", f"{n_zero / len(df_raw):.1%} populasi (dipertahankan)", accent="amber"), unsafe_allow_html=True)

    # Baris 1 EDA: 2 grafik sejajar (sama tinggi, tanpa spacer hack)
    eda_left, eda_right = st.columns(2, gap="large")

    with eda_left:
        g3_grp = np.where(df_raw[TARGET] == 0, "G3 = 0 (38 siswa)", "G3 > 0 (357 siswa)")
        fig_hist = px.histogram(
            df_raw.assign(Kelompok=g3_grp),
            x=TARGET,
            color="Kelompok",
            nbins=21,
            color_discrete_map={"G3 = 0 (38 siswa)": COLORS["rose"], "G3 > 0 (357 siswa)": COLORS["accent"]},
        )
        fig_hist.add_vline(
            x=float(df_raw[TARGET].mean()),
            line_dash="dash",
            line_color="#E6EDF3",
            annotation_text=f"Rata-rata: {df_raw[TARGET].mean():.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#E6EDF3", size=11),
        )
        style_plotly(fig_hist, "Distribusi Nilai Akhir (G3)", height=360)
        fig_hist.update_xaxes(title="Nilai Akhir (G3)", range=[-0.6, 20.6])
        fig_hist.update_yaxes(title="Jumlah Siswa")
        st.plotly_chart(fig_hist, width="stretch", theme=None)
        st.caption("Lonjakan 38 siswa di angka 0 membuat distribusi menceng ke kiri (tanpa G3=0, rata-rata naik menjadi 11,52).")

    with eda_right:
        r_g2_g3 = float(df_raw["G2"].corr(df_raw[TARGET]))
        fig_scat = px.scatter(
            df_raw.assign(Status=np.where(df_raw[TARGET] == 0, "G3 = 0", "G3 > 0")),
            x="G2",
            y=TARGET,
            color="Status",
            color_discrete_map={"G3 = 0": COLORS["rose"], "G3 > 0": COLORS["accent"]},
            hover_data=["G1", "failures", "absences"],
        )
        fig_scat.add_trace(go.Scatter(
            x=[0, 20], y=[0, 20], mode="lines", name="Garis G3 = G2",
            line=dict(color="#8B949E", dash="dash"),
        ))
        style_plotly(fig_scat, f"Hubungan Nilai G2 terhadap G3 (r = {r_g2_g3:.3f})", height=360)
        fig_scat.update_xaxes(title="Nilai Periode 2 (G2)", range=[-0.6, 20.6])
        fig_scat.update_yaxes(title="Nilai Akhir (G3)", range=[-0.6, 20.6])
        st.plotly_chart(fig_scat, width="stretch", theme=None)
        st.caption("13 siswa dengan G2=0 seluruhnya berakhir G3=0; 25 siswa lainnya memiliki G2 > 0 namun turun ke G3=0.")

    # Baris 2 EDA: Boxplot faktor non-nilai pilihan pengguna
    section_title("Sebaran Nilai Akhir (G3) Berdasarkan Faktor Non-Nilai")
    box_feat = st.selectbox(
        "Pilih faktor non-nilai:",
        ["failures", "Medu", "higher", "studytime", "goout", "schoolsup", "address"],
        format_func=lambda c: f"{LABELS.get(c, c)} ({c})",
        key="eda_box_feat",
    )

    df_box = df_raw[[box_feat, TARGET]].copy()
    if box_feat in ORDINAL_LABELS:
        cats_sorted = sorted(df_box[box_feat].unique())
        df_box["Kategori"] = df_box[box_feat].map(lambda v: ORDINAL_LABELS[box_feat].get(int(v), str(v)))
        cat_order = [ORDINAL_LABELS[box_feat].get(int(v), str(v)) for v in cats_sorted]
    elif box_feat in VALUE_LABELS:
        cats_sorted = sorted(df_box[box_feat].unique())
        df_box["Kategori"] = df_box[box_feat].map(lambda v: VALUE_LABELS[box_feat].get(str(v), str(v)))
        cat_order = [VALUE_LABELS[box_feat].get(str(v), str(v)) for v in cats_sorted]
    else:
        df_box["Kategori"] = df_box[box_feat].astype(str)
        cat_order = sorted(df_box["Kategori"].unique())

    fig_box = px.box(
        df_box,
        x="Kategori",
        y=TARGET,
        category_orders={"Kategori": cat_order},
        points="outliers",
    )
    fig_box.update_traces(marker_color=COLORS["accent"], line_color=COLORS["accent"])
    style_plotly(fig_box, f"Sebaran G3 Menurut {LABELS.get(box_feat, box_feat)}", height=360, show_legend=False)
    fig_box.update_xaxes(title=LABELS.get(box_feat, box_feat))
    fig_box.update_yaxes(title="Nilai Akhir (G3)", range=[-0.6, 20.6])
    st.plotly_chart(fig_box, width="stretch", theme=None)

    with st.expander("Korelasi Spearman Fitur Numerik/Ordinal terhadap G3", expanded=False):
        num_eda_cols = df_raw.select_dtypes(include=np.number).columns.drop(TARGET)
        spearman_s = df_raw[num_eda_cols].corrwith(df_raw[TARGET], method="spearman").sort_values()
        fig_sp = go.Figure(go.Bar(
            x=spearman_s.values,
            y=[f"{LABELS.get(c, c)} ({c})" for c in spearman_s.index],
            orientation="h",
            marker_color=[COLORS["emerald"] if v > 0 else COLORS["rose"] for v in spearman_s.values],
            text=[f"{v:+.2f}" for v in spearman_s.values],
            textposition="outside",
        ))
        style_plotly(fig_sp, "Korelasi Peringkat Spearman terhadap G3", height=420, show_legend=False)
        fig_sp.update_xaxes(title="Koefisien Korelasi Spearman", range=[-0.55, 1.1])
        st.plotly_chart(fig_sp, width="stretch", theme=None)

    with st.expander("Catatan Cleaning & Pratinjau Dataset", expanded=False):
        st.markdown(
            """
            - **Keutuhan data**: 395 baris dan 33 kolom tanpa missing value maupun duplikasi baris penuh.
            - **Penanganan `G3 = 0` dan outlier**: 38 baris `G3 = 0` (9,6%) dan nilai `absences` ekstrem (maksimum 75) tetap dipertahankan karena valid secara domain dan masih berada dalam rentang dokumentasi (0-93).
            - **Pengecualian `absences` pada model regresi**: Seluruh 38 siswa dengan `G3 = 0` tercatat memiliki `absences = 0`, dan waktu pencatatannya tidak terinci, sehingga fitur `absences` dikeluarkan dari model regresi default untuk menghindari kebocoran informasi masa depan.
            - **Kode ordinal vs besaran asli**: Variabel seperti `studytime`, `traveltime`, `Medu`, `Fedu`, `famrel`, `goout`, `Dalc`, `Walc`, dan `health` merupakan kode kategori berurutan.
            """
        )
        st.dataframe(df_raw.head(50), width="stretch", hide_index=True)


# =============================================================================
# TAB 4 - PREDIKSI MASSAL
# =============================================================================
with tab_batch:
    section_title("Prediksi Massal dari File CSV atau Excel")
    st.caption("Unggah file berisi banyak data siswa untuk mendapatkan prediksi G3 dan segmentasi cluster sekaligus (terpisah dari input sidebar).")

    needed = []
    for sch in (reg_schema, clu_schema):
        needed += [c for c in sch["feature_order"] if c not in needed and c != TARGET]
    template = pd.DataFrame([user_inputs])[[c for c in needed if c in user_inputs]]

    b1, b2, b3 = st.columns(3, gap="medium")
    b1.download_button(
        "Unduh Template CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="template_siswa.csv",
        mime="text/csv",
        width="stretch",
    )
    if b2.button("Uji dengan 30 Data Sampel", width="stretch"):
        st.session_state["batch_demo_df"] = (
            df_raw.sample(n=min(30, len(df_raw)), random_state=RANDOM_STATE)
            .drop(columns=[TARGET])
            .reset_index(drop=True)
        )
    if "batch_demo_df" in st.session_state:
        if b3.button("Bersihkan Data Sampel", width="stretch"):
            del st.session_state["batch_demo_df"]

    up = st.file_uploader("Unggah file data siswa (.csv atau .xlsx)", type=["csv", "xlsx"], key="upload_batch")

    df_up = None
    source_label = ""
    if up is not None:
        try:
            df_up = pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv") else pd.read_excel(up)
            source_label = f"File unggahan: {up.name}"
        except Exception as e:
            st.error(f"File tidak dapat dibaca: {e}")
    elif "batch_demo_df" in st.session_state:
        df_up = st.session_state["batch_demo_df"].copy()
        source_label = "Data sampel (30 siswa acak tanpa kolom G3)"

    if df_up is not None:
        st.caption(f"Sumber aktif: {source_label} ({len(df_up)} baris, {df_up.shape[1]} kolom)")
        result = df_up.copy()
        front_cols = []

        miss_r = [c for c in reg_schema["feature_order"] if c not in df_up.columns]
        if miss_r:
            st.warning(f"Prediksi G3 dilewati karena kolom berikut tidak ditemukan: {', '.join(miss_r)}")
        else:
            result["Prediksi_G3"] = predict_g3(reg_model, clean_batch(df_up, reg_schema)).round(2)
            result["Status_Risiko"] = np.where(result["Prediksi_G3"] < reg_schema["pass_threshold"], "Berisiko", "Aman")
            front_cols += ["Prediksi_G3", "Status_Risiko"]

        df_for_clu = df_up.copy()
        if clu_schema["include_grades"] and TARGET not in df_for_clu.columns and "Prediksi_G3" in result.columns:
            df_for_clu[TARGET] = result["Prediksi_G3"]

        miss_c = [c for c in clu_schema["feature_order"] if c not in df_for_clu.columns]
        if miss_c:
            st.warning(f"Segmentasi dilewati karena kolom berikut tidak ditemukan: {', '.join(miss_c)}")
        else:
            labs = predict_cluster(clu_model, clu_schema, clean_batch(df_for_clu, clu_schema))
            result["Cluster"] = labs
            result["Nama_Cluster"] = [clu_schema["cluster_names"].get(int(i), f"Cluster {i}") for i in labs]
            front_cols += ["Cluster", "Nama_Cluster"]

        if "Prediksi_G3" in result.columns:
            n_tot = len(result)
            n_risk = int((result["Status_Risiko"] == "Berisiko").sum())
            pct_risk = n_risk / max(n_tot, 1) * 100
            avg_p = float(result["Prediksi_G3"].mean())

            bk1, bk2, bk3 = st.columns(3, gap="medium")
            bk1.markdown(kpi_card("Total Siswa Diproses", f"{n_tot}", "Baris valid terprediksi", accent="sky"), unsafe_allow_html=True)
            bk2.markdown(kpi_card("Rata-rata Prediksi G3", f"{avg_p:.2f} / 20", f"Median: {result['Prediksi_G3'].median():.2f}", accent="sky"), unsafe_allow_html=True)
            bk3.markdown(kpi_card("Siswa Berisiko (< 10)", f"{n_risk} siswa", f"{pct_risk:.1f}% memerlukan perhatian", accent="rose" if n_risk > 0 else "emerald"), unsafe_allow_html=True)

        ordered_cols = front_cols + [c for c in result.columns if c not in front_cols]
        result = result[ordered_cols]

        view_df = result.copy()
        if "Status_Risiko" in view_df.columns:
            filt_risk = st.selectbox("Filter status risiko:", ["Semua", "Berisiko", "Aman"], key="batch_f_risk")
            if filt_risk != "Semua":
                view_df = view_df[view_df["Status_Risiko"] == filt_risk]

        st.dataframe(view_df, width="stretch", hide_index=True)
        st.download_button(
            "Unduh Hasil Prediksi (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_siswa.csv",
            mime="text/csv",
        )
