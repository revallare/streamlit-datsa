# =============================================================================
#  PREDIKSI NILAI AKHIR MATEMATIKA SISWA MENGGUNAKAN MACHINE LEARNING
#  DAN IMPLEMENTASI STREAMLIT — KELOMPOK 2 (MATA KULIAH DATA SAINS)
#
#  Satu file terpadu berisi: pemuatan data -> pelatihan model regresi &
#  clustering (train-test split, pemilihan algoritma, tuning, evaluasi,
#  interpretasi) -> aplikasi web interaktif bertema gelap.
#
#  Struktur repo untuk deploy (Streamlit Community Cloud / lokal):
#      app_gabungan.py        <- file ini
#      student-mat.xlsx       <- dataset (jika tidak ada, aplikasi meminta upload)
#      requirements.txt       <- daftar dependensi
#
#  (Opsional) Untuk tema gelap native Streamlit tanpa jeda transisi,
#  tambahkan file .streamlit/config.toml berisi:
#      [theme]
#      base="dark"
#      primaryColor="#6366F1"
#      backgroundColor="#0B0F19"
#      secondaryBackgroundColor="#111827"
#      textColor="#F8FAFC"
#
#  Jalankan lokal:  streamlit run app_gabungan.py
# =============================================================================
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import kruskal

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# =============================================================================
#  KONFIGURASI HALAMAN & IDENTITAS PROYEK
# =============================================================================
APP_TITLE = "Prediksi Nilai Akhir Matematika Siswa Menggunakan Machine Learning dan Implementasi Streamlit"
COURSE_INFO = "Kelompok 2 · Mata Kuliah Data Sains"
TEAM_MEMBERS = [
    ("Khenya Aghisna Sulihanyalevi", "M0124011"),
    ("Andhika Rian Wira", "M0124022"),
    ("Irene Galuh Aditya Putri", "M0124035"),
    ("Kurrotul Hana", "M0124037"),
    ("Reni Putri Anggraini", "M0124046"),
    ("Reval Aulia Rahman Erlangga", "M0124047"),
]

st.set_page_config(
    page_title="Prediksi Nilai Akhir Matematika Siswa — Kelompok 2",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PASS_THRESHOLD = 10

MODE_CFG = {
    "Cepat": dict(
        cv_repeats=1, n_iter=8, n_tune=2, perm_repeats=10, boot=10,
        learning_curve=False, trees=150, km_init=10, gmm_init=2, final_init=20,
    ),
    "Lengkap": dict(
        cv_repeats=2, n_iter=25, n_tune=3, perm_repeats=30, boot=30,
        learning_curve=True, trees=300, km_init=20, gmm_init=5, final_init=50,
    ),
}

# =============================================================================
#  PALET & CUSTOM CSS (TEMA GELAP PROFESIONAL)
# =============================================================================
COLORS = {
    "bg": "#0B0F19",
    "surface": "#111827",
    "card": "#161E2E",
    "card_alt": "#1B2538",
    "border": "rgba(148, 163, 184, 0.16)",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "subtle": "#64748B",
    "primary": "#6366F1",
    "primary_light": "#818CF8",
    "sky": "#38BDF8",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "violet": "#A78BFA",
}
CLUSTER_PALETTE = ["#38BDF8", "#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#A78BFA", "#EC4899", "#14B8A6"]


def inject_dark_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: #0B0F19;
            color: #F8FAFC;
        }
        .stApp {
            background: radial-gradient(circle at 12% 0%, rgba(99, 102, 241, 0.09), transparent 38%),
                        radial-gradient(circle at 88% 8%, rgba(56, 189, 248, 0.06), transparent 32%),
                        #0B0F19;
        }
        [data-testid="stHeader"] {
            background-color: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
        }
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        /* Hero Header */
        .hero-box {
            background: linear-gradient(135deg, rgba(22, 30, 46, 0.96) 0%, rgba(17, 24, 39, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-left: 4px solid #6366F1;
            border-radius: 14px;
            padding: 1.35rem 1.65rem;
            margin-bottom: 1.5rem;
        }
        .hero-kicker {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #818CF8;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            font-size: 1.48rem;
            font-weight: 700;
            line-height: 1.32;
            color: #F8FAFC;
            margin: 0 0 0.65rem 0;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .pill {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 500;
            color: #CBD5E1;
            background: rgba(148, 163, 184, 0.10);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 0.2rem 0.72rem;
        }
        .pill-accent {
            color: #C7D2FE;
            background: rgba(99, 102, 241, 0.16);
            border-color: rgba(99, 102, 241, 0.35);
        }

        /* Section Titles */
        .sec-kicker {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #818CF8;
            margin-bottom: 0.15rem;
        }
        .sec-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #F8FAFC;
            margin: 0 0 0.25rem 0;
        }
        .sec-sub {
            font-size: 0.86rem;
            color: #94A3B8;
            margin: 0 0 1rem 0;
        }

        /* KPI Cards */
        .kpi-card {
            background: #161E2E;
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            height: 100%;
        }
        .kpi-label {
            font-size: 0.74rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #F8FAFC;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: #94A3B8;
            margin-top: 0.32rem;
        }
        .badge-safe {
            display: inline-block;
            padding: 0.22rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #34D399;
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.32);
        }
        .badge-risk {
            display: inline-block;
            padding: 0.22rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #FB7185;
            background: rgba(244, 63, 94, 0.14);
            border: 1px solid rgba(244, 63, 94, 0.32);
        }
        .badge-info {
            display: inline-block;
            padding: 0.22rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #7DD3FC;
            background: rgba(56, 189, 248, 0.14);
            border: 1px solid rgba(56, 189, 248, 0.32);
        }

        /* Callout Cards */
        .callout {
            background: #161E2E;
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-left: 3px solid #6366F1;
            border-radius: 10px;
            padding: 0.9rem 1.15rem;
            margin: 0.75rem 0;
            font-size: 0.88rem;
            color: #CBD5E1;
            line-height: 1.55;
        }
        .callout-safe { border-left-color: #10B981; background: rgba(16, 185, 129, 0.06); }
        .callout-risk { border-left-color: #F43F5E; background: rgba(244, 63, 94, 0.06); }
        .callout-warn { border-left-color: #F59E0B; background: rgba(245, 158, 11, 0.06); }
        .callout strong { color: #F8FAFC; }

        /* Sidebar Brand & Status */
        .sb-brand {
            padding: 0.35rem 0 0.85rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 0.9rem;
        }
        .sb-kicker {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #818CF8;
        }
        .sb-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-top: 0.18rem;
            line-height: 1.3;
        }
        .sb-status {
            background: #161E2E;
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
            font-size: 0.78rem;
            color: #CBD5E1;
            line-height: 1.6;
        }
        .sb-status-title {
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }

        /* Team Grid */
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 0.85rem;
            margin-top: 0.6rem;
        }
        .member-card {
            background: #161E2E;
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 11px;
            padding: 0.9rem 1.05rem;
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .member-idx {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: rgba(99, 102, 241, 0.18);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #A5B4FC;
            font-size: 0.82rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .member-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: #F8FAFC;
            line-height: 1.3;
        }
        .member-nim {
            font-size: 0.78rem;
            color: #94A3B8;
            font-variant-numeric: tabular-nums;
            margin-top: 0.12rem;
        }

        /* Streamlit Widget Polish */
        div[data-testid="stForm"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
        }
        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
            border-radius: 9px;
            font-weight: 600;
            font-size: 0.86rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background-color: #1B2538;
            color: #F8FAFC;
            transition: all 0.15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #6366F1;
            background-color: #232F46;
            color: #FFFFFF;
        }
        .stFormSubmitButton > button[kind="primary"],
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
            border: 1px solid #818CF8;
            color: #FFFFFF;
        }
        .stFormSubmitButton > button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #94A3B8;
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.55rem 0.95rem;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #F8FAFC !important;
            background-color: rgba(99, 102, 241, 0.12);
            border-bottom: 2px solid #6366F1 !important;
        }
        [data-testid="stExpander"] {
            background-color: #161E2E;
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 10px;
        }
        .app-footer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.14);
            font-size: 0.78rem;
            color: #64748B;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_dark_css()

# =============================================================================
#  METADATA FITUR (SESUAI DOKUMENTASI UCI & BRIEF STUDENT PERFORMANCE)
# =============================================================================
LABELS = {
    "school": "Sekolah",
    "sex": "Jenis kelamin",
    "age": "Usia (tahun)",
    "address": "Area tempat tinggal",
    "famsize": "Ukuran keluarga",
    "Pstatus": "Status tinggal orang tua",
    "Medu": "Pendidikan ibu",
    "Fedu": "Pendidikan ayah",
    "Mjob": "Pekerjaan ibu",
    "Fjob": "Pekerjaan ayah",
    "reason": "Alasan memilih sekolah",
    "guardian": "Wali siswa",
    "traveltime": "Waktu tempuh ke sekolah",
    "studytime": "Waktu belajar mingguan",
    "failures": "Riwayat gagal kelas sebelumnya",
    "schoolsup": "Bimbingan tambahan sekolah",
    "famsup": "Dukungan belajar keluarga",
    "paid": "Les berbayar Matematika",
    "activities": "Kegiatan ekstrakurikuler",
    "nursery": "Pernah pendidikan prasekolah (TK)",
    "higher": "Berencana melanjutkan kuliah",
    "internet": "Akses internet di rumah",
    "romantic": "Menjalin hubungan romantis",
    "famrel": "Kualitas hubungan keluarga",
    "freetime": "Waktu luang setelah sekolah",
    "goout": "Intensitas keluar bersama teman",
    "Dalc": "Konsumsi alkohol hari kerja",
    "Walc": "Konsumsi alkohol akhir pekan",
    "health": "Kondisi kesehatan saat ini",
    "absences": "Jumlah ketidakhadiran",
    "G1": "Nilai periode 1 (G1)",
    "G2": "Nilai periode 2 (G2)",
    "G3": "Nilai akhir (G3)",
}

HELP = {
    "school": "GP = Gabriel Pereira, MS = Mousinho da Silveira.",
    "failures": "Nilai aktual dataset 0, 1, 2, atau 3 (3 berarti 3 kali atau lebih).",
    "absences": "Total ketidakhadiran sekolah (0–93). Default dikecualikan sesuai keputusan cleaning karena waktu pencatatan tidak terinci.",
    "G1": "Nilai ujian Matematika periode pertama (skala 0–20).",
    "G2": "Nilai ujian Matematika periode kedua (skala 0–20).",
}

# Label deskriptif untuk variabel ordinal agar pengguna tidak melihat angka kode mentah
ORDINAL_LABELS = {
    "Medu": {0: "0 — Tidak bersekolah", 1: "1 — SD (Kelas 4)", 2: "2 — SMP (Kelas 5–9)", 3: "3 — SMA / Menengah", 4: "4 — Pendidikan Tinggi"},
    "Fedu": {0: "0 — Tidak bersekolah", 1: "1 — SD (Kelas 4)", 2: "2 — SMP (Kelas 5–9)", 3: "3 — SMA / Menengah", 4: "4 — Pendidikan Tinggi"},
    "traveltime": {1: "1 — < 15 menit", 2: "2 — 15–30 menit", 3: "3 — 30–60 menit", 4: "4 — > 60 menit"},
    "studytime": {1: "1 — < 2 jam/minggu", 2: "2 — 2–5 jam/minggu", 3: "3 — 5–10 jam/minggu", 4: "4 — > 10 jam/minggu"},
    "failures": {0: "0 — Tidak pernah", 1: "1 — 1 kali", 2: "2 — 2 kali", 3: "3 — 3 kali atau lebih"},
    "famrel": {1: "1 — Sangat buruk", 2: "2 — Buruk", 3: "3 — Cukup", 4: "4 — Baik", 5: "5 — Sangat baik"},
    "freetime": {1: "1 — Sangat sedikit", 2: "2 — Sedikit", 3: "3 — Sedang", 4: "4 — Banyak", 5: "5 — Sangat banyak"},
    "goout": {1: "1 — Sangat jarang", 2: "2 — Jarang", 3: "3 — Sedang", 4: "4 — Sering", 5: "5 — Sangat sering"},
    "Dalc": {1: "1 — Sangat rendah", 2: "2 — Rendah", 3: "3 — Sedang", 4: "4 — Tinggi", 5: "5 — Sangat tinggi"},
    "Walc": {1: "1 — Sangat rendah", 2: "2 — Rendah", 3: "3 — Sedang", 4: "4 — Tinggi", 5: "5 — Sangat tinggi"},
    "health": {1: "1 — Sangat buruk", 2: "2 — Buruk", 3: "3 — Cukup", 4: "4 — Baik", 5: "5 — Sangat baik"},
}

VALUE_LABELS = {
    "school": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"},
    "sex": {"F": "Perempuan", "M": "Laki-laki"},
    "address": {"U": "Perkotaan", "R": "Pedesaan"},
    "famsize": {"LE3": "Maksimal 3 orang (LE3)", "GT3": "Lebih dari 3 orang (GT3)"},
    "Pstatus": {"T": "Tinggal bersama", "A": "Terpisah"},
    "guardian": {"mother": "Ibu", "father": "Ayah", "other": "Lainnya"},
    "reason": {"home": "Dekat rumah", "reputation": "Reputasi sekolah", "course": "Pilihan program", "other": "Lainnya"},
    "Mjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Pelayanan publik", "at_home": "Di rumah", "other": "Lainnya"},
    "Fjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Pelayanan publik", "at_home": "Di rumah", "other": "Lainnya"},
}
for _c in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
    VALUE_LABELS[_c] = {"yes": "Ya", "no": "Tidak"}

GROUPS = {
    "Akademik": ["G1", "G2", "failures", "studytime", "absences", "schoolsup", "paid", "higher", "traveltime", "reason", "nursery"],
    "Keluarga": ["Medu", "Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"],
    "Demografi": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "Gaya Hidup": ["goout", "freetime", "Dalc", "Walc", "activities", "internet", "romantic", "health"],
}

COMPACT_FEATURES = [
    "G1", "G2", "failures", "studytime", "absences", "age",
    "higher", "schoolsup", "paid", "goout", "Medu", "famsup",
]


# =============================================================================
#  HELPER VISUALISASI PLOTLY & KOMPONEN UI
# =============================================================================
def style_plotly(fig: go.Figure, height: int = 380, title: str = None, margin: dict = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        title=dict(text=title, font=dict(size=14, color=COLORS["text"]), x=0.0, xanchor="left") if title else None,
        margin=margin or dict(l=20, r=20, t=44 if title else 24, b=24),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["muted"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#475569", font=dict(color="#F8FAFC", size=12)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.22)")
    return fig


def section_header(kicker: str, title: str, subtitle: str = ""):
    sub_html = f'<p class="sec-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="sec-kicker">{kicker}</div>'
        f'<h3 class="sec-title">{title}</h3>'
        f"{sub_html}",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", badge_html: str = ""):
    val_block = badge_html if badge_html else f'<div class="kpi-value">{value}</div>'
    sub_block = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f"{val_block}{sub_block}"
        f"</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  BAGIAN A - PEMUATAN DATA
# =============================================================================
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
    for fname in ("student-mat.xlsx", "student-mat.csv"):
        p = BASE / fname
        if p.exists():
            return p.read_bytes(), fname
    st.warning("File dataset `student-mat.xlsx` tidak ditemukan di direktori aplikasi. Unggah dataset untuk melanjutkan.")
    up = st.file_uploader("Unggah student-mat.xlsx atau student-mat.csv", type=["xlsx", "csv"], key="upload_dataset")
    if up is None:
        st.stop()
    return up.getvalue(), up.name


raw_bytes, raw_name = get_raw_data()
df_raw = read_data(raw_bytes, raw_name)
missing_cols = [c for c in LABELS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Dataset tidak valid, kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
    st.stop()
DATA_KEY = f"{raw_name}-{df_raw.shape}-{int(pd.util.hash_pandas_object(df_raw, index=True).sum() % 10**12)}"


# =============================================================================
#  BAGIAN B - FUNGSI BANTU PEMODELAN
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
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
            ]),
            cat_cols,
        ),
    ])


def regression_report(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    denom = max(n - n_features - 1, 1)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
        "Adj. R2": 1 - (1 - r2) * (n - 1) / denom,
        "Akurasi ±1 poin": float(np.mean(np.abs(np.asarray(y_true) - y_pred) <= 1)),
        "Akurasi ±2 poin": float(np.mean(np.abs(np.asarray(y_true) - y_pred) <= 2)),
    }


def eval_labels(Z, labels):
    if len(set(labels)) < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan)
    return dict(
        silhouette=float(silhouette_score(Z, labels)),
        calinski=float(calinski_harabasz_score(Z, labels)),
        davies=float(davies_bouldin_score(Z, labels)),
    )


# =============================================================================
#  BAGIAN C - PELATIHAN MODEL REGRESI (DI-CACHE)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_regression(_df, data_key, use_prev_grades, include_absences, mode):
    cfg = MODE_CFG[mode]
    drop_cols = [TARGET]
    if not use_prev_grades:
        drop_cols += ["G1", "G2"]
    if not include_absences and "absences" in _df.columns:
        drop_cols.append("absences")

    X = _df.drop(columns=drop_cols).copy()
    y = _df[TARGET].astype(float)
    num_cols, cat_cols = split_columns(X)
    X[num_cols] = X[num_cols].astype(float)

    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins,
    )

    def P(scale):
        return make_preprocessor(num_cols, cat_cols, scale)

    models = {
        "Baseline (Rata-rata)": DummyRegressor(strategy="mean"),
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
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([("pre", P(False)), ("m", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1))])
    except Exception:
        pass

    cv = RepeatedKFold(n_splits=5, n_repeats=cfg["cv_repeats"], random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows = []
    for name, mdl in models.items():
        r = cross_validate(mdl, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "Model": name,
            "CV R2": float(r["test_r2"].mean()),
            "CV R2 std": float(r["test_r2"].std()),
            "CV MAE": float(-r["test_mae"].mean()),
            "CV RMSE": float(-r["test_rmse"].mean()),
        })
    bench = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)

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
        search = RandomizedSearchCV(
            models[name], param_spaces[name], n_iter=cfg["n_iter"],
            scoring="neg_root_mean_squared_error", cv=5,
            random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        tune_rows.append({
            "Model": name,
            "CV RMSE (Tuned)": float(-search.best_score_),
            "Parameter Terbaik": str({
                k.replace("m__", ""): (round(v, 4) if isinstance(v, float) else v)
                for k, v in search.best_params_.items()
            }),
        })
    tune_df = pd.DataFrame(tune_rows).sort_values("CV RMSE (Tuned)").reset_index(drop=True)
    best_name = tune_df.loc[0, "Model"]
    best_model = tuned[best_name]

    pred_train = np.clip(best_model.predict(X_train), 0, 20)
    pred_test = np.clip(best_model.predict(X_test), 0, 20)
    rep = pd.DataFrame({
        "Train": regression_report(y_train, pred_train, X.shape[1]),
        "Test": regression_report(y_test, pred_test, X.shape[1]),
        "Baseline (Test)": regression_report(y_test, np.full(len(y_test), y_train.mean()), X.shape[1]),
    }).T
    resid = y_test.values - pred_test

    act_pass, pred_pass = (y_test >= PASS_THRESHOLD), (pred_test >= PASS_THRESHOLD)
    tp = int(((~act_pass) & (~pred_pass)).sum())
    fn = int(((~act_pass) & pred_pass).sum())
    fp = int((act_pass & (~pred_pass)).sum())
    tn = int((act_pass & pred_pass).sum())
    risk = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }

    lc_data = None
    if cfg["learning_curve"]:
        sizes, tr_s, va_s = learning_curve(
            best_model, X_train, y_train, cv=5, n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.2, 1.0, 6), random_state=RANDOM_STATE,
        )
        lc_data = {"sizes": sizes.tolist(), "train_rmse": (-tr_s.mean(1)).tolist(), "val_rmse": (-va_s.mean(1)).tolist()}

    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=cfg["perm_repeats"],
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = (
        pd.DataFrame({
            "Fitur": X.columns,
            "Label": [LABELS.get(c, c) for c in X.columns],
            "Kenaikan RMSE": perm.importances_mean,
            "Std": perm.importances_std,
        })
        .sort_values("Kenaikan RMSE", ascending=False)
        .reset_index(drop=True)
    )

    top_num = [f for f in imp["Fitur"] if f in num_cols][:3]
    pdp_data = []
    for f in top_num:
        try:
            pd_res = partial_dependence(best_model, X_train, [f], grid_resolution=20)
            grid_vals = pd_res["grid_values"][0] if "grid_values" in pd_res else pd_res["values"][0]
            avg_vals = pd_res["average"][0]
            pdp_data.append({"feature": f, "label": LABELS.get(f, f), "grid": grid_vals.tolist(), "avg": avg_vals.tolist()})
        except Exception:
            pass

    coef = None
    final_est = best_model.named_steps["m"]
    if hasattr(final_est, "coef_"):
        names = best_model.named_steps["pre"].get_feature_names_out()
        coef = pd.Series(np.ravel(final_est.coef_), index=names).sort_values(key=np.abs, ascending=False).head(10)

    mask0 = y_test.values == 0
    mae_zero = float(np.abs(resid[mask0]).mean()) if mask0.sum() else None
    mae_other = float(np.abs(resid[~mask0]).mean()) if (~mask0).sum() else None

    schema = {
        "model_name": best_name,
        "target": TARGET,
        "use_prev_grades": bool(use_prev_grades),
        "include_absences": bool(include_absences),
        "feature_order": X.columns.tolist(),
        "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())} for c in num_cols},
        "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "metrics_test": {k: float(v) for k, v in rep.loc["Test"].items()},
        "top_features": imp.head(10)["Fitur"].tolist(),
        "pass_threshold": PASS_THRESHOLD,
    }
    return dict(
        model=best_model, schema=schema, bench=bench, tune=tune_df, report=rep, importance=imp,
        risk=risk, coef=coef, y_test=y_test.values, pred_test=pred_test, resid=resid,
        lc_data=lc_data, pdp_data=pdp_data, n_train=len(X_train), n_test=len(X_test),
        n_features=X.shape[1], num_cols=num_cols, cat_cols=cat_cols,
        mae_zero=mae_zero, mae_other=mae_other, tunable=tunable,
        gap_r2=float(rep.loc["Train", "R2"] - rep.loc["Test", "R2"]),
    )


# =============================================================================
#  BAGIAN D - PELATIHAN MODEL CLUSTERING (DI-CACHE)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_clustering(_df, data_key, include_grades, include_absences, min_k_final, mode):
    cfg = MODE_CFG[mode]
    drop_cols = []
    if not include_grades:
        drop_cols += ["G1", "G2", "G3"]
    if not include_absences and "absences" in _df.columns:
        drop_cols.append("absences")

    feat_df = _df.drop(columns=drop_cols).copy()
    num_cols, cat_cols = split_columns(feat_df)
    feat_df[num_cols] = feat_df[num_cols].astype(float)

    preprocessor = make_preprocessor(num_cols, cat_cols, scale=True)
    transformer = Pipeline([("pre", preprocessor), ("pca", PCA(n_components=0.80, random_state=RANDOM_STATE))])

    df_train, df_test = train_test_split(feat_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Z_train = transformer.fit_transform(df_train)
    Z_test = transformer.transform(df_test)

    rows = []
    for k in range(2, 9):
        algos = {
            "KMeans": KMeans(n_clusters=k, n_init=cfg["km_init"], random_state=RANDOM_STATE),
            "GMM": GaussianMixture(n_components=k, covariance_type="diag", n_init=cfg["gmm_init"], random_state=RANDOM_STATE),
            "Agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        }
        for name, mdl in algos.items():
            lab = mdl.fit_predict(Z_train)
            m = eval_labels(Z_train, lab)
            m.update(
                Algoritma=name, k=k,
                inertia=float(mdl.inertia_) if name == "KMeans" else np.nan,
                bic=float(mdl.bic(Z_train)) if name == "GMM" else np.nan,
            )
            rows.append(m)
    sel = pd.DataFrame(rows)
    sel["rank_score"] = (
        sel["silhouette"].rank(ascending=False)
        + sel["calinski"].rank(ascending=False)
        + sel["davies"].rank(ascending=True)
    )

    nn = NearestNeighbors(n_neighbors=10).fit(Z_train)
    kdist = np.sort(nn.kneighbors(Z_train)[0][:, -1])
    db_rows = []
    for eps in np.quantile(kdist, [0.5, 0.6, 0.7, 0.8, 0.9]):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z_train)
        mask = lab != -1
        n_cl = len(set(lab[mask]))
        sil = float(silhouette_score(Z_train[mask], lab[mask])) if n_cl >= 2 and mask.sum() > n_cl else np.nan
        db_rows.append({
            "eps": round(float(eps), 3),
            "n_cluster": n_cl,
            "noise (%)": round(100 * float((~mask).mean()), 1),
            "silhouette (non-noise)": sil,
        })
    dbscan_df = pd.DataFrame(db_rows)

    deployable = sel[sel.Algoritma.isin(["KMeans", "GMM"]) & (sel.k >= min_k_final)]
    best = deployable.sort_values("rank_score").iloc[0]
    best_algo, best_k = best["Algoritma"], int(best["k"])

    def build_model(algo, k):
        if algo == "KMeans":
            return KMeans(n_clusters=k, n_init=cfg["final_init"], random_state=RANDOM_STATE)
        return GaussianMixture(
            n_components=k, covariance_type="diag",
            n_init=max(cfg["gmm_init"], 5), random_state=RANDOM_STATE,
        )

    final_model = build_model(best_algo, best_k).fit(Z_train)
    final_pipeline = Pipeline(transformer.steps + [("cluster", final_model)])

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

    prof_cols = _df.select_dtypes(include=np.number).columns.tolist()
    prof = df_all.groupby("cluster")[prof_cols].mean()
    z_prof = (prof - df_all[prof_cols].mean()) / df_all[prof_cols].std()

    cat_report = []
    for c in cat_cols:
        ct = pd.crosstab(df_all["cluster"], df_all[c], normalize="index")
        overall = df_all[c].value_counts(normalize=True)
        for cl in ct.index:
            for cat in ct.columns:
                cat_report.append({
                    "cluster": cl, "fitur": LABELS.get(c, c), "kategori": VALUE_LABELS.get(c, {}).get(cat, str(cat)),
                    "proporsi_cluster": ct.loc[cl, cat], "proporsi_populasi": overall[cat],
                    "lift": ct.loc[cl, cat] / overall[cat],
                })
    cat_report = pd.DataFrame(cat_report)
    notable = cat_report[(cat_report["lift"] >= 1.3) & (cat_report["proporsi_cluster"] >= 0.20)]
    notable = notable.sort_values(["cluster", "lift"], ascending=[True, False]).groupby("cluster").head(5)

    H, p = kruskal(*[g["G3"].values for _, g in df_all.groupby("cluster")])

    def label_cluster(cl):
        z = z_prof.loc[cl]
        traits = []
        if z.get("G3", 0) > 0.5:
            traits.append("Performa Tinggi")
        elif z.get("G3", 0) < -0.5:
            traits.append("Performa Rendah")
        else:
            traits.append("Performa Menengah")
        if z.get("failures", 0) > 0.5:
            traits.append("riwayat gagal tinggi")
        if z.get("absences", 0) > 0.5:
            traits.append("absen tinggi")
        if z.get("studytime", 0) > 0.5:
            traits.append("waktu belajar tinggi")
        if z.get("Dalc", 0) + z.get("Walc", 0) > 1.0:
            traits.append("konsumsi alkohol tinggi")
        if z.get("goout", 0) > 0.5:
            traits.append("sosialisasi tinggi")
        return " · ".join(traits)

    cluster_names = {int(cl): f"Cluster {cl} — {label_cluster(cl)}" for cl in prof.index}
    schema = {
        "algorithm": best_algo,
        "k": best_k,
        "include_grades": bool(include_grades),
        "include_absences": bool(include_absences),
        "feature_order": feat_df.columns.tolist(),
        "numeric": {c: {"min": float(feat_df[c].min()), "max": float(feat_df[c].max()), "median": float(feat_df[c].median())} for c in num_cols},
        "categorical": {c: sorted(feat_df[c].dropna().unique().tolist()) for c in cat_cols},
        "label_map": {int(k): int(v) for k, v in remap.items()},
        "cluster_names": cluster_names,
        "cluster_sizes": {int(k): int(v) for k, v in size.items()},
        "metrics": {
            "silhouette_train": float(ev.loc["Train", "silhouette"]),
            "silhouette_test": float(ev.loc["Test", "silhouette"]),
            "davies_bouldin_test": float(ev.loc["Test", "davies"]),
            "calinski_test": float(ev.loc["Test", "calinski"]),
            "stability_ari": float(ari.mean()),
        },
    }
    return dict(
        model=final_pipeline, schema=schema, sel=sel, dbscan=dbscan_df, ev=ev, ari=ari,
        rand_sil=rand_sil, df_all=df_all, prof=prof, z_prof=z_prof, size=size, notable=notable,
        kruskal_H=float(H), kruskal_p=float(p), P2=P2, pca2_var=pca2.explained_variance_ratio_.tolist(),
        sample_sil=sample_sil, n_train=len(df_train), n_test=len(df_test),
        n_pc=Z_train.shape[1], var_pc=float(transformer.named_steps["pca"].explained_variance_ratio_.sum()),
    )


# =============================================================================
#  BAGIAN E - FUNGSI BANTU APLIKASI & PRESET
# =============================================================================
def is_whole(x) -> bool:
    return float(x).is_integer()


def merged_input_specs(reg_schema: dict, clu_schema: dict):
    """Gabungkan fitur input dari regresi & clustering, KECUALI target G3."""
    numeric, categorical, order = {}, {}, []
    for sch in (reg_schema, clu_schema):
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


def compute_default_profile(df: pd.DataFrame, order, numeric, categorical) -> dict:
    prof = {}
    for c in order:
        if c in numeric:
            v = numeric[c]
            med = v["median"]
            prof[c] = int(round(med)) if (is_whole(v["min"]) and is_whole(v["max"])) else float(med)
        else:
            mode_val = df[c].mode().iloc[0] if c in df.columns and not df[c].mode().empty else categorical[c][0]
            prof[c] = mode_val if mode_val in categorical[c] else categorical[c][0]
    return prof


def build_presets(base_profile: dict, order: list) -> dict:
    def apply_overrides(overrides: dict) -> dict:
        p = base_profile.copy()
        for k, v in overrides.items():
            if k in order:
                p[k] = v
        return p

    return {
        "Rata-rata Kelas": base_profile.copy(),
        "Siswa Berprestasi": apply_overrides({
            "G1": 16, "G2": 17, "failures": 0, "studytime": 3, "higher": "yes",
            "schoolsup": "no", "famsup": "yes", "paid": "yes", "goout": 2,
            "Dalc": 1, "Walc": 1, "Medu": 4, "Fedu": 4, "absences": 2,
        }),
        "Siswa Perlu Pendampingan": apply_overrides({
            "G1": 7, "G2": 6, "failures": 2, "studytime": 1, "higher": "no",
            "schoolsup": "yes", "famsup": "no", "paid": "no", "goout": 4,
            "Dalc": 3, "Walc": 4, "Medu": 1, "Fedu": 1, "absences": 14,
        }),
        "Siswa Menengah (Batas Lulus)": apply_overrides({
            "G1": 10, "G2": 10, "failures": 1, "studytime": 2, "higher": "yes",
            "schoolsup": "no", "famsup": "yes", "goout": 3, "absences": 6,
        }),
    }


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


def render_feature_widget(col: str, numeric: dict, categorical: dict, current_val):
    label = LABELS.get(col, col)
    help_txt = HELP.get(col)

    if col in ORDINAL_LABELS and col in numeric:
        ord_map = ORDINAL_LABELS[col]
        lo, hi = int(numeric[col]["min"]), int(numeric[col]["max"])
        opts = [k for k in sorted(ord_map.keys()) if lo <= k <= hi]
        if not opts:
            opts = sorted(ord_map.keys())
        cur = int(round(current_val)) if current_val in opts else opts[min(len(opts) // 2, len(opts) - 1)]
        idx = opts.index(cur)
        return st.selectbox(
            label, opts, index=idx, help=help_txt,
            format_func=lambda x, m=ord_map: m.get(x, str(x)),
        )

    if col in numeric:
        v = numeric[col]
        lo, hi = v["min"], v["max"]
        if is_whole(lo) and is_whole(hi):
            lo_i, hi_i = int(lo), int(hi)
            if hi_i <= lo_i:
                hi_i = lo_i + 1
            cur_i = int(min(max(round(float(current_val)), lo_i), hi_i))
            return st.slider(label, lo_i, hi_i, cur_i, help=help_txt)
        if hi <= lo:
            hi = lo + 1.0
        cur_f = float(min(max(float(current_val), lo), hi))
        return st.slider(label, float(lo), float(hi), cur_f, help=help_txt)

    opts = categorical[col]
    fmt = VALUE_LABELS.get(col, {})
    idx = opts.index(current_val) if current_val in opts else 0
    return st.selectbox(
        label, opts, index=idx, help=help_txt,
        format_func=lambda x, f=fmt: f.get(x, str(x)),
    )


# =============================================================================
#  BAGIAN F - SIDEBAR: NAVIGASI & PENGATURAN MODEL
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-kicker">Data Sains · Kelompok 2</div>'
        '<div class="sb-title">Analisis Performa Matematika</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigasi",
        [
            "Prediksi & Segmentasi",
            "Prediksi Massal",
            "Evaluasi Regresi",
            "Evaluasi Clustering",
            "Data, Unduhan & Tentang",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    with st.expander("Pengaturan Model", expanded=False):
        with st.form("model_settings"):
            f_prev = st.checkbox(
                "Regresi: gunakan G1 & G2",
                value=True,
                help="Nonaktifkan untuk simulasi peringatan dini tanpa nilai periode sebelumnya.",
            )
            f_abs = st.checkbox(
                "Sertakan fitur absences",
                value=False,
                help="Default nonaktif mengikuti keputusan cleaning (waktu pencatatan absences tidak terinci). Aktifkan untuk membandingkan performa model.",
            )
            f_inc = st.checkbox(
                "Clustering: gunakan G1–G3",
                value=True,
                help="Saat memprediksi siswa baru, G3 otomatis diisi dari hasil prediksi regresi.",
            )
            f_mink = st.slider(
                "Clustering: k minimal", 2, 5, 3,
                help="k = 2 hanya membagi siswa menjadi dua kelompok besar.",
            )
            f_mode = st.radio(
                "Mode pelatihan", ["Cepat", "Lengkap"], index=0,
                help="Cepat cocok untuk demo/server gratis; Lengkap melakukan tuning lebih mendalam beserta learning curve.",
            )
            st.form_submit_button("Terapkan & Latih Ulang", type="primary", use_container_width=True)

    use_prev_grades, include_absences, include_grades, min_k_final, train_mode = (
        f_prev, f_abs, f_inc, f_mink, f_mode,
    )


# =============================================================================
#  BAGIAN G - PELATIHAN MODEL (OTOMATIS & DI-CACHE)
# =============================================================================
with st.spinner(f"Menyiapkan model regresi & clustering (mode {train_mode})..."):
    RES_R = train_regression(df_raw, DATA_KEY, use_prev_grades, include_absences, train_mode)
    RES_C = train_clustering(df_raw, DATA_KEY, include_grades, include_absences, min_k_final, train_mode)

reg_model, reg_schema = RES_R["model"], RES_R["schema"]
clu_model, clu_schema = RES_C["model"], RES_C["schema"]

with st.sidebar:
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    rm = reg_schema["metrics_test"]
    cm = clu_schema["metrics"]
    abs_txt = "Disertakan" if include_absences else "Dikecualikan (default)"
    st.markdown(
        f'<div class="sb-status">'
        f'<div class="sb-status-title">Ringkasan Model Aktif</div>'
        f'<strong>Regresi:</strong> {reg_schema["model_name"]}<br>'
        f'R² Test: <strong>{rm["R2"]:.3f}</strong> · RMSE: <strong>{rm["RMSE"]:.2f}</strong><br>'
        f'<strong>Clustering:</strong> {clu_schema["algorithm"]} (k = {clu_schema["k"]})<br>'
        f'Silhouette: <strong>{cm["silhouette_test"]:.3f}</strong> · ARI: <strong>{cm["stability_ari"]:.2f}</strong><br>'
        f'<strong>Fitur absences:</strong> {abs_txt}'
        f"</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
#  HERO HEADER (BERSIH & TIDAK PADAT)
# =============================================================================
st.markdown(
    f'<div class="hero-box">'
    f'<div class="hero-kicker">{COURSE_INFO}</div>'
    f'<h1 class="hero-title">{APP_TITLE}</h1>'
    f'<div class="hero-pills">'
    f'<span class="pill pill-accent">Dataset: UCI Student Performance (Matematika, n = {len(df_raw)})</span>'
    f'<span class="pill">Model Regresi: {reg_schema["model_name"]}</span>'
    f'<span class="pill">Segmentasi: {clu_schema["algorithm"]} (k = {clu_schema["k"]})</span>'
    f'<span class="pill">Fitur absences: {abs_txt}</span>'
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# =============================================================================
#  HALAMAN 1 — PREDIKSI & SEGMENTASI OTOMATIS
# =============================================================================
if page == "Prediksi & Segmentasi":
    order, numeric_specs, categorical_specs = merged_input_specs(reg_schema, clu_schema)
    default_profile = compute_default_profile(df_raw, order, numeric_specs, categorical_specs)
    presets = build_presets(default_profile, order)

    # Sinkronkan session state jika konfigurasi fitur berubah
    cfg_sig = f"{DATA_KEY}-{use_prev_grades}-{include_absences}-{include_grades}"
    if "active_profile" not in st.session_state or st.session_state.get("cfg_sig") != cfg_sig:
        st.session_state["active_profile"] = default_profile.copy()
        st.session_state["active_preset"] = "Rata-rata Kelas"
        st.session_state["cfg_sig"] = cfg_sig

    # Baris Preset & Mode Input
    section_header(
        "Langkah 1 · Profil Siswa",
        "Masukkan Data Siswa atau Pilih Profil Contoh",
        "Pilih preset untuk simulasi cepat, sesuaikan parameter di bawah, lalu klik tombol prediksi untuk melihat nilai akhir dan klaster otomatis.",
    )

    p_cols = st.columns(len(presets) + 1)
    for idx, (pname, pvals) in enumerate(presets.items()):
        is_sel = st.session_state.get("active_preset") == pname
        if p_cols[idx].button(pname, type="primary" if is_sel else "secondary", use_container_width=True, key=f"preset_{idx}"):
            st.session_state["active_profile"] = pvals.copy()
            st.session_state["active_preset"] = pname
            st.rerun()

    with p_cols[-1]:
        input_mode = st.radio(
            "Tampilan Form",
            ["Ringkas", "Lengkap"],
            horizontal=True,
            label_visibility="collapsed",
            help="Ringkas menampilkan fitur paling berpengaruh; Lengkap menampilkan seluruh fitur yang digunakan model.",
        )

    cur_profile = st.session_state["active_profile"]

    with st.form("student_profile_form"):
        form_vals = cur_profile.copy()
        if input_mode == "Ringkas":
            shown = [c for c in COMPACT_FEATURES if c in order]
            # Pastikan top features model ikut tampil jika belum ada
            for tf in reg_schema["top_features"]:
                if tf in order and tf not in shown and len(shown) < 12:
                    shown.append(tf)
            st.caption(
                f"Menampilkan {len(shown)} fitur utama. {len(order) - len(shown)} fitur lainnya otomatis menggunakan nilai median/modus kelas."
            )
            cols_grid = st.columns(3)
            for i, col in enumerate(shown):
                with cols_grid[i % 3]:
                    form_vals[col] = render_feature_widget(col, numeric_specs, categorical_specs, cur_profile.get(col, default_profile[col]))
        else:
            ftabs = st.tabs(list(GROUPS.keys()))
            placed = set()
            for tab_obj, (gname, gcols) in zip(ftabs, GROUPS.items()):
                present = [c for c in gcols if c in order]
                with tab_obj:
                    if not present:
                        st.caption("Tidak ada fitur aktif pada kelompok ini.")
                    else:
                        cols_grid = st.columns(3)
                        for i, col in enumerate(present):
                            with cols_grid[i % 3]:
                                form_vals[col] = render_feature_widget(col, numeric_specs, categorical_specs, cur_profile.get(col, default_profile[col]))
                            placed.add(col)
            rest = [c for c in order if c not in placed]
            if rest:
                with st.expander("Fitur Tambahan"):
                    cols_grid = st.columns(3)
                    for i, col in enumerate(rest):
                        with cols_grid[i % 3]:
                            form_vals[col] = render_feature_widget(col, numeric_specs, categorical_specs, cur_profile.get(col, default_profile[col]))

        st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([3, 1])
        submitted = btn_col1.form_submit_button("Prediksi Nilai & Klasterisasi Siswa", type="primary", use_container_width=True)
        reset_btn = btn_col2.form_submit_button("Reset ke Rata-rata", use_container_width=True)

    if reset_btn:
        st.session_state["active_profile"] = default_profile.copy()
        st.session_state["active_preset"] = "Rata-rata Kelas"
        st.rerun()
    elif submitted:
        st.session_state["active_profile"] = form_vals
        st.session_state["active_preset"] = "Kustom"
        cur_profile = form_vals

    # =========================================================================
    #  LANGKAH 2 — HASIL PREDIKSI & SEGMENTASI OTOMATIS (SATU HALAMAN)
    # =========================================================================
    X_reg_row = to_frame(cur_profile, reg_schema["feature_order"], reg_schema["numeric"].keys())
    pred_val = float(predict_g3(reg_model, X_reg_row)[0])
    thr = reg_schema["pass_threshold"]
    rmse = reg_schema["metrics_test"]["RMSE"]
    lo_b, hi_b = max(0.0, pred_val - rmse), min(20.0, pred_val + rmse)
    pop_mean = float(df_raw[TARGET].mean())
    delta_mean = pred_val - pop_mean
    is_risk = pred_val < thr

    # Otomatis klasterisasi menggunakan hasil prediksi G3
    clu_inputs = cur_profile.copy()
    if TARGET in clu_schema["feature_order"]:
        clu_inputs[TARGET] = pred_val
    X_clu_row = to_frame(clu_inputs, clu_schema["feature_order"], clu_schema["numeric"].keys())
    cl_id = int(predict_cluster(clu_model, clu_schema, X_clu_row)[0])
    cl_names = clu_schema["cluster_names"]
    cl_label = cl_names.get(cl_id, f"Cluster {cl_id}")
    cl_short = cl_label.split(" — ")[0]
    cl_desc = cl_label.split(" — ")[1] if " — " in cl_label else ""

    st.markdown("<div style='height: 1.1rem'></div>", unsafe_allow_html=True)
    section_header(
        "Langkah 2 · Hasil Analisis Siswa",
        "Prediksi Nilai Akhir (G3) & Segmentasi Profil Otomatis",
        "Hasil prediksi regresi otomatis diteruskan ke model clustering untuk mengelompokkan profil siswa.",
    )

    g_col, k_col = st.columns([1.15, 1.85], gap="large")
    with g_col:
        gauge_color = COLORS["rose"] if is_risk else (COLORS["amber"] if pred_val < 14 else COLORS["emerald"])
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred_val,
                number={"suffix": " / 20", "font": {"size": 34, "color": COLORS["text"]}},
                delta={
                    "reference": pop_mean,
                    "valueformat": "+.1f",
                    "increasing": {"color": COLORS["emerald"]},
                    "decreasing": {"color": COLORS["rose"]},
                    "suffix": " vs rata-rata",
                },
                gauge={
                    "axis": {"range": [0, 20], "tickwidth": 1, "tickcolor": COLORS["muted"], "dtick": 5},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "rgba(148,163,184,0.08)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, thr], "color": "rgba(244, 63, 94, 0.14)"},
                        {"range": [thr, 14], "color": "rgba(245, 158, 11, 0.12)"},
                        {"range": [14, 20], "color": "rgba(16, 185, 129, 0.14)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F8FAFC", "width": 2.5},
                        "thickness": 0.8,
                        "value": thr,
                    },
                },
                title={"text": "Estimasi Nilai Akhir (Batas Lulus = 10)", "font": {"size": 13, "color": COLORS["muted"]}},
            )
        )
        style_plotly(fig_gauge, height=275, margin=dict(l=24, r=24, t=42, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with k_col:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            kpi_card("Prediksi Nilai Akhir (G3)", f"{pred_val:.1f} / 20", f"Selisih {delta_mean:+.1f} poin dari rata-rata ({pop_mean:.1f})")
        with r1c2:
            kpi_card("Rentang Estimasi (±1 RMSE)", f"{lo_b:.1f} – {hi_b:.1f}", f"RMSE test = {rmse:.2f} · R² = {reg_schema['metrics_test']['R2']:.2f}")

        st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            badge = (
                '<span class="badge-risk">Berisiko (&lt; 10)</span>'
                if is_risk
                else '<span class="badge-safe">Aman (Lulus)</span>'
            )
            kpi_card("Status Kelulusan", "", f"Ambang batas kelulusan: {thr} / 20", badge_html=badge)
        with r2c2:
            cl_badge = f'<span class="badge-info">{cl_short}</span>'
            kpi_card("Segmentasi Otomatis", "", cl_desc or f"Model {clu_schema['algorithm']} (k = {clu_schema['k']})", badge_html=cl_badge)

        if is_risk:
            st.markdown(
                f'<div class="callout callout-risk"><strong>Perlu Pendampingan Awal:</strong> '
                f'Prediksi nilai akhir ({pred_val:.1f}) berada di bawah batas kelulusan ({thr}). '
                f'Disarankan evaluasi faktor akademik utama (nilai periode sebelumnya, riwayat gagal kelas, dan jam belajar mingguan).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="callout callout-safe"><strong>Posisi Akademik Aman:</strong> '
                f'Prediksi nilai akhir ({pred_val:.1f}) melampaui batas kelulusan ({thr}). '
                f'Pertahankan konsistensi belajar; hasil ini adalah estimasi statistik, bukan kepastian nilai akhir.</div>',
                unsafe_allow_html=True,
            )

    # Sub-tab Detail Hasil (Segmentasi Otomatis & Simulasi What-If)
    res_tab1, res_tab2 = st.tabs(["Detail Segmentasi Otomatis", "Simulasi What-If"])

    with res_tab1:
        prof_df, df_all = RES_C["prof"], RES_C["df_all"]
        cl_size = int(RES_C["size"].loc[cl_id])
        cl_g3 = float(prof_df.loc[cl_id, "G3"])
        auto_note = (
            f"Nilai G3 untuk klasterisasi otomatis diisi dari prediksi regresi ({pred_val:.1f})."
            if clu_schema["include_grades"]
            else "Klasterisasi dilakukan hanya dari karakteristik non-nilai (tanpa G1–G3)."
        )
        st.markdown(
            f'<div class="callout"><strong>{cl_label}</strong> · Anggota: <strong>{cl_size} siswa</strong> '
            f'({cl_size / len(df_all):.1%} dari dataset) · Rata-rata G3 cluster: <strong>{cl_g3:.1f}</strong>.<br>'
            f'<span style="color:#94A3B8; font-size:0.82rem;">{auto_note} Silhouette test = {clu_schema["metrics"]["silhouette_test"]:.2f} '
            f'(baca segmentasi sebagai kecenderungan profil, bukan batas tegas).</span></div>',
            unsafe_allow_html=True,
        )

        sc1, sc2 = st.columns([1.35, 1.0], gap="large")
        with sc1:
            # Radar / Bar Perbandingan Siswa vs Cluster vs Populasi (dinormalisasi 0–1)
            radar_cols = [
                c for c in ["G1", "G2", "G3", "studytime", "failures", "goout", "Medu", "famrel"]
                if c in prof_df.columns and c in clu_inputs
            ]
            r_labels = [LABELS.get(c, c) for c in radar_cols]
            mins = df_all[radar_cols].min()
            maxs = df_all[radar_cols].max().replace(0, 1)
            span = (maxs - mins).replace(0, 1)

            v_stu = [(float(clu_inputs[c]) - mins[c]) / span[c] for c in radar_cols]
            v_clu = [(float(prof_df.loc[cl_id, c]) - mins[c]) / span[c] for c in radar_cols]
            v_all = [(float(df_all[c].mean()) - mins[c]) / span[c] for c in radar_cols]

            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=v_stu + [v_stu[0]], theta=r_labels + [r_labels[0]], fill="toself",
                name="Siswa Ini", line=dict(color=COLORS["sky"], width=2.5), fillcolor="rgba(56,189,248,0.18)",
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=v_clu + [v_clu[0]], theta=r_labels + [r_labels[0]], fill="toself",
                name=f"Rata-rata {cl_short}", line=dict(color=COLORS["primary_light"], width=2), fillcolor="rgba(99,102,241,0.14)",
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=v_all + [v_all[0]], theta=r_labels + [r_labels[0]],
                name="Rata-rata Semua Siswa", line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
            ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="rgba(17,24,39,0.55)",
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(148,163,184,0.16)"),
                    angularaxis=dict(gridcolor="rgba(148,163,184,0.16)", tickfont=dict(size=11, color=COLORS["text"])),
                )
            )
            style_plotly(fig_rad, height=360, title="Perbandingan Profil Ternormalisasi (Skala 0–1)")
            st.plotly_chart(fig_rad, use_container_width=True)

        with sc2:
            if hasattr(clu_model, "predict_proba"):
                proba = clu_model.predict_proba(X_clu_row)[0]
                lm = clu_schema["label_map"]
                pr_items = sorted([(lm[i], float(p)) for i, p in enumerate(proba)], key=lambda x: x[0])
                fig_pr = go.Figure(go.Bar(
                    x=[f"Cluster {k}" for k, _ in pr_items],
                    y=[v * 100 for _, v in pr_items],
                    marker_color=[COLORS["primary"] if k == cl_id else "rgba(148,163,184,0.35)" for k, _ in pr_items],
                    text=[f"{v:.1%}" for _, v in pr_items],
                    textposition="outside",
                ))
                style_plotly(fig_pr, height=220, title="Probabilitas Keanggotaan Cluster (%)")
                fig_pr.update_yaxes(range=[0, 112])
                st.plotly_chart(fig_pr, use_container_width=True)

            key_cols = [
                c for c in ["G1", "G2", "G3", "failures", "studytime", "absences", "goout", "Medu", "Fedu", "age"]
                if c in prof_df.columns and c in clu_inputs
            ]
            cmp_df = pd.DataFrame(
                {
                    "Siswa Ini": [clu_inputs[c] for c in key_cols],
                    f"Rata-rata {cl_short}": [prof_df.loc[cl_id, c] for c in key_cols],
                    "Rata-rata Kelas": [df_all[c].mean() for c in key_cols],
                },
                index=[LABELS.get(c, c) for c in key_cols],
            ).astype(float).round(2)
            st.dataframe(cmp_df, use_container_width=True)

        with st.expander("Lihat Profil Rata-rata Seluruh Cluster"):
            prof_show = prof_df.copy()
            prof_show.index = [cl_names.get(i, f"Cluster {i}") for i in prof_show.index]
            prof_show.columns = [LABELS.get(c, c) for c in prof_show.columns]
            st.dataframe(prof_show.T.round(2), use_container_width=True)

    with res_tab2:
        st.caption("Ubah satu fitur sambil menahan fitur lain pada profil saat ini untuk mengamati sensitivitas prediksi G3.")
        num_feats = [c for c in reg_schema["feature_order"] if c in reg_schema["numeric"]]
        top_pref = [c for c in reg_schema["top_features"] if c in num_feats]
        options = top_pref + [c for c in num_feats if c not in top_pref]
        feat = st.selectbox("Pilih fitur yang disimulasikan", options, format_func=lambda c: LABELS.get(c, c))
        spec = reg_schema["numeric"][feat]
        if is_whole(spec["min"]) and is_whole(spec["max"]):
            grid = np.arange(int(spec["min"]), int(spec["max"]) + 1, dtype=float)
        else:
            grid = np.linspace(spec["min"], spec["max"], 30)

        X_var = X_reg_row.loc[X_reg_row.index.repeat(len(grid))].reset_index(drop=True)
        X_var[feat] = grid
        preds_sim = predict_g3(reg_model, X_var)
        cur_fval = float(cur_profile[feat])

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=grid, y=preds_sim, mode="lines+markers",
            name="Prediksi G3", line=dict(color=COLORS["sky"], width=3),
            marker=dict(size=6, color=COLORS["primary_light"]),
            hovertemplate=f"{LABELS.get(feat, feat)}: %{{x:g}}<br>Prediksi G3: %{{y:.2f}}<extra></extra>",
        ))
        fig_sim.add_hline(y=thr, line_dash="dash", line_color=COLORS["rose"], annotation_text="Batas Lulus (10)", annotation_position="bottom right")
        fig_sim.add_vline(x=cur_fval, line_dash="dot", line_color=COLORS["emerald"], annotation_text=f"Saat ini ({cur_fval:g})", annotation_position="top right")
        style_plotly(fig_sim, height=340, title=f"Sensitivitas Prediksi G3 terhadap {LABELS.get(feat, feat)}")
        fig_sim.update_xaxes(title_text=LABELS.get(feat, feat))
        fig_sim.update_yaxes(title_text="Prediksi G3 (0–20)", range=[0, 20.5])
        st.plotly_chart(fig_sim, use_container_width=True)

        best_val = grid[int(preds_sim.argmax())]
        st.caption(
            f"Dengan fitur lain tetap, prediksi tertinggi ({preds_sim.max():.1f}) dicapai saat "
            f"{LABELS.get(feat, feat)} = {best_val:g}. Grafik ini mencerminkan asosiasi dalam model, bukan hubungan sebab-akibat."
        )


# =============================================================================
#  HALAMAN 2 — PREDIKSI MASSAL
# =============================================================================
elif page == "Prediksi Massal":
    section_header(
        "Pemrosesan Batch",
        "Prediksi & Segmentasi Massal dari File CSV / Excel",
        "Unggah data banyak siswa sekaligus untuk mendapatkan estimasi nilai akhir, status risiko, dan cluster secara otomatis.",
    )

    order, numeric_specs, categorical_specs = merged_input_specs(reg_schema, clu_schema)
    default_profile = compute_default_profile(df_raw, order, numeric_specs, categorical_specs)
    template = pd.DataFrame([default_profile])[order]

    bc1, bc2 = st.columns([1, 2], gap="large")
    with bc1:
        st.markdown(
            '<div class="callout"><strong>Format Template:</strong><br>'
            f'Template mencakup <strong>{len(order)} kolom fitur</strong> aktif tanpa kolom target G3. '
            'Jika clustering menggunakan G1–G3, sistem otomatis mengisi G3 dari hasil prediksi regresi.</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Unduh Template CSV (1 Baris Contoh)",
            template.to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_siswa.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with bc2:
        up = st.file_uploader("Unggah file data siswa (.csv atau .xlsx)", type=["csv", "xlsx"], key="upload_batch")

    if up is not None:
        try:
            df_up = pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"File tidak dapat dibaca: {e}")
            df_up = None

        if df_up is not None and len(df_up) > 0:
            result = df_up.copy()
            miss_r = [c for c in reg_schema["feature_order"] if c not in df_up.columns]
            if miss_r:
                st.warning(f"Prediksi G3 dilewati karena kolom berikut hilang: {', '.join(miss_r)}")
            else:
                result["Prediksi_G3"] = predict_g3(reg_model, clean_batch(df_up, reg_schema)).round(2)
                result["Status_Risiko"] = np.where(result["Prediksi_G3"] < reg_schema["pass_threshold"], "Berisiko", "Aman")

            # Siapkan input clustering (isi G3 dari prediksi jika tidak tersedia di file unggahan)
            df_clu_in = df_up.copy()
            if TARGET in clu_schema["feature_order"] and TARGET not in df_clu_in.columns and "Prediksi_G3" in result.columns:
                df_clu_in[TARGET] = result["Prediksi_G3"]

            miss_c = [c for c in clu_schema["feature_order"] if c not in df_clu_in.columns]
            if miss_c:
                st.warning(f"Segmentasi dilewati karena kolom berikut hilang: {', '.join(miss_c)}")
            else:
                labs = predict_cluster(clu_model, clu_schema, clean_batch(df_clu_in, clu_schema))
                result["Cluster"] = labs
                result["Nama_Cluster"] = [clu_schema["cluster_names"].get(int(i), f"Cluster {i}") for i in labs]

            # Ringkasan KPI Batch
            st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                kpi_card("Total Siswa", f"{len(result)}", f"{df_up.shape[1]} kolom terbaca")
            if "Prediksi_G3" in result:
                n_risk = int((result["Status_Risiko"] == "Berisiko").sum())
                n_safe = len(result) - n_risk
                with m2:
                    kpi_card("Rata-rata Prediksi G3", f"{result['Prediksi_G3'].mean():.2f}", f"Rentang {result['Prediksi_G3'].min():.1f} – {result['Prediksi_G3'].max():.1f}")
                with m3:
                    kpi_card("Siswa Berisiko", f"{n_risk}", f"{n_risk / len(result):.1%} di bawah batas lulus")
                with m4:
                    kpi_card("Siswa Aman", f"{n_safe}", f"{n_safe / len(result):.1%} di atas batas lulus")

                ch1, ch2 = st.columns(2, gap="large")
                with ch1:
                    fig_bh = go.Figure(go.Histogram(x=result["Prediksi_G3"], nbinsx=20, marker_color=COLORS["primary"]))
                    fig_bh.add_vline(x=PASS_THRESHOLD, line_dash="dash", line_color=COLORS["rose"], annotation_text="Batas Lulus (10)")
                    style_plotly(fig_bh, height=300, title="Distribusi Prediksi G3")
                    fig_bh.update_xaxes(title_text="Prediksi G3")
                    fig_bh.update_yaxes(title_text="Jumlah Siswa")
                    st.plotly_chart(fig_bh, use_container_width=True)
                with ch2:
                    fig_pie = go.Figure(go.Pie(
                        labels=["Aman", "Berisiko"],
                        values=[n_safe, n_risk],
                        hole=0.55,
                        marker=dict(colors=[COLORS["emerald"], COLORS["rose"]]),
                    ))
                    style_plotly(fig_pie, height=300, title="Proporsi Status Kelulusan")
                    st.plotly_chart(fig_pie, use_container_width=True)

            # Filter Tabel Hasil
            f_col1, f_col2 = st.columns(2)
            view_df = result.copy()
            if "Status_Risiko" in view_df:
                st_filter = f_col1.selectbox("Filter Status Risiko", ["Semua", "Berisiko", "Aman"])
                if st_filter != "Semua":
                    view_df = view_df[view_df["Status_Risiko"] == st_filter]
            if "Nama_Cluster" in view_df:
                cl_opts = ["Semua"] + sorted(view_df["Nama_Cluster"].unique().tolist())
                cl_filter = f_col2.selectbox("Filter Cluster", cl_opts)
                if cl_filter != "Semua":
                    view_df = view_df[view_df["Nama_Cluster"] == cl_filter]

            st.dataframe(view_df, use_container_width=True)
            st.download_button(
                "Unduh Hasil Prediksi & Segmentasi (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_siswa.csv",
                mime="text/csv",
                type="primary",
            )


# =============================================================================
#  HALAMAN 3 — EVALUASI REGRESI
# =============================================================================
elif page == "Evaluasi Regresi":
    R = RES_R
    rep = R["report"]
    r2_t, mae_t, rmse_t = rep.loc["Test", "R2"], rep.loc["Test", "MAE"], rep.loc["Test", "RMSE"]
    acc2_t = rep.loc["Test", "Akurasi ±2 poin"]

    section_header(
        "Evaluasi Model Supervised",
        "Performa & Diagnostik Model Regresi Nilai Akhir (G3)",
        f"Split data: {R['n_train']} latih / {R['n_test']} uji (stratified kuantil G3) · {R['n_features']} fitur aktif.",
    )

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        kpi_card("Model Terbaik", R["schema"]["model_name"], "Dipilih via CV & tuning")
    with e2:
        kpi_card("Test R²", f"{r2_t:.3f}", f"Train R²: {rep.loc['Train', 'R2']:.3f}")
    with e3:
        kpi_card("Test RMSE", f"{rmse_t:.2f}", f"Baseline: {rep.loc['Baseline (Test)', 'RMSE']:.2f}")
    with e4:
        kpi_card("Test MAE", f"{mae_t:.2f}", f"MedAE: {rep.loc['Test', 'MedAE']:.2f}")
    with e5:
        kpi_card("Akurasi ±2 Poin", f"{acc2_t:.0%}", f"±1 poin: {rep.loc['Test', 'Akurasi ±1 poin']:.0%}")

    top3 = [LABELS.get(f, f) for f in R["importance"]["Fitur"].head(3).tolist()]
    overfit_txt = (
        "indikasi overfitting — pertimbangkan regularisasi lebih kuat"
        if R["gap_r2"] > 0.15
        else "generalisasi stabil (tidak overfitting berat)"
    )
    zero_note = (
        f"MAE pada siswa dengan G3 = 0: <strong>{R['mae_zero']:.2f}</strong> poin vs siswa lain: <strong>{R['mae_other']:.2f}</strong> poin. "
        if R["mae_zero"] is not None and R["mae_other"] is not None
        else ""
    )
    st.markdown(
        f'<div class="callout">'
        f'<strong>Ringkasan Interpretasi:</strong> Model <strong>{R["schema"]["model_name"]}</strong> '
        f'menjelaskan sekitar <strong>{r2_t:.1%}</strong> variasi nilai akhir pada data uji '
        f'(selisih R² train–test = {R["gap_r2"]:.3f}, {overfit_txt}). '
        f'Fitur paling berpengaruh: <strong>{", ".join(top3)}</strong>. {zero_note}'
        f'</div>',
        unsafe_allow_html=True,
    )

    rtab1, rtab2, rtab3 = st.tabs(["Perbandingan Algoritma & Tuning", "Diagnostik & Deteksi Risiko", "Interpretasi Fitur"])

    with rtab1:
        st.markdown("**Ringkasan Metrik Train, Test, dan Baseline**")
        st.dataframe(rep.round(3), use_container_width=True)

        bench_sorted = R["bench"].sort_values("CV RMSE", ascending=False)
        fig_bench = go.Figure(go.Bar(
            y=bench_sorted["Model"],
            x=bench_sorted["CV RMSE"],
            orientation="h",
            marker_color=[COLORS["primary"] if m == R["schema"]["model_name"] else "rgba(56,189,248,0.55)" for m in bench_sorted["Model"]],
            text=[f"{v:.2f}" for v in bench_sorted["CV RMSE"]],
            textposition="outside",
        ))
        style_plotly(fig_bench, height=400, title="Perbandingan Algoritma pada Cross-Validation (CV RMSE — Lebih Kecil Lebih Baik)")
        fig_bench.update_xaxes(title_text="CV RMSE")
        st.plotly_chart(fig_bench, use_container_width=True)

        c_b1, c_b2 = st.columns(2, gap="large")
        with c_b1:
            st.markdown("**Tabel Benchmark Cross-Validation**")
            st.dataframe(R["bench"].round(3), use_container_width=True)
        with c_b2:
            st.markdown(f"**Hasil Hyperparameter Tuning ({', '.join(R['tunable'])})**")
            st.dataframe(R["tune"].round(3), use_container_width=True)

    with rtab2:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(
                x=R["y_test"], y=R["pred_test"], mode="markers",
                marker=dict(size=8, color=np.abs(R["resid"]), colorscale="Plasma", showscale=True,
                            colorbar=dict(title="|Error|", thickness=12)),
                hovertemplate="Aktual: %{x:.1f}<br>Prediksi: %{y:.2f}<extra></extra>",
                name="Data Uji",
            ))
            fig_ap.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode="lines", line=dict(dash="dash", color=COLORS["muted"]), name="Garis Ideal"))
            style_plotly(fig_ap, height=360, title=f"Aktual vs Prediksi ({R['schema']['model_name']})")
            fig_ap.update_xaxes(title_text="Nilai Aktual (G3)", range=[-0.5, 20.5])
            fig_ap.update_yaxes(title_text="Prediksi G3", range=[-0.5, 20.5])
            st.plotly_chart(fig_ap, use_container_width=True)

        with d2:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=R["pred_test"], y=R["resid"], mode="markers",
                marker=dict(size=8, color=COLORS["sky"], opacity=0.8),
                hovertemplate="Prediksi: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
                name="Residual",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color=COLORS["rose"])
            style_plotly(fig_res, height=360, title="Plot Residual (Pola Acak Menunjukkan Fit yang Baik)")
            fig_res.update_xaxes(title_text="Prediksi G3")
            fig_res.update_yaxes(title_text="Residual (Aktual − Prediksi)")
            st.plotly_chart(fig_res, use_container_width=True)

        d3, d4 = st.columns(2, gap="large")
        with d3:
            fig_rhist = go.Figure(go.Histogram(x=R["resid"], nbinsx=20, marker_color=COLORS["emerald"], opacity=0.85))
            fig_rhist.add_vline(x=0, line_dash="dash", line_color=COLORS["text"])
            style_plotly(fig_rhist, height=300, title="Distribusi Residual")
            fig_rhist.update_xaxes(title_text="Residual")
            fig_rhist.update_yaxes(title_text="Frekuensi")
            st.plotly_chart(fig_rhist, use_container_width=True)

        with d4:
            if R["lc_data"] is not None:
                lc = R["lc_data"]
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["train_rmse"], mode="lines+markers", name="Train RMSE", line=dict(color=COLORS["sky"])))
                fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["val_rmse"], mode="lines+markers", name="Validasi RMSE", line=dict(color=COLORS["amber"])))
                style_plotly(fig_lc, height=300, title="Learning Curve")
                fig_lc.update_xaxes(title_text="Jumlah Data Latih")
                fig_lc.update_yaxes(title_text="RMSE")
                st.plotly_chart(fig_lc, use_container_width=True)
            else:
                rk = R["risk"]
                st.markdown(f"**Deteksi Siswa Berisiko (G3 < {PASS_THRESHOLD})**")
                rc1, rc2 = st.columns(2)
                with rc1:
                    kpi_card("Terdeteksi Benar (TP)", str(rk["tp"]), f"Recall berisiko: {rk['recall']:.0%}")
                    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
                    kpi_card("Salah Alarm (FP)", str(rk["fp"]), f"Precision berisiko: {rk['precision']:.0%}")
                with rc2:
                    kpi_card("Terlewat (FN)", str(rk["fn"]), "Siswa berisiko diprediksi aman")
                    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
                    kpi_card("Benar Lulus (TN)", str(rk["tn"]), "Siswa aman diprediksi aman")

        if R["lc_data"] is not None:
            rk = R["risk"]
            st.markdown(f"**Deteksi Siswa Berisiko (G3 < {PASS_THRESHOLD})**")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Terdeteksi Benar (TP)", str(rk["tp"]), f"Recall: {rk['recall']:.0%}")
            with k2:
                kpi_card("Terlewat (FN)", str(rk["fn"]), "Berisiko diprediksi aman")
            with k3:
                kpi_card("Salah Alarm (FP)", str(rk["fp"]), f"Precision: {rk['precision']:.0%}")
            with k4:
                kpi_card("Benar Lulus (TN)", str(rk["tn"]), "Aman diprediksi aman")

    with rtab3:
        top_imp = R["importance"].head(12).iloc[::-1]
        ic1, ic2 = st.columns([1.35, 1.0], gap="large")
        with ic1:
            fig_imp = go.Figure(go.Bar(
                y=top_imp["Label"],
                x=top_imp["Kenaikan RMSE"],
                error_x=dict(type="data", array=top_imp["Std"], color=COLORS["muted"]),
                orientation="h",
                marker_color=COLORS["primary_light"],
            ))
            style_plotly(fig_imp, height=400, title="Permutation Importance pada Data Uji (Kenaikan RMSE)")
            fig_imp.update_xaxes(title_text="Kenaikan RMSE saat fitur diacak")
            st.plotly_chart(fig_imp, use_container_width=True)
        with ic2:
            st.markdown("**10 Fitur Paling Berpengaruh**")
            st.dataframe(R["importance"][["Label", "Kenaikan RMSE", "Std"]].head(10).round(4), use_container_width=True)
            if R["coef"] is not None:
                st.markdown("**Koefisien Terbesar (Fitur Terstandarisasi)**")
                st.dataframe(R["coef"].round(3).rename("Koefisien").to_frame(), use_container_width=True)

        if R["pdp_data"]:
            fig_pdp = make_subplots(rows=1, cols=len(R["pdp_data"]), subplot_titles=[d["label"] for d in R["pdp_data"]])
            for i, d in enumerate(R["pdp_data"], start=1):
                fig_pdp.add_trace(
                    go.Scatter(x=d["grid"], y=d["avg"], mode="lines+markers", line=dict(color=COLORS["sky"], width=2.5), showlegend=False),
                    row=1, col=i,
                )
            style_plotly(fig_pdp, height=320, title="Partial Dependence: Pengaruh Rata-rata Fitur Utama terhadap Prediksi G3")
            st.plotly_chart(fig_pdp, use_container_width=True)


# =============================================================================
#  HALAMAN 4 — EVALUASI CLUSTERING
# =============================================================================
elif page == "Evaluasi Clustering":
    C = RES_C
    ev = C["ev"]
    sil_t, ari_m = ev.loc["Test", "silhouette"], float(C["ari"].mean())
    sil_txt = "kuat" if sil_t > 0.5 else ("sedang" if sil_t >= 0.25 else "lemah (tumpang-tindih)")
    stab_txt = "sangat stabil" if ari_m > 0.8 else ("cukup stabil" if ari_m >= 0.6 else "kurang stabil")

    section_header(
        "Evaluasi Model Unsupervised",
        "Kualitas, Stabilitas & Karakteristik Segmentasi Siswa",
        f"Split data: {C['n_train']} latih / {C['n_test']} uji · PCA mempertahankan {C['n_pc']} komponen ({C['var_pc']:.0%} varians).",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Algoritma Terpilih", clu_schema["algorithm"], f"k = {clu_schema['k']} cluster")
    with c2:
        kpi_card("Silhouette (Test)", f"{sil_t:.3f}", f"Label acak: {C['rand_sil']:.3f}")
    with c3:
        kpi_card("Stabilitas ARI", f"{ari_m:.2f}", f"Kategori: {stab_txt}")
    with c4:
        kpi_card("Davies-Bouldin", f"{ev.loc['Test', 'davies']:.2f}", "Lebih kecil lebih baik")
    with c5:
        kpi_card("Kruskal-Wallis (G3)", f"p = {C['kruskal_p']:.1e}", "Signifikan" if C["kruskal_p"] < 0.05 else "Tidak signifikan")

    st.markdown(
        f'<div class="callout">'
        f'<strong>Ringkasan Interpretasi:</strong> Struktur cluster tergolong <strong>{sil_txt}</strong> '
        f'(Silhouette test = {sil_t:.3f}) dan <strong>{stab_txt}</strong> terhadap resampling bootstrap (ARI = {ari_m:.2f}). '
        f'Uji Kruskal-Wallis menunjukkan perbedaan G3 antar cluster {"signifikan secara statistik" if C["kruskal_p"] < 0.05 else "tidak signifikan"} '
        f'(H = {C["kruskal_H"]:.1f}, p = {C["kruskal_p"]:.1e}). Pada data survei campuran, cluster dibaca sebagai segmentasi deskriptif.'
        f"</div>",
        unsafe_allow_html=True,
    )

    ctab1, ctab2, ctab3 = st.tabs(["Pemilihan Algoritma & k", "Sebaran PCA & Silhouette", "Profil & Heatmap Z-Score"])

    with ctab1:
        # 2x2 grid responsif (menggantikan 1x4 terlalu lebar)
        fig_sel = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Silhouette (Lebih Besar Lebih Baik)",
                "Calinski-Harabasz (Lebih Besar Lebih Baik)",
                "Davies-Bouldin (Lebih Kecil Lebih Baik)",
                "Elbow KMeans (Inertia)",
            ],
            vertical_spacing=0.18, horizontal_spacing=0.10,
        )
        algo_colors = {"KMeans": COLORS["sky"], "GMM": COLORS["primary_light"], "Agglomerative": COLORS["emerald"]}
        for name, g in C["sel"].groupby("Algoritma"):
            col_a = algo_colors.get(name, COLORS["amber"])
            fig_sel.add_trace(go.Scatter(x=g["k"], y=g["silhouette"], mode="lines+markers", name=name, line=dict(color=col_a), legendgroup=name), row=1, col=1)
            fig_sel.add_trace(go.Scatter(x=g["k"], y=g["calinski"], mode="lines+markers", name=name, line=dict(color=col_a), legendgroup=name, showlegend=False), row=1, col=2)
            fig_sel.add_trace(go.Scatter(x=g["k"], y=g["davies"], mode="lines+markers", name=name, line=dict(color=col_a), legendgroup=name, showlegend=False), row=2, col=1)
        km_df = C["sel"][C["sel"].Algoritma == "KMeans"]
        fig_sel.add_trace(go.Scatter(x=km_df["k"], y=km_df["inertia"], mode="lines+markers", name="Inertia", line=dict(color=COLORS["rose"]), showlegend=False), row=2, col=2)
        style_plotly(fig_sel, height=560)
        st.plotly_chart(fig_sel, use_container_width=True)

        t1, t2 = st.columns(2, gap="large")
        with t1:
            st.markdown("**Metrik Evaluasi Train vs Test**")
            st.dataframe(ev.round(3), use_container_width=True)
        with t2:
            st.markdown("**10 Kombinasi Algoritma & k Terbaik**")
            st.dataframe(C["sel"].sort_values("rank_score").head(10)[["Algoritma", "k", "silhouette", "calinski", "davies"]].round(3), use_container_width=True)

        with st.expander("DBSCAN (Pembanding — tidak memiliki .predict() untuk data baru)"):
            st.dataframe(C["dbscan"].round(3), use_container_width=True)

    with ctab2:
        p1, p2 = st.columns([1.3, 1.0], gap="large")
        with p1:
            fig_pca = go.Figure()
            for cl in sorted(C["df_all"]["cluster"].unique()):
                mask = C["df_all"]["cluster"].values == cl
                sub = C["df_all"][mask]
                fig_pca.add_trace(go.Scatter(
                    x=C["P2"][mask, 0], y=C["P2"][mask, 1], mode="markers",
                    name=f"Cluster {cl}",
                    marker=dict(size=7, color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], opacity=0.85, line=dict(width=0.5, color="#0B0F19")),
                    customdata=np.stack([sub["G3"], sub["failures"], sub["studytime"]], axis=-1),
                    hovertemplate=f"Cluster {cl}<br>G3: %{{customdata[0]}}<br>Failures: %{{customdata[1]}}<br>Studytime: %{{customdata[2]}}<extra></extra>",
                ))
            style_plotly(fig_pca, height=430, title=f"Sebaran Cluster pada PCA 2D ({clu_schema['algorithm']}, k = {clu_schema['k']})")
            fig_pca.update_xaxes(title_text=f"PC1 ({C['pca2_var'][0]:.1%})")
            fig_pca.update_yaxes(title_text=f"PC2 ({C['pca2_var'][1]:.1%})")
            st.plotly_chart(fig_pca, use_container_width=True)

        with p2:
            sil_df = pd.DataFrame({"Cluster": [f"Cluster {c}" for c in C["df_all"]["cluster"]], "Silhouette": C["sample_sil"]})
            fig_sil = go.Figure()
            for cl in sorted(C["df_all"]["cluster"].unique()):
                vals = sil_df[sil_df["Cluster"] == f"Cluster {cl}"]["Silhouette"]
                fig_sil.add_trace(go.Box(
                    y=vals, name=f"Cluster {cl}",
                    marker_color=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)], boxmean=True,
                ))
            fig_sil.add_hline(y=float(C["sample_sil"].mean()), line_dash="dash", line_color=COLORS["rose"], annotation_text="Rata-rata")
            style_plotly(fig_sil, height=430, title="Distribusi Nilai Silhouette per Cluster")
            fig_sil.update_yaxes(title_text="Nilai Silhouette")
            st.plotly_chart(fig_sil, use_container_width=True)

    with ctab3:
        z_t = C["z_prof"].T
        z_labels = [LABELS.get(c, c) for c in z_t.index]
        fig_hm = go.Figure(go.Heatmap(
            z=z_t.values,
            x=[f"Cluster {c}" for c in z_t.columns],
            y=z_labels,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(z_t.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Z-score", thickness=14),
        ))
        style_plotly(fig_hm, height=560, title="Deviasi Rata-rata Fitur per Cluster (Z-Score: Merah = Di Atas Rata-rata, Biru = Di Bawah)")
        st.plotly_chart(fig_hm, use_container_width=True)

        if len(C["notable"]):
            st.markdown("**Kategori Menonjol per Cluster** (Lift >= 1.3 dan Proporsi >= 20%)")
            st.dataframe(C["notable"].round(2), use_container_width=True)


# =============================================================================
#  HALAMAN 5 — DATA, UNDUHAN & TENTANG PROYEK
# =============================================================================
elif page == "Data, Unduhan & Tentang":
    section_header(
        "Dokumentasi & Sumber Daya",
        "Eksplorasi Data, Unduh Model & Informasi Tim Penyusun",
        "Ringkasan dataset UCI Student Performance (Matematika), artefak model siap pakai, serta identitas proyek Kelompok 2.",
    )

    dtab1, dtab2, dtab3 = st.tabs(["Eksplorasi Data (EDA)", "Unduh Artefak Model", "Tentang Proyek & Tim"])

    with dtab1:
        n_zero = int((df_raw[TARGET] == 0).sum())
        da1, da2, da3, da4 = st.columns(4)
        with da1:
            kpi_card("Jumlah Observasi", f"{df_raw.shape[0]}", f"Sumber: {raw_name}")
        with da2:
            kpi_card("Jumlah Variabel", f"{df_raw.shape[1]}", "30 fitur + G1, G2, G3")
        with da3:
            kpi_card("Missing Value", f"{int(df_raw.isna().sum().sum())}", "Data bersih tanpa duplikat")
        with da4:
            kpi_card("Siswa G3 = 0", f"{n_zero}", f"{n_zero / len(df_raw):.1%} (dipertahankan, valid)")

        st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)
        ec1, ec2 = st.columns(2, gap="large")
        with ec1:
            fig_g3 = go.Figure(go.Histogram(x=df_raw[TARGET], nbinsx=21, marker_color=COLORS["primary"]))
            fig_g3.add_vline(x=PASS_THRESHOLD, line_dash="dash", line_color=COLORS["rose"], annotation_text="Batas Lulus (10)")
            style_plotly(fig_g3, height=340, title="Distribusi Nilai Akhir (G3)")
            fig_g3.update_xaxes(title_text="Nilai Akhir (G3)")
            fig_g3.update_yaxes(title_text="Jumlah Siswa")
            st.plotly_chart(fig_g3, use_container_width=True)
        with ec2:
            corr = df_raw.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET).sort_values()
            fig_cor = go.Figure(go.Bar(
                y=[LABELS.get(c, c) for c in corr.index],
                x=corr.values,
                orientation="h",
                marker_color=[COLORS["emerald"] if v > 0 else COLORS["rose"] for v in corr.values],
            ))
            style_plotly(fig_cor, height=340, title="Korelasi Fitur Numerik terhadap G3")
            fig_cor.update_xaxes(title_text="Koefisien Korelasi Pearson")
            st.plotly_chart(fig_cor, use_container_width=True)

        fig_box = go.Figure()
        for fval in sorted(df_raw["failures"].unique()):
            fig_box.add_trace(go.Box(
                y=df_raw[df_raw["failures"] == fval][TARGET],
                name=ORDINAL_LABELS["failures"].get(int(fval), str(fval)),
                marker_color=CLUSTER_PALETTE[int(fval) % len(CLUSTER_PALETTE)],
            ))
        style_plotly(fig_box, height=330, title="Distribusi Nilai Akhir (G3) Berdasarkan Riwayat Kegagalan Kelas")
        fig_box.update_yaxes(title_text="Nilai Akhir (G3)")
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Pratinjau Dataset (50 Baris Pertama)**")
        st.dataframe(df_raw.head(50), use_container_width=True)

    with dtab2:
        st.markdown(
            '<div class="callout">File <code>.joblib</code> menyimpan pipeline preprocessing dan model terlatih. '
            "Pastikan versi <code>scikit-learn</code> pada lingkungan tujuan sama dengan lingkungan pelatihan.</div>",
            unsafe_allow_html=True,
        )
        dl1, dl2 = st.columns(2, gap="large")
        with dl1:
            st.markdown("**Artefak Model Regresi**")
            st.download_button("Unduh Model Regresi (.joblib)", model_bytes(reg_model), file_name="model_regresi.joblib", use_container_width=True)
            st.download_button("Unduh Skema Regresi (.json)", json.dumps(reg_schema, indent=2).encode("utf-8"), file_name="schema_regresi.json", mime="application/json", use_container_width=True)
        with dl2:
            st.markdown("**Artefak Model Clustering & Dataset**")
            st.download_button("Unduh Model Clustering (.joblib)", model_bytes(clu_model), file_name="model_cluster.joblib", use_container_width=True)
            st.download_button("Unduh Skema Clustering (.json)", json.dumps(clu_schema, indent=2).encode("utf-8"), file_name="schema_cluster.json", mime="application/json", use_container_width=True)
            st.download_button("Unduh Data dengan Label Cluster (.csv)", RES_C["df_all"].to_csv(index=False).encode("utf-8"), file_name="data_dengan_cluster.csv", mime="text/csv", use_container_width=True)

    with dtab3:
        st.markdown(
            f'<div class="callout">'
            f'<strong>Judul Proyek:</strong> {APP_TITLE}<br>'
            f'<strong>Mata Kuliah:</strong> Data Sains · <strong>Kelompok 2</strong><br>'
            f'<strong>Sumber Data:</strong> Cortez, P. &amp; Silva, A. (2008), <em>Student Performance Dataset</em> (Matematika, 395 observasi, 33 variabel), UCI Machine Learning Repository.<br>'
            f'<strong>Keputusan Metodologi Utama:</strong> Nilai G3 = 0 (38 siswa) dan outlier valid dipertahankan; fitur <code>absences</code> dikecualikan secara default karena cakupan waktu pencatatannya tidak terinci (tersedia toggle di Pengaturan Model untuk perbandingan); seluruh encoding dan scaling di-fit hanya pada data latih.'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### Tim Penyusun — Kelompok 2")
        cards_html = "".join(
            f'<div class="member-card">'
            f'<div class="member-idx">{i}</div>'
            f'<div><div class="member-name">{name}</div><div class="member-nim">{nim}</div></div>'
            f"</div>"
            for i, (name, nim) in enumerate(TEAM_MEMBERS, start=1)
        )
        st.markdown(f'<div class="team-grid">{cards_html}</div>', unsafe_allow_html=True)


# =============================================================================
#  FOOTER MINIMALIS
# =============================================================================
st.markdown(
    f'<div class="app-footer">'
    f'<span>{COURSE_INFO} · {APP_TITLE}</span>'
    f'<span>Dataset: UCI Student Performance (Matematika)</span>'
    f"</div>",
    unsafe_allow_html=True,
)
