# =============================================================================
#  APLIKASI STREAMLIT TERPADU - PREDIKSI NILAI AKHIR MATEMATIKA SISWA
#  Executive Dark Analytics Edition (Machine Learning: Regresi & Clustering)
#
#  Satu file berisi: pemuatan data -> pelatihan model regresi & clustering
#  (train-test split, pemilihan algoritma, tuning, evaluasi, interpretasi) ->
#  dashboard interaktif Plotly.
#
#  Struktur folder untuk deploy (Streamlit Community Cloud / lokal):
#      app_gabungan.py        <- file ini
#      student-mat.xlsx       <- dataset (jika tidak ada, aplikasi meminta upload)
#      requirements.txt       <- daftar dependensi (termasuk plotly)
#      .streamlit/config.toml <- (opsional) konfigurasi tema gelap Streamlit
#
#  Jalankan lokal:  streamlit run app_gabungan.py
# =============================================================================
import html
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
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

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN & TEMA EXECUTIVE DARK ANALYTICS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Nilai Akhir Matematika | ML Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
TARGET = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PASS_THRESHOLD = 10

# Palet Executive Dark Analytics
COLORS = {
    "bg": "#0B1120",
    "surface": "#111827",
    "surface_alt": "#162033",
    "border": "rgba(148, 163, 184, 0.16)",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "cyan": "#38BDF8",
    "indigo": "#818CF8",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "violet": "#A78BFA",
}
CLUSTER_PALETTE = ["#F43F5E", "#F59E0B", "#38BDF8", "#10B981", "#A78BFA", "#FB7185", "#2DD4BF", "#60A5FA"]

MODE_CFG = {
    "Cepat": dict(
        cv_repeats=1, n_iter=8, n_tune=2, perm_repeats=10, boot=10, learning_curve=False,
        trees=150, km_init=10, gmm_init=2, final_init=20,
    ),
    "Lengkap": dict(
        cv_repeats=2, n_iter=25, n_tune=3, perm_repeats=30, boot=30, learning_curve=True,
        trees=300, km_init=20, gmm_init=5, final_init=50,
    ),
}


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Canvas gelap dengan ambient glow halus */
        .stApp {
            background:
                radial-gradient(900px circle at 15% -5%, rgba(56, 189, 248, 0.10), transparent 45%),
                radial-gradient(700px circle at 90% 0%, rgba(99, 102, 241, 0.10), transparent 40%),
                #0B1120;
            color: #F8FAFC;
        }

        /* Sidebar polished */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #0B1120 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        /* Main padding */
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(22, 34, 56, 0.92) 60%, rgba(30, 41, 59, 0.88) 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 16px;
            padding: 1.35rem 1.65rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 12px 32px rgba(2, 6, 23, 0.45);
        }
        .hero-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #38BDF8;
            margin-bottom: 0.3rem;
        }
        .hero-title {
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 0.92rem;
            color: #94A3B8;
            margin: 0 0 0.85rem 0;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.26rem 0.68rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #E2E8F0;
        }
        .chip-cyan {
            border-color: rgba(56, 189, 248, 0.40);
            color: #7DD3FC;
            background: rgba(14, 116, 144, 0.18);
        }
        .chip-emerald {
            border-color: rgba(16, 185, 129, 0.40);
            color: #6EE7B7;
            background: rgba(6, 95, 70, 0.20);
        }
        .chip-amber {
            border-color: rgba(245, 158, 11, 0.42);
            color: #FCD34D;
            background: rgba(146, 64, 14, 0.22);
        }
        .chip-indigo {
            border-color: rgba(129, 140, 248, 0.42);
            color: #C7D2FE;
            background: rgba(67, 56, 202, 0.22);
        }

        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(155deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.86));
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            height: 100%;
            box-shadow: 0 6px 20px rgba(2, 6, 23, 0.30);
            transition: border-color 0.2s ease;
        }
        .kpi-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
        }
        .kpi-accent-cyan { border-top: 3px solid #38BDF8; }
        .kpi-accent-emerald { border-top: 3px solid #10B981; }
        .kpi-accent-amber { border-top: 3px solid #F59E0B; }
        .kpi-accent-rose { border-top: 3px solid #F43F5E; }
        .kpi-accent-indigo { border-top: 3px solid #818CF8; }

        .kpi-label {
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            color: #F8FAFC;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        .kpi-sub {
            font-size: 0.8rem;
            color: #94A3B8;
            line-height: 1.35;
        }
        .delta-pos { color: #34D399; font-weight: 600; }
        .delta-neg { color: #FB7185; font-weight: 600; }
        .delta-neu { color: #7DD3FC; font-weight: 600; }

        /* Callout / Insight Panel */
        .insight-panel {
            background: rgba(17, 24, 39, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-left: 4px solid #38BDF8;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.65rem 0 1rem 0;
        }
        .insight-panel-emerald { border-left-color: #10B981; background: rgba(6, 78, 59, 0.16); }
        .insight-panel-amber { border-left-color: #F59E0B; background: rgba(120, 53, 15, 0.16); }
        .insight-panel-rose { border-left-color: #F43F5E; background: rgba(136, 19, 55, 0.18); }
        .insight-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 0.35rem;
        }
        .insight-body {
            font-size: 0.85rem;
            color: #CBD5E1;
            line-height: 1.55;
        }
        .insight-body ul {
            margin: 0.3rem 0 0 1.1rem;
            padding: 0;
        }
        .insight-body li {
            margin-bottom: 0.25rem;
        }

        /* Section Title */
        .section-heading {
            font-size: 1.05rem;
            font-weight: 700;
            color: #F1F5F9;
            letter-spacing: -0.01em;
            margin: 1.05rem 0 0.45rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }

        /* Sidebar brand & status card */
        .sb-brand {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.24), rgba(67, 56, 202, 0.20));
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.85rem;
        }
        .sb-brand-title {
            font-size: 0.98rem;
            font-weight: 800;
            color: #F8FAFC;
            margin: 0;
        }
        .sb-brand-sub {
            font-size: 0.74rem;
            color: #94A3B8;
            margin-top: 0.18rem;
        }
        .sb-status {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            font-size: 0.78rem;
            color: #CBD5E1;
            line-height: 1.6;
        }
        .sb-status strong { color: #F8FAFC; }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(15, 23, 42, 0.75);
            padding: 0.35rem;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.5rem 0.95rem;
            font-weight: 600;
            font-size: 0.86rem;
            color: #94A3B8;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.20), rgba(99, 102, 241, 0.20)) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        /* Expander polish */
        div[data-testid="stExpander"] {
            background: rgba(17, 24, 39, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 12px;
            margin-bottom: 0.55rem;
        }

        /* Buttons polish */
        .stButton > button {
            border-radius: 9px;
            font-weight: 600;
            font-size: 0.80rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(30, 41, 59, 0.75);
            color: #E2E8F0;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #38BDF8;
            color: #F8FAFC;
            background: rgba(14, 116, 144, 0.28);
        }

        /* Compact metric overrides if used */
        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", accent: str = "cyan") -> str:
    return (
        f'<div class="kpi-card kpi-accent-{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def insight_box(title: str, body_html: str, tone: str = "cyan") -> str:
    cls = f"insight-panel insight-panel-{tone}" if tone != "cyan" else "insight-panel"
    return f'<div class="{cls}"><div class="insight-title">{html.escape(title)}</div><div class="insight-body">{body_html}</div></div>'


def section_heading(text: str):
    st.markdown(f'<div class="section-heading">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_plotly(fig: go.Figure, title: str = "", height: int = 380, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F1F5F9", family="Inter, sans-serif"), x=0.01, xanchor="left")
        if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=48, r=24, t=48 if title else 24, b=42),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.18)",
            borderwidth=1,
            font=dict(size=11, color="#E2E8F0"),
        ),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#38BDF8", font=dict(color="#F8FAFC", size=12)),
    )
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    )
    return fig


inject_custom_css()

# -----------------------------------------------------------------------------
# METADATA FITUR (label, kode ordinal deskriptif, bantuan) - sesuai UCI & Brief
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
    "Medu": "Kode ordinal 0–4: 0 tidak sekolah s.d. 4 pendidikan tinggi.",
    "Fedu": "Kode ordinal 0–4: 0 tidak sekolah s.d. 4 pendidikan tinggi.",
    "traveltime": "Kode kategori waktu tempuh rumah ke sekolah (bukan jumlah jam langsung).",
    "studytime": "Kode kategori durasi belajar per minggu (contoh: kode 2 = 2–5 jam/minggu).",
    "failures": "Jumlah kegagalan kelas sebelumnya pada dataset aktual (0, 1, 2, 3).",
    "famrel": "Skala 1 (sangat buruk) sampai 5 (sangat baik).",
    "freetime": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "goout": "Skala 1 (sangat jarang) sampai 5 (sangat sering).",
    "Dalc": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "Walc": "Skala 1 (sangat rendah) sampai 5 (sangat tinggi).",
    "health": "Skala 1 (sangat buruk) sampai 5 (sangat baik).",
    "absences": "Total ketidakhadiran (0–93). Pada catatan cleaning proyek, fitur ini dikeluarkan dari model regresi default karena risiko informasi masa depan.",
    "G1": "Nilai periode pertama Matematika (skala 0–20).",
    "G2": "Nilai periode kedua Matematika (skala 0–20).",
    "G3": "Nilai akhir Matematika (skala 0–20). Target prediksi regresi.",
}

# Label deskriptif untuk kode ordinal (membedakan kode dari besaran asli sesuai Brief)
ORDINAL_LABELS = {
    "Medu": {0: "0 · Tidak sekolah", 1: "1 · SD (1–4)", 2: "2 · SMP (5–9)", 3: "3 · SMA", 4: "4 · Pend. Tinggi"},
    "Fedu": {0: "0 · Tidak sekolah", 1: "1 · SD (1–4)", 2: "2 · SMP (5–9)", 3: "3 · SMA", 4: "4 · Pend. Tinggi"},
    "traveltime": {1: "1 · <15 mnt", 2: "2 · 15–30 mnt", 3: "3 · 30–60 mnt", 4: "4 · >60 mnt"},
    "studytime": {1: "1 · <2 jam/mgg", 2: "2 · 2–5 jam/mgg", 3: "3 · 5–10 jam/mgg", 4: "4 · >10 jam/mgg"},
    "failures": {0: "0 · Tidak pernah", 1: "1 · 1 kali", 2: "2 · 2 kali", 3: "3 · ≥3 kali"},
    "famrel": {1: "1 · Sangat buruk", 2: "2 · Buruk", 3: "3 · Cukup", 4: "4 · Baik", 5: "5 · Sangat baik"},
    "health": {1: "1 · Sangat buruk", 2: "2 · Buruk", 3: "3 · Cukup", 4: "4 · Baik", 5: "5 · Sangat baik"},
    "freetime": {1: "1 · Sangat rendah", 2: "2 · Rendah", 3: "3 · Sedang", 4: "4 · Tinggi", 5: "5 · Sangat tinggi"},
    "goout": {1: "1 · Sangat jarang", 2: "2 · Jarang", 3: "3 · Sedang", 4: "4 · Sering", 5: "5 · Sangat sering"},
    "Dalc": {1: "1 · Sangat rendah", 2: "2 · Rendah", 3: "3 · Sedang", 4: "4 · Tinggi", 5: "5 · Sangat tinggi"},
    "Walc": {1: "1 · Sangat rendah", 2: "2 · Rendah", 3: "3 · Sedang", 4: "4 · Tinggi", 5: "5 · Sangat tinggi"},
}

VALUE_LABELS = {
    "school": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"},
    "sex": {"F": "Perempuan", "M": "Laki-laki"},
    "address": {"U": "Perkotaan", "R": "Pedesaan"},
    "famsize": {"LE3": "≤ 3 anggota", "GT3": "> 3 anggota"},
    "Pstatus": {"T": "Tinggal bersama", "A": "Terpisah"},
    "guardian": {"mother": "Ibu", "father": "Ayah", "other": "Lainnya"},
    "reason": {"home": "Dekat rumah", "reputation": "Reputasi sekolah", "course": "Pilihan program", "other": "Lainnya"},
    "Mjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Layanan publik", "at_home": "Di rumah", "other": "Lainnya"},
    "Fjob": {"teacher": "Guru", "health": "Kesehatan", "services": "Layanan publik", "at_home": "Di rumah", "other": "Lainnya"},
}
BINARY_YES_NO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
for _c in BINARY_YES_NO:
    VALUE_LABELS[_c] = {"yes": "Ya", "no": "Tidak"}

# Kelompok fitur di Sidebar (G3 sengaja tidak diletakkan di form input prediksi utama)
SIDEBAR_GROUPS = {
    "⭐ Fitur Utama (Prioritas)": ["G1", "G2", "failures", "studytime", "higher", "absences", "Medu", "goout"],
    "📚 Akademik & Dukungan": ["reason", "traveltime", "schoolsup", "paid", "nursery"],
    "👨‍👩‍👧 Latar Belakang Keluarga": ["Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"],
    "👤 Demografi Siswa": ["school", "sex", "age", "address", "famsize", "Pstatus"],
    "🎯 Gaya Hidup & Kesehatan": ["activities", "internet", "romantic", "freetime", "Dalc", "Walc", "health"],
}

# Preset profil siswa untuk simulasi cepat
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
    "dropout": {
        "G1": 6, "G2": 0, "failures": 2, "studytime": 1, "higher": "no", "absences": 0,
        "Medu": 1, "Fedu": 1, "goout": 4, "Dalc": 2, "Walc": 3, "schoolsup": "no",
        "famsup": "no", "internet": "no", "famrel": 3, "age": 18, "address": "R",
    },
}


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
    """Cari dataset di folder app atau cwd; jika tidak ada minta upload."""
    for folder in (BASE, Path.cwd()):
        for fname in ("student-mat.xlsx", "student-mat.csv"):
            p = folder / fname
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
        silhouette=silhouette_score(Z, labels),
        calinski=calinski_harabasz_score(Z, labels),
        davies=davies_bouldin_score(Z, labels),
    )


# =============================================================================
#  BAGIAN C - PELATIHAN MODEL REGRESI (DI-CACHE)
# =============================================================================
@st.cache_resource(show_spinner=False)
def train_regression(_df, data_key, use_prev_grades, exclude_absences, mode):
    cfg = MODE_CFG[mode]
    drop_cols = [TARGET]
    if not use_prev_grades:
        drop_cols += ["G1", "G2"]
    if exclude_absences and "absences" in _df.columns:
        drop_cols.append("absences")

    X = _df.drop(columns=drop_cols).copy()
    y = _df[TARGET].astype(float)
    num_cols, cat_cols = split_columns(X)
    X[num_cols] = X[num_cols].astype(float)

    # 1) Train-test split (stratified menurut kuantil target)
    y_bins = pd.qcut(y, q=5, duplicates="drop", labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins
    )

    # 2) Benchmark algoritma dengan Repeated K-Fold CV
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
        "Random Forest": Pipeline([
            ("pre", P(False)),
            ("m", RandomForestRegressor(n_estimators=cfg["trees"], random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("pre", P(False)),
            ("m", ExtraTreesRegressor(n_estimators=cfg["trees"], random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([("pre", P(False)), ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("pre", P(False)), ("m", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([
            ("pre", P(False)),
            ("m", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    except Exception:
        pass

    cv = RepeatedKFold(n_splits=5, n_repeats=cfg["cv_repeats"], random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows = []
    for name, mdl in models.items():
        r = cross_validate(mdl, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "Model": name,
            "CV R2": r["test_r2"].mean(),
            "CV R2 std": r["test_r2"].std(),
            "CV MAE": -r["test_mae"].mean(),
            "CV RMSE": -r["test_rmse"].mean(),
        })
    bench = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)

    # 3) Tuning hyperparameter pada kandidat terbaik
    param_spaces = {
        "Ridge": {"m__alpha": np.logspace(-2, 3, 30)},
        "Lasso": {"m__alpha": np.logspace(-3, 0.5, 30)},
        "ElasticNet": {"m__alpha": np.logspace(-3, 0.5, 30), "m__l1_ratio": np.linspace(0.1, 0.9, 9)},
        "KNN": {"m__n_neighbors": list(range(3, 40)), "m__weights": ["uniform", "distance"], "m__p": [1, 2]},
        "SVR (RBF)": {
            "m__C": np.logspace(-1, 2, 20),
            "m__epsilon": [0.1, 0.3, 0.5, 1.0],
            "m__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
        },
        "Random Forest": {
            "m__n_estimators": [150, 300, 500],
            "m__max_depth": [None, 4, 6, 8, 12],
            "m__min_samples_leaf": [1, 2, 4, 8],
            "m__max_features": ["sqrt", 0.3, 0.5, 1.0],
        },
        "Extra Trees": {
            "m__n_estimators": [150, 300, 500],
            "m__max_depth": [None, 4, 6, 8, 12],
            "m__min_samples_leaf": [1, 2, 4, 8],
            "m__max_features": ["sqrt", 0.3, 0.5, 1.0],
        },
        "Gradient Boosting": {
            "m__n_estimators": [100, 200, 300, 500],
            "m__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "m__max_depth": [2, 3, 4],
            "m__subsample": [0.6, 0.8, 1.0],
            "m__min_samples_leaf": [1, 3, 5, 10],
        },
        "HistGradientBoosting": {
            "m__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "m__max_iter": [100, 200, 400],
            "m__max_depth": [None, 3, 5],
            "m__min_samples_leaf": [5, 10, 20],
            "m__l2_regularization": [0, 0.1, 1.0],
        },
        "XGBoost": {
            "m__n_estimators": [200, 300, 500],
            "m__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "m__max_depth": [2, 3, 4, 5],
            "m__subsample": [0.6, 0.8, 1.0],
            "m__colsample_bytree": [0.5, 0.8, 1.0],
        },
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
            "CV RMSE (tuned)": -search.best_score_,
            "Best params": str({
                k.replace("m__", ""): (round(v, 4) if isinstance(v, float) else v)
                for k, v in search.best_params_.items()
            }),
        })
    tune_df = pd.DataFrame(tune_rows).sort_values("CV RMSE (tuned)").reset_index(drop=True)
    best_name = tune_df.loc[0, "Model"]
    best_model = tuned[best_name]

    # 4) Evaluasi pada data train, test, baseline, dan benchmark G3=G2
    pred_train = np.clip(best_model.predict(X_train), 0, 20)
    pred_test = np.clip(best_model.predict(X_test), 0, 20)
    rep_dict = {
        "Train": regression_report(y_train, pred_train, X.shape[1]),
        "Test": regression_report(y_test, pred_test, X.shape[1]),
        "Baseline Rata-rata (Test)": regression_report(y_test, np.full(len(y_test), y_train.mean()), X.shape[1]),
    }
    if use_prev_grades and "G2" in X_test.columns:
        rep_dict["Benchmark G3 = G2 (Test)"] = regression_report(y_test, X_test["G2"].values, 1)
    rep = pd.DataFrame(rep_dict).T
    resid = y_test.values - pred_test

    act_pass, pred_pass = (y_test >= PASS_THRESHOLD), (pred_test >= PASS_THRESHOLD)
    tp = int(((~act_pass) & (~pred_pass)).sum())
    fn = int(((~act_pass) & pred_pass).sum())
    fp = int((act_pass & (~pred_pass)).sum())
    tn = int((act_pass & pred_pass).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    risk = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": rec, "precision": prec, "f1": f1}

    # Learning curve (mode Lengkap)
    lc_data = None
    if cfg["learning_curve"]:
        sizes, tr_s, va_s = learning_curve(
            best_model, X_train, y_train, cv=5, n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.2, 1.0, 6), random_state=RANDOM_STATE,
        )
        lc_data = {
            "sizes": sizes.tolist(),
            "train_rmse": (-tr_s.mean(1)).tolist(),
            "val_rmse": (-va_s.mean(1)).tolist(),
        }

    # 5) Interpretasi: permutation importance, PDP numerik, koefisien
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=cfg["perm_repeats"],
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = (
        pd.DataFrame({
            "Fitur": X.columns,
            "Label": [LABELS.get(c, c) for c in X.columns],
            "Kenaikan RMSE bila diacak": perm.importances_mean,
            "Std": perm.importances_std,
        })
        .sort_values("Kenaikan RMSE bila diacak", ascending=False)
        .reset_index(drop=True)
    )

    top_num = [f for f in imp["Fitur"] if f in num_cols][:3]
    pdp_data = {}
    for feat in top_num:
        try:
            pdp_res = partial_dependence(best_model, X_train, features=[feat], grid_resolution=20)
            grid = pdp_res.get("grid_values", pdp_res.get("values"))[0]
            avg = pdp_res["average"][0]
            pdp_data[feat] = {"grid": np.asarray(grid, dtype=float), "avg": np.asarray(avg, dtype=float)}
        except Exception:
            pass

    coef = None
    final_est = best_model.named_steps["m"]
    if hasattr(final_est, "coef_"):
        names = best_model.named_steps["pre"].get_feature_names_out()
        coef = pd.Series(np.ravel(final_est.coef_), index=names).sort_values(key=np.abs, ascending=False).head(10)

    mask0 = y_test.values == 0
    mae_zero = float(np.abs(resid[mask0]).mean()) if mask0.sum() else None
    rmse_zero = float(np.sqrt(np.mean(resid[mask0] ** 2))) if mask0.sum() else None
    mae_other = float(np.abs(resid[~mask0]).mean()) if (~mask0).sum() else None
    rmse_other = float(np.sqrt(np.mean(resid[~mask0] ** 2))) if (~mask0).sum() else None

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
        "metrics_test": {k: float(v) for k, v in rep.loc["Test"].items()},
        "top_features": imp.head(10)["Fitur"].tolist(),
        "pass_threshold": PASS_THRESHOLD,
    }
    return dict(
        model=best_model, schema=schema, bench=bench, tune=tune_df, report=rep, importance=imp,
        risk=risk, coef=coef, pdp_data=pdp_data, lc_data=lc_data,
        y_test=y_test.values, pred_test=pred_test, resid=resid,
        n_train=len(X_train), n_test=len(X_test), n_features=X.shape[1],
        num_cols=num_cols, cat_cols=cat_cols,
        mae_zero=mae_zero, rmse_zero=rmse_zero, mae_other=mae_other, rmse_other=rmse_other,
        n_test_zero=int(mask0.sum()), n_test_other=int((~mask0).sum()),
        tunable=tunable, gap_r2=float(rep.loc["Train", "R2"] - rep.loc["Test", "R2"]),
    )


# =============================================================================
#  BAGIAN D - PELATIHAN MODEL CLUSTERING (DI-CACHE)
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

    df_train, df_test = train_test_split(feat_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Z_train = transformer.fit_transform(df_train)
    Z_test = transformer.transform(df_test)

    # Pemilihan algoritma & k
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
                inertia=mdl.inertia_ if name == "KMeans" else np.nan,
                bic=mdl.bic(Z_train) if name == "GMM" else np.nan,
            )
            rows.append(m)
    sel = pd.DataFrame(rows)
    sel["rank_score"] = (
        sel["silhouette"].rank(ascending=False)
        + sel["calinski"].rank(ascending=False)
        + sel["davies"].rank(ascending=True)
    )

    # DBSCAN (pembanding)
    nn = NearestNeighbors(n_neighbors=10).fit(Z_train)
    kdist = np.sort(nn.kneighbors(Z_train)[0][:, -1])
    db_rows = []
    for eps in np.quantile(kdist, [0.5, 0.6, 0.7, 0.8, 0.9]):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Z_train)
        mask = lab != -1
        n_cl = len(set(lab[mask]))
        sil = silhouette_score(Z_train[mask], lab[mask]) if n_cl >= 2 and mask.sum() > n_cl else np.nan
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
            n_components=k, covariance_type="diag", n_init=max(cfg["gmm_init"], 5), random_state=RANDOM_STATE
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

    # Interpretasi: label diurutkan menurut rata-rata G3 (0 = terendah -> k-1 = tertinggi)
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
    pass_rates = df_all.groupby("cluster")["G3"].apply(lambda s: float((s >= PASS_THRESHOLD).mean()))

    cat_report = []
    for c in cat_cols:
        ct = pd.crosstab(df_all["cluster"], df_all[c], normalize="index")
        overall = df_all[c].value_counts(normalize=True)
        for cl in ct.index:
            for cat in ct.columns:
                cat_report.append({
                    "cluster": cl, "fitur": c, "kategori": cat,
                    "proporsi_cluster": ct.loc[cl, cat],
                    "proporsi_populasi": overall[cat],
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
            traits.append("belajar rajin")
        if z.get("Dalc", 0) + z.get("Walc", 0) > 1.0:
            traits.append("konsumsi alkohol tinggi")
        if z.get("goout", 0) > 0.5:
            traits.append("sosialisasi tinggi")
        return ", ".join(traits)

    cluster_names = {int(cl): f"Cluster {cl} — {label_cluster(cl)}" for cl in prof.index}
    schema = {
        "algorithm": best_algo,
        "k": best_k,
        "include_grades": bool(include_grades),
        "feature_order": feat_df.columns.tolist(),
        "numeric": {
            c: {"min": float(feat_df[c].min()), "max": float(feat_df[c].max()), "median": float(feat_df[c].median())}
            for c in num_cols
        },
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
        rand_sil=rand_sil, df_all=df_all, prof=prof, z_prof=z_prof, pass_rates=pass_rates,
        size=size, notable=notable, kruskal_H=float(H), kruskal_p=float(p),
        P2=P2, pca2_var=pca2.explained_variance_ratio_.tolist(), sample_sil=sample_sil,
        n_train=len(df_train), n_test=len(df_test), n_pc=Z_train.shape[1],
        var_pc=float(transformer.named_steps["pca"].explained_variance_ratio_.sum()),
    )


# =============================================================================
#  BAGIAN E - FUNGSI BANTU APLIKASI & FORM SIDEBAR
# =============================================================================
def is_whole(x) -> bool:
    return float(x).is_integer()


def merged_feature_specs(schemas):
    numeric, categorical, order = {}, {}, []
    for sch in schemas:
        for c in sch["feature_order"]:
            if c == TARGET:
                continue  # G3 tidak masuk form input prediksi utama
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
        elif col in categorical:
            if val in categorical[col]:
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


def render_input_form(order, numeric, categorical) -> dict:
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

    for gname, cols in SIDEBAR_GROUPS.items():
        present = [c for c in cols if c in order]
        if not present:
            continue
        with st.expander(gname, expanded=(gname.startswith("⭐"))):
            for c in present:
                inputs[c] = widget(c)
                placed.add(c)

    rest = [c for c in order if c not in placed]
    if rest:
        with st.expander("📌 Fitur Lainnya", expanded=False):
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


def generate_student_insights(inputs: dict, pred: float, thr: float, use_prev: bool) -> tuple[str, str, list[str]]:
    notes = []
    if pred < thr:
        tone = "rose"
        title = "Perlu Intervensi Dini — Prediksi di Bawah Batas Lulus (< 10)"
    elif pred < thr + 2.0:
        tone = "amber"
        title = "Zona Waspada — Prediksi Mendekati Batas Lulus (10–12)"
    else:
        tone = "emerald"
        title = "Performa Aman — Prediksi di Atas Batas Lulus (≥ 12)"

    if use_prev and "G1" in inputs and "G2" in inputs:
        g1, g2 = float(inputs["G1"]), float(inputs["G2"])
        if g2 == 0:
            notes.append("<strong>Nilai G2 = 0</strong>: Pada data historis, seluruh siswa dengan G2 = 0 berakhir dengan G3 = 0 (indikasi tidak mengikuti ujian / drop out). Verifikasi kehadiran & pendampingan segera sangat disarankan.")
        elif g2 < g1 - 1.5:
            notes.append(f"<strong>Tren nilai menurun</strong> dari G1 ({g1:g}) ke G2 ({g2:g}). Identifikasi materi semester kedua yang menjadi hambatan.")
        elif g2 > g1 + 1.5:
            notes.append(f"<strong>Tren nilai membaik</strong> dari G1 ({g1:g}) ke G2 ({g2:g}). Pertahankan ritme belajar saat ini.")

    if inputs.get("failures", 0) >= 1:
        notes.append(f"<strong>Riwayat kegagalan kelas ({int(inputs['failures'])} kali)</strong> berkorelasi negatif kuat terhadap nilai akhir pada data historis.")
    if inputs.get("studytime", 2) <= 1:
        notes.append("<strong>Waktu belajar mingguan rendah (&lt;2 jam/minggu)</strong>. Menambah jadwal belajar terstruktur dapat membantu konsistensi pemahaman.")
    if inputs.get("higher") == "no":
        notes.append("<strong>Motivasi studi lanjut</strong>: Siswa yang tidak berencana kuliah secara historis memiliki rata-rata G3 jauh lebih rendah (6,8 vs 10,6).")
    if inputs.get("goout", 3) >= 4:
        notes.append("<strong>Intensitas keluar bersama teman tinggi</strong>; perhatikan keseimbangan waktu sosial dan persiapan ujian.")
    if inputs.get("Dalc", 1) >= 3 or inputs.get("Walc", 1) >= 4:
        notes.append("<strong>Konsumsi alkohol tinggi</strong> tercatat sebagai faktor gaya hidup yang berasosiasi dengan penurunan fokus akademik.")
    if inputs.get("absences", 0) >= 15:
        notes.append(f"<strong>Ketidakhadiran tinggi ({int(inputs['absences'])} hari)</strong> berpotensi membuat siswa tertinggal materi inti.")

    if not notes:
        notes.append("Profil belajar, riwayat akademik, dan gaya hidup siswa berada pada rentang yang mendukung capaian nilai stabil.")
    notes.append("<em>Catatan metodologis: Seluruh indikator di atas merupakan pola asosiasi statistik dari data historis Portugal (2005–2006), bukan bukti sebab-akibat langsung.</em>")
    return tone, title, notes


# =============================================================================
#  BAGIAN F - SIDEBAR: PENGATURAN MODEL & PROFIL SISWA
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-title">🎓 Student Math Analytics</div>
            <div class="sb-brand-sub">Executive Dark ML Dashboard · UCI student-mat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Pengaturan Model & Metodologi", expanded=False):
        with st.form("model_settings"):
            f_prev = st.toggle(
                "Regresi: Gunakan nilai G1 & G2",
                value=True,
                help="Nonaktifkan untuk skenario peringatan dini sebelum nilai periode 1 & 2 tersedia.",
            )
            f_no_abs = st.toggle(
                "Regresi: Kecualikan fitur absences",
                value=True,
                help="Sesuai catatan cleaning proyek: waktu pencatatan absences tidak terinci dan seluruh G3=0 memiliki absences=0 (risiko data leakage).",
            )
            f_inc = st.toggle(
                "Clustering: Sertakan fitur G1–G3",
                value=True,
                help="Toggle penggunaan nilai (termasuk G3) pada model clustering. Nonaktifkan untuk segmentasi murni berbasis profil non-nilai.",
            )
            f_mink = st.slider(
                "Clustering: k minimal model final", 2, 5, 3,
                help="k = 2 cenderung hanya membelah siswa menjadi kelompok tinggi dan rendah.",
            )
            f_mode = st.radio(
                "Mode pelatihan", ["Cepat", "Lengkap"], index=0, horizontal=True,
                help="Cepat: tuning ringan & cepat. Lengkap: evaluasi + learning curve menyeluruh.",
            )
            st.form_submit_button("🔄 Terapkan & Latih Ulang", width="stretch")

    use_prev_grades, exclude_absences, include_grades, min_k_final, train_mode = (
        f_prev, f_no_abs, f_inc, f_mink, f_mode,
    )

# =============================================================================
#  BAGIAN G - PELATIHAN MODEL (OTOMATIS, DI-CACHE PER KONFIGURASI)
# =============================================================================
with st.spinner(f"Melatih & mengevaluasi model regresi (mode {train_mode})..."):
    RES_R = train_regression(df_raw, DATA_KEY, use_prev_grades, exclude_absences, train_mode)
with st.spinner(f"Melatih & mengevaluasi model clustering (mode {train_mode})..."):
    RES_C = train_clustering(df_raw, DATA_KEY, include_grades, min_k_final, train_mode)

reg_model, reg_schema = RES_R["model"], RES_R["schema"]
clu_model, clu_schema = RES_C["model"], RES_C["schema"]

# Siapkan form profil siswa di sidebar (dengan Preset & Reset)
order, numeric_specs, categorical_specs = merged_feature_specs([reg_schema, clu_schema])
default_inputs = compute_default_inputs(df_raw, order, numeric_specs, categorical_specs)
ensure_session_inputs(default_inputs, numeric_specs, categorical_specs)

with st.sidebar:
    st.markdown("#### 👤 Profil Siswa")
    st.caption("Pilih skenario cepat atau sesuaikan profil di bawah. Hasil diperbarui otomatis.")

    p1, p2 = st.columns(2)
    p1.button(
        "🏆 Berprestasi", width="stretch",
        on_click=apply_preset_callback,
        args=(STUDENT_PRESETS["berprestasi"], default_inputs, numeric_specs, categorical_specs),
    )
    p2.button(
        "📊 Median", width="stretch",
        on_click=apply_preset_callback,
        args=({}, default_inputs, numeric_specs, categorical_specs),
    )
    p3, p4 = st.columns(2)
    p3.button(
        "⚠️ Berisiko", width="stretch",
        on_click=apply_preset_callback,
        args=(STUDENT_PRESETS["berisiko"], default_inputs, numeric_specs, categorical_specs),
    )
    p4.button(
        "🚨 Drop Out", width="stretch",
        on_click=apply_preset_callback,
        args=(STUDENT_PRESETS["dropout"], default_inputs, numeric_specs, categorical_specs),
    )

    user_inputs = render_input_form(order, numeric_specs, categorical_specs)

    st.divider()
    rm = reg_schema["metrics_test"]
    cm = clu_schema["metrics"]
    st.markdown(
        f"""
        <div class="sb-status">
            <strong>Model Aktif ({html.escape(train_mode)})</strong><br/>
            • <strong>Regresi:</strong> {html.escape(reg_schema['model_name'])}<br/>
            &nbsp;&nbsp;R² = {rm['R2']:.3f} · RMSE = {rm['RMSE']:.2f} · MAE = {rm['MAE']:.2f}<br/>
            • <strong>Clustering:</strong> {html.escape(clu_schema['algorithm'])} (k = {clu_schema['k']})<br/>
            &nbsp;&nbsp;Silhouette = {cm['silhouette_test']:.2f} · ARI = {cm['stability_ari']:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Hitung prediksi regresi utama sekali untuk dipakai di Hero, Tab 1, dan Tab 2
X_reg_row = to_frame(user_inputs, reg_schema["feature_order"], reg_schema["numeric"].keys())
pred_g3_val = float(predict_g3(reg_model, X_reg_row)[0])

# =============================================================================
#  HERO BANNER & NAVIGASI TAB UTAMA
# =============================================================================
mode_chip = "Dengan G1 & G2" if use_prev_grades else "Mode Peringatan Dini (Tanpa G1/G2)"
abs_chip = "Absences: Dikecualikan (Clean)" if exclude_absences else "Absences: Disertakan"
clu_g3_chip = "Clustering: + G1–G3" if include_grades else "Clustering: Non-Nilai"

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-kicker">Project Data Sains · Machine Learning &amp; Streamlit Analytics</div>
        <div class="hero-title">Prediksi Nilai Akhir Matematika Siswa (G3)</div>
        <div class="hero-subtitle">
            Dashboard analitik terpadu untuk estimasi nilai akhir (regresi), simulasi what-if, segmentasi profil siswa (clustering), prediksi massal, dan evaluasi model end-to-end.
        </div>
        <div class="chip-row">
            <span class="chip chip-cyan">📊 Dataset: {len(df_raw)} Siswa · {df_raw.shape[1]} Fitur</span>
            <span class="chip chip-emerald">🤖 Regresi: {html.escape(reg_schema['model_name'])} (R² {rm['R2']:.2f})</span>
            <span class="chip chip-indigo">🧩 {html.escape(clu_schema['algorithm'])} k={clu_schema['k']} · {clu_g3_chip}</span>
            <span class="chip chip-amber">⚙️ {mode_chip}</span>
            <span class="chip">🛡️ {abs_chip}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_reg, tab_clu, tab_batch, tab_evr, tab_evc, tab_data = st.tabs([
    "📈 Prediksi Nilai",
    "🧩 Segmentasi Siswa",
    "📂 Prediksi Massal",
    "🔬 Evaluasi Regresi",
    "🔬 Evaluasi Clustering",
    "🗂️ Data, EDA & Unduhan",
])


# =============================================================================
#  TAB 1 - PREDIKSI NILAI (HERO TAB)
# =============================================================================
with tab_reg:
    thr = reg_schema["pass_threshold"]
    rmse = reg_schema["metrics_test"]["RMSE"]
    mae = reg_schema["metrics_test"]["MAE"]
    r2_val = reg_schema["metrics_test"]["R2"]
    lo_b, hi_b = max(0.0, pred_g3_val - rmse), min(20.0, pred_g3_val + rmse)
    pop_mean = float(df_raw[TARGET].mean())
    delta_mean = pred_g3_val - pop_mean
    percentile = float((df_raw[TARGET] <= pred_g3_val).mean() * 100)

    if pred_g3_val < thr:
        status_label, status_accent, gauge_color = "Berisiko", "rose", COLORS["rose"]
        status_sub = f'<span class="delta-neg">{pred_g3_val - thr:+.1f} poin</span> dari batas lulus ({thr})'
    elif pred_g3_val < thr + 2.0:
        status_label, status_accent, gauge_color = "Waspada", "amber", COLORS["amber"]
        status_sub = f'<span class="delta-neu">{pred_g3_val - thr:+.1f} poin</span> di atas batas lulus ({thr})'
    else:
        status_label, status_accent, gauge_color = "Aman · Lulus", "emerald", COLORS["emerald"]
        status_sub = f'<span class="delta-pos">{pred_g3_val - thr:+.1f} poin</span> di atas batas lulus ({thr})'

    # Baris 1: Gauge + 4 KPI Cards
    col_gauge, col_kpis = st.columns([5, 7], gap="large")

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(pred_g3_val, 2),
            number={"suffix": " / 20", "font": {"size": 36, "color": "#F8FAFC", "family": "Inter"}},
            delta={
                "reference": round(pop_mean, 2),
                "increasing": {"color": COLORS["emerald"]},
                "decreasing": {"color": COLORS["rose"]},
                "suffix": " vs rata-rata",
                "font": {"size": 14},
            },
            title={"text": "Estimasi Nilai Akhir Matematika (G3)", "font": {"size": 14, "color": "#CBD5E1"}},
            gauge={
                "axis": {"range": [0, 20], "tickwidth": 1, "tickcolor": "#64748B", "dtick": 4},
                "bar": {"color": gauge_color, "thickness": 0.30},
                "bgcolor": "rgba(15, 23, 42, 0.7)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.2)",
                "steps": [
                    {"range": [0, 10], "color": "rgba(244, 63, 94, 0.16)"},
                    {"range": [10, 14], "color": "rgba(56, 189, 248, 0.14)"},
                    {"range": [14, 20], "color": "rgba(16, 185, 129, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "thickness": 0.85,
                    "value": thr,
                },
            },
        ))
        fig_gauge.update_layout(
            height=290,
            paper_bgcolor="rgba(17, 24, 39, 0.82)",
            font=dict(family="Inter, sans-serif", color="#E2E8F0"),
            margin=dict(l=24, r=24, t=54, b=16),
        )
        st.plotly_chart(fig_gauge, width="stretch", theme=None)

    with col_kpis:
        k_r1_c1, k_r1_c2 = st.columns(2, gap="medium")
        delta_cls = "delta-pos" if delta_mean >= 0 else "delta-neg"
        k_r1_c1.markdown(
            kpi_card(
                "Prediksi Nilai Akhir (G3)",
                f"{pred_g3_val:.1f} / 20",
                f'<span class="{delta_cls}">{delta_mean:+.1f} poin</span> vs rata-rata kelas ({pop_mean:.1f})',
                accent="cyan",
            ),
            unsafe_allow_html=True,
        )
        k_r1_c2.markdown(
            kpi_card("Status Risiko Akademik", status_label, status_sub, accent=status_accent),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
        k_r2_c1, k_r2_c2 = st.columns(2, gap="medium")
        k_r2_c1.markdown(
            kpi_card(
                "Rentang Perkiraan (±1 RMSE)",
                f"{lo_b:.1f} – {hi_b:.1f}",
                f"RMSE test: {rmse:.2f} · MAE test: {mae:.2f} · R²: {r2_val:.2f}",
                accent="indigo",
            ),
            unsafe_allow_html=True,
        )
        k_r2_c2.markdown(
            kpi_card(
                "Posisi di Populasi",
                f"Persentil ke-{percentile:.0f}",
                f"Lebih tinggi/setara dengan {percentile:.0f}% dari {len(df_raw)} siswa",
                accent="cyan",
            ),
            unsafe_allow_html=True,
        )

    # Ringkasan & rekomendasi kontekstual
    tone, insight_title, insight_items = generate_student_insights(user_inputs, pred_g3_val, thr, use_prev_grades)
    ul_html = "<ul>" + "".join(f"<li>{item}</li>" for item in insight_items) + "</ul>"
    st.markdown(insight_box(insight_title, ul_html, tone=tone), unsafe_allow_html=True)

    # Baris 2: Tren Nilai Siswa & Posisi dalam Distribusi Populasi
    v_left, v_right = st.columns(2, gap="large")

    with v_left:
        if use_prev_grades and "G1" in user_inputs and "G2" in user_inputs:
            stages = ["Periode 1 (G1)", "Periode 2 (G2)", "Prediksi Akhir (G3)"]
            student_scores = [float(user_inputs["G1"]), float(user_inputs["G2"]), pred_g3_val]
            pop_scores = [float(df_raw["G1"].mean()), float(df_raw["G2"].mean()), pop_mean]

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=stages, y=pop_scores, mode="lines+markers",
                name="Rata-rata Populasi",
                line=dict(color="#64748B", width=2, dash="dot"),
                marker=dict(size=7, color="#94A3B8"),
            ))
            fig_traj.add_trace(go.Scatter(
                x=stages, y=student_scores, mode="lines+markers+text",
                name="Siswa Ini",
                line=dict(color=COLORS["cyan"], width=3.5),
                marker=dict(size=11, color=gauge_color, line=dict(width=2, color="#F8FAFC")),
                text=[f"{v:.1f}" for v in student_scores],
                textposition="top center",
                textfont=dict(color="#F8FAFC", size=12),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.10)",
            ))
            fig_traj.add_hline(
                y=thr, line_dash="dash", line_color=COLORS["rose"], opacity=0.8,
                annotation_text=f"Batas Lulus ({thr})", annotation_position="bottom right",
                annotation_font_color="#FDA4AF",
            )
            style_plotly(fig_traj, "Lintasan Perkembangan Nilai (G1 → G2 → Prediksi G3)", height=340)
            fig_traj.update_yaxes(range=[0, 21.5], title="Skala Nilai (0–20)")
            st.plotly_chart(fig_traj, width="stretch", theme=None)
        else:
            # Mode peringatan dini (tanpa G1/G2): bandingkan fitur kunci siswa vs lulus/tidak lulus
            cmp_feats = [c for c in ["failures", "studytime", "Medu", "Fedu", "goout", "famrel"] if c in user_inputs]
            passed_df = df_raw[df_raw[TARGET] >= thr]
            failed_df = df_raw[df_raw[TARGET] < thr]
            fig_early = go.Figure()
            lbls = [LABELS.get(c, c) for c in cmp_feats]
            fig_early.add_trace(go.Bar(name="Siswa Ini", x=lbls, y=[float(user_inputs[c]) for c in cmp_feats], marker_color=COLORS["cyan"]))
            fig_early.add_trace(go.Bar(name="Rata-rata Lulus (≥10)", x=lbls, y=[float(passed_df[c].mean()) for c in cmp_feats], marker_color=COLORS["emerald"]))
            fig_early.add_trace(go.Bar(name="Rata-rata Berisiko (<10)", x=lbls, y=[float(failed_df[c].mean()) for c in cmp_feats], marker_color=COLORS["rose"]))
            fig_early.update_layout(barmode="group")
            style_plotly(fig_early, "Profil Siswa vs Rata-rata Kelompok Lulus & Berisiko (Mode Peringatan Dini)", height=340)
            st.plotly_chart(fig_early, width="stretch", theme=None)

    with v_right:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_raw[TARGET], nbinsx=21, name="Populasi Siswa",
            marker=dict(color="rgba(129, 140, 248, 0.55)", line=dict(color="rgba(148, 163, 184, 0.35)", width=1)),
        ))
        fig_dist.add_vline(
            x=thr, line_dash="dash", line_color=COLORS["rose"], line_width=2,
            annotation_text="Batas Lulus (10)", annotation_position="top left",
            annotation_font_color="#FDA4AF",
        )
        fig_dist.add_vline(
            x=pred_g3_val, line_dash="solid", line_color=COLORS["cyan"], line_width=3,
            annotation_text=f"Prediksi: {pred_g3_val:.1f}", annotation_position="top right",
            annotation_font_color="#7DD3FC",
        )
        style_plotly(fig_dist, "Posisi Prediksi terhadap Distribusi G3 Populasi (395 Siswa)", height=340, show_legend=False)
        fig_dist.update_xaxes(title="Nilai Akhir (G3)", range=[-0.6, 20.6])
        fig_dist.update_yaxes(title="Jumlah Siswa")
        st.plotly_chart(fig_dist, width="stretch", theme=None)

    # Baris 3: Simulasi What-If Interaktif
    section_heading("🎛️ Simulasi What-If Interaktif")
    st.caption("Pilih satu fitur numerik/ordinal untuk disimulasikan sambil menahan fitur lainnya tetap sesuai profil sidebar.")

    num_feats = [c for c in reg_schema["feature_order"] if c in reg_schema["numeric"]]
    top_pref = [c for c in reg_schema["top_features"] if c in num_feats]
    wf_options = top_pref + [c for c in num_feats if c not in top_pref]

    wf_c1, wf_c2 = st.columns([4, 8], gap="large")
    with wf_c1:
        feat = st.selectbox(
            "Fitur yang disimulasikan", wf_options,
            format_func=lambda c: f"{LABELS.get(c, c)} ({c})",
            key="whatif_feat",
        )
        spec = reg_schema["numeric"][feat]
        if is_whole(spec["min"]) and is_whole(spec["max"]):
            grid = np.arange(int(spec["min"]), int(spec["max"]) + 1, dtype=float)
        else:
            grid = np.linspace(spec["min"], spec["max"], 30)

        X_var = X_reg_row.loc[X_reg_row.index.repeat(len(grid))].reset_index(drop=True)
        X_var[feat] = grid
        sim_preds = predict_g3(reg_model, X_var)
        best_idx = int(np.argmax(sim_preds))
        best_val, best_pred = float(grid[best_idx]), float(sim_preds[best_idx])
        cur_val = float(user_inputs[feat])
        gain = best_pred - pred_g3_val

        fmt_val = (
            ORDINAL_LABELS[feat].get(int(round(best_val)), f"{best_val:g}")
            if feat in ORDINAL_LABELS else f"{best_val:g}"
        )
        fmt_cur = (
            ORDINAL_LABELS[feat].get(int(round(cur_val)), f"{cur_val:g}")
            if feat in ORDINAL_LABELS else f"{cur_val:g}"
        )

        st.markdown(
            kpi_card(
                "Potensi Prediksi Tertinggi",
                f"{best_pred:.1f} / 20",
                f"Saat <strong>{html.escape(LABELS.get(feat, feat))}</strong> = {html.escape(fmt_val)}<br/>"
                f"Saat ini ({html.escape(fmt_cur)}): <strong>{pred_g3_val:.1f}</strong> "
                f"(<span class='delta-pos'>{gain:+.1f} poin</span>)",
                accent="emerald" if gain > 0.2 else "cyan",
            ),
            unsafe_allow_html=True,
        )
        st.caption("Hasil simulasi mencerminkan respons model terhadap pola data historis, bukan jaminan sebab-akibat.")

    with wf_c2:
        x_tick_labels = (
            [ORDINAL_LABELS[feat].get(int(v), f"{v:g}") for v in grid]
            if feat in ORDINAL_LABELS else None
        )
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=grid, y=sim_preds, mode="lines+markers",
            name="Kurva Prediksi G3",
            line=dict(color=COLORS["cyan"], width=3),
            marker=dict(size=7, color=COLORS["cyan"]),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)",
            hovertemplate=f"{LABELS.get(feat, feat)}: %{{x}}<br>Prediksi G3: %{{y:.2f}}<extra></extra>",
        ))
        fig_wf.add_trace(go.Scatter(
            x=[cur_val], y=[pred_g3_val], mode="markers",
            name="Posisi Siswa Saat Ini",
            marker=dict(size=14, color=COLORS["amber"], symbol="diamond", line=dict(width=2, color="#F8FAFC")),
            hovertemplate="Saat ini: %{x}<br>Prediksi G3: %{y:.2f}<extra></extra>",
        ))
        fig_wf.add_hline(
            y=thr, line_dash="dash", line_color=COLORS["rose"], opacity=0.75,
            annotation_text=f"Batas Lulus ({thr})", annotation_position="bottom right",
            annotation_font_color="#FDA4AF",
        )
        style_plotly(fig_wf, f"Sensitivitas Prediksi G3 terhadap {LABELS.get(feat, feat)}", height=330)
        fig_wf.update_yaxes(title="Prediksi G3 (0–20)", range=[0, 21])
        if x_tick_labels is not None:
            fig_wf.update_xaxes(title=LABELS.get(feat, feat), tickmode="array", tickvals=grid, ticktext=x_tick_labels)
        else:
            fig_wf.update_xaxes(title=LABELS.get(feat, feat))
        st.plotly_chart(fig_wf, width="stretch", theme=None)


# =============================================================================
#  TAB 2 - SEGMENTASI SISWA
# =============================================================================
with tab_clu:
    clu_inputs = dict(user_inputs)
    if clu_schema["include_grades"]:
        tg_col1, tg_col2 = st.columns([6, 6], gap="large")
        with tg_col1:
            use_manual_g3 = st.toggle(
                "Gunakan nilai G3 aktual manual untuk clustering",
                value=False,
                key="toggle_manual_g3",
                help="Nonaktif = otomatis menggunakan hasil prediksi G3 dari model regresi agar lebih praktis. Aktif = masukkan nilai G3 aktual siswa jika sudah diketahui.",
            )
        with tg_col2:
            if use_manual_g3:
                manual_g3 = st.slider(
                    "Nilai Akhir (G3) Aktual untuk Clustering", 0, 20, int(round(pred_g3_val)), key="manual_g3_val"
                )
                clu_inputs[TARGET] = float(manual_g3)
                g3_source_txt = f"Menggunakan nilai G3 aktual manual = {manual_g3}"
            else:
                clu_inputs[TARGET] = round(pred_g3_val, 2)
                g3_source_txt = f"Otomatis menggunakan hasil Prediksi G3 = {pred_g3_val:.1f}"
    else:
        g3_source_txt = "Mode segmentasi non-nilai (G1–G3 dinonaktifkan di Pengaturan Model)"

    X_c = to_frame(clu_inputs, clu_schema["feature_order"], clu_schema["numeric"].keys())
    cl = int(predict_cluster(clu_model, clu_schema, X_c)[0])
    names = clu_schema["cluster_names"]
    prof_df, df_all = RES_C["prof"], RES_C["df_all"]
    cl_size = int(RES_C["size"].loc[cl])
    cl_pct = cl_size / len(df_all) * 100
    cl_g3 = float(prof_df.loc[cl, "G3"])
    cl_pass = float(RES_C["pass_rates"].loc[cl] * 100)
    mt = clu_schema["metrics"]

    # Banner Cluster + 4 KPI Cards
    cl_tone = "emerald" if cl_g3 >= 12 else ("amber" if cl_g3 >= 9.5 else "rose")
    st.markdown(
        insight_box(
            f"Hasil Segmentasi: {names.get(cl, f'Cluster {cl}')}",
            f"Sumber fitur nilai: <strong>{html.escape(g3_source_txt)}</strong>.<br/>"
            f"Model <strong>{html.escape(clu_schema['algorithm'])} (k = {clu_schema['k']})</strong> · "
            f"Silhouette test = <strong>{mt['silhouette_test']:.2f}</strong> · Stabilitas ARI = <strong>{mt['stability_ari']:.2f}</strong>. "
            "Pada data survei campuran dengan silhouette &lt; 0,25, segmentasi sebaiknya dibaca sebagai kecenderungan deskriptif, bukan batas kaku.",
            tone=cl_tone,
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(kpi_card("Cluster Terpilih", f"Cluster {cl}", html.escape(names.get(cl, "").split("—")[-1].strip()), accent="indigo"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Ukuran Cluster", f"{cl_size} siswa", f"{cl_pct:.1f}% dari total populasi", accent="cyan"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Rata-rata G3 Cluster", f"{cl_g3:.1f} / 20", f"Rata-rata populasi: {df_all['G3'].mean():.1f}", accent=cl_tone), unsafe_allow_html=True)
    c4.markdown(kpi_card("Tingkat Lulus Cluster", f"{cl_pass:.0f}%", f"Persentase siswa dengan G3 ≥ {PASS_THRESHOLD}", accent="emerald" if cl_pass >= 70 else "amber"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
    ch1, ch2 = st.columns([5, 7], gap="large")

    with ch1:
        if hasattr(clu_model, "predict_proba"):
            proba = clu_model.predict_proba(X_c)[0]
            lm = clu_schema["label_map"]
            pr = pd.Series({f"Cluster {lm[i]}": p * 100 for i, p in enumerate(proba)}).sort_index()
            fig_pr = go.Figure(go.Bar(
                x=pr.values, y=pr.index, orientation="h",
                marker_color=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(pr))],
                text=[f"{v:.1f}%" for v in pr.values], textposition="outside",
            ))
            style_plotly(fig_pr, "Probabilitas Keanggotaan Cluster (GMM)", height=350, show_legend=False)
            fig_pr.update_xaxes(title="Probabilitas (%)", range=[0, 112])
            st.plotly_chart(fig_pr, width="stretch", theme=None)
        else:
            sizes_s = RES_C["size"].sort_index()
            fig_sz = go.Figure(go.Pie(
                labels=[f"Cluster {i}" for i in sizes_s.index],
                values=sizes_s.values,
                hole=0.52,
                marker=dict(colors=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in sizes_s.index],
                            line=dict(color="#0B1120", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#F8FAFC"),
            ))
            style_plotly(fig_sz, "Proporsi Jumlah Siswa per Cluster", height=350, show_legend=True)
            st.plotly_chart(fig_sz, width="stretch", theme=None)

    with ch2:
        # Radar perbandingan (dinormalisasi 0-100% sesuai min-max dataset)
        radar_cols = [c for c in ["G1", "G2", "G3", "studytime", "failures", "goout", "absences", "Medu"]
                      if c in prof_df.columns and c in clu_inputs]
        r_labels = [LABELS.get(c, c) for c in radar_cols]

        def norm_vals(vals_dict):
            out = []
            for c in radar_cols:
                lo_v, hi_v = float(df_all[c].min()), float(df_all[c].max())
                span = max(hi_v - lo_v, 1e-6)
                out.append((float(vals_dict[c]) - lo_v) / span * 100)
            return out

        stu_r = norm_vals(clu_inputs)
        clu_r = norm_vals(prof_df.loc[cl].to_dict())
        pop_r = norm_vals(df_all[radar_cols].mean().to_dict())

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(
            r=pop_r + [pop_r[0]], theta=r_labels + [r_labels[0]],
            name="Rata-rata Populasi", line=dict(color="#64748B", dash="dot", width=2),
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=clu_r + [clu_r[0]], theta=r_labels + [r_labels[0]],
            name=f"Rata-rata Cluster {cl}", fill="toself",
            fillcolor="rgba(129, 140, 248, 0.18)", line=dict(color=COLORS["indigo"], width=2.5),
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=stu_r + [stu_r[0]], theta=r_labels + [r_labels[0]],
            name="Siswa Ini", fill="toself",
            fillcolor="rgba(56, 189, 248, 0.22)", line=dict(color=COLORS["cyan"], width=3),
        ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="rgba(15, 23, 42, 0.55)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.16)", tickfont=dict(size=10, color="#94A3B8")),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.16)", tickfont=dict(size=11, color="#E2E8F0")),
            ),
        )
        style_plotly(fig_rad, f"Radar Profil Siswa vs Cluster {cl} (Skala Relatif 0–100%)", height=350)
        st.plotly_chart(fig_rad, width="stretch", theme=None)

    section_heading("📋 Perbandingan Detail & Ringkasan Seluruh Cluster")
    t_left, t_right = st.columns([5, 7], gap="large")
    with t_left:
        st.caption("Perbandingan fitur numerik utama (nilai asli)")
        key_cols = [c for c in ["G1", "G2", "G3", "absences", "failures", "studytime", "goout", "Dalc", "Walc", "Medu", "Fedu", "age"]
                    if c in prof_df.columns and c in clu_inputs]
        cmp_df = pd.DataFrame({
            "Siswa Ini": [clu_inputs[c] for c in key_cols],
            f"Rata-rata C{cl}": [prof_df.loc[cl, c] for c in key_cols],
            "Populasi": [df_all[c].mean() for c in key_cols],
        }, index=[LABELS.get(c, c) for c in key_cols]).astype(float).round(2)
        st.dataframe(cmp_df, width="stretch")

    with t_right:
        st.caption("Profil rata-rata seluruh cluster")
        prof_show = prof_df.copy()
        prof_show.index = [names.get(i, f"Cluster {i}") for i in prof_show.index]
        prof_show.columns = [LABELS.get(c, c) for c in prof_show.columns]
        st.dataframe(prof_show.T.round(2), width="stretch")


# =============================================================================
#  TAB 3 - PREDIKSI MASSAL (BATCH PREDICTION)
# =============================================================================
with tab_batch:
    section_heading("📂 Prediksi Massal & Segmentasi dari File")
    st.caption("Unggah file CSV atau Excel berisi banyak data siswa, atau gunakan data sampel bawaan untuk mencoba fitur batch secara instan.")

    needed = []
    for sch in (reg_schema, clu_schema):
        needed += [c for c in sch["feature_order"] if c not in needed and c != TARGET]
    template = pd.DataFrame([user_inputs])[[c for c in needed if c in user_inputs]]

    b_act1, b_act2, b_act3 = st.columns([4, 4, 4], gap="medium")
    with b_act1:
        st.download_button(
            "⬇️ Unduh Template CSV (1 baris contoh)",
            template.to_csv(index=False).encode("utf-8"),
            file_name="template_siswa.csv",
            mime="text/csv",
            width="stretch",
        )
    with b_act2:
        if st.button("⚡ Coba 30 Sampel Acak Dataset", width="stretch"):
            st.session_state["batch_demo_df"] = df_raw.sample(n=min(30, len(df_raw)), random_state=RANDOM_STATE).drop(columns=[TARGET]).reset_index(drop=True)
    with b_act3:
        if "batch_demo_df" in st.session_state and st.button("🗑️ Bersihkan Data Demo", width="stretch"):
            del st.session_state["batch_demo_df"]

    up = st.file_uploader("Unggah file data siswa (.csv atau .xlsx)", type=["csv", "xlsx"], key="upload_batch")

    df_up = None
    source_label = ""
    if up is not None:
        try:
            df_up = pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv") else pd.read_excel(up)
            source_label = f"File unggahan: {up.name}"
        except Exception as e:  # noqa: BLE001
            st.error(f"File tidak dapat dibaca: {e}")
    elif "batch_demo_df" in st.session_state:
        df_up = st.session_state["batch_demo_df"].copy()
        source_label = "Data demo (30 sampel acak dari student-mat tanpa kolom G3)"

    if df_up is not None:
        st.caption(f"Sumber aktif: **{source_label}** — {len(df_up)} baris, {df_up.shape[1]} kolom.")
        result = df_up.copy()

        # 1) Regresi batch
        miss_r = [c for c in reg_schema["feature_order"] if c not in df_up.columns]
        if miss_r:
            st.warning(f"Prediksi G3 dilewati karena kolom berikut tidak ditemukan: {', '.join(miss_r)}")
        else:
            result["Prediksi_G3"] = predict_g3(reg_model, clean_batch(df_up, reg_schema)).round(2)
            result["Status_Risiko"] = np.where(result["Prediksi_G3"] < reg_schema["pass_threshold"], "Berisiko", "Aman")

        # 2) Clustering batch (otomatis gunakan Prediksi_G3 jika G3 tidak ada di file upload)
        df_for_clu = df_up.copy()
        used_pred_for_clu = False
        if clu_schema["include_grades"] and TARGET not in df_for_clu.columns and "Prediksi_G3" in result.columns:
            df_for_clu[TARGET] = result["Prediksi_G3"]
            used_pred_for_clu = True

        miss_c = [c for c in clu_schema["feature_order"] if c not in df_for_clu.columns]
        if miss_c:
            st.warning(f"Clustering dilewati karena kolom berikut tidak ditemukan: {', '.join(miss_c)}")
        else:
            labs = predict_cluster(clu_model, clu_schema, clean_batch(df_for_clu, clu_schema))
            result["Cluster"] = labs
            result["Nama_Cluster"] = [clu_schema["cluster_names"].get(int(i), f"Cluster {i}") for i in labs]
            if used_pred_for_clu:
                st.info("Kolom `G3` tidak tersedia pada file; segmentasi cluster otomatis memanfaatkan nilai `Prediksi_G3`.")

        # KPI Ringkasan Batch
        if "Prediksi_G3" in result.columns:
            n_tot = len(result)
            n_risk = int((result["Status_Risiko"] == "Berisiko").sum())
            pct_risk = n_risk / max(n_tot, 1) * 100
            avg_p = float(result["Prediksi_G3"].mean())
            bk1, bk2, bk3, bk4 = st.columns(4, gap="medium")
            bk1.markdown(kpi_card("Total Siswa Diproses", f"{n_tot}", "Baris valid terprediksi", accent="cyan"), unsafe_allow_html=True)
            bk2.markdown(kpi_card("Rata-rata Prediksi G3", f"{avg_p:.2f} / 20", f"Median: {result['Prediksi_G3'].median():.2f}", accent="indigo"), unsafe_allow_html=True)
            bk3.markdown(kpi_card("Siswa Berisiko (<10)", f"{n_risk} siswa", f"<span class='delta-neg'>{pct_risk:.1f}%</span> perlu pendampingan", accent="rose"), unsafe_allow_html=True)
            bk4.markdown(kpi_card("Tingkat Kelulusan Prediksi", f"{100 - pct_risk:.1f}%", f"{n_tot - n_risk} siswa berstatus Aman", accent="emerald"), unsafe_allow_html=True)

            st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
            bc1, bc2 = st.columns(2, gap="large")
            with bc1:
                fig_bh = px.histogram(
                    result, x="Prediksi_G3", color="Status_Risiko", nbins=18,
                    color_discrete_map={"Aman": COLORS["emerald"], "Berisiko": COLORS["rose"]},
                )
                fig_bh.add_vline(x=PASS_THRESHOLD, line_dash="dash", line_color="#F8FAFC")
                style_plotly(fig_bh, "Distribusi Prediksi Nilai Akhir (Batch)", height=310)
                fig_bh.update_xaxes(title="Prediksi G3")
                fig_bh.update_yaxes(title="Jumlah Siswa")
                st.plotly_chart(fig_bh, width="stretch", theme=None)
            with bc2:
                if "Cluster" in result.columns:
                    cl_cnt = result["Cluster"].value_counts().sort_index()
                    fig_bc = go.Figure(go.Bar(
                        x=[f"Cluster {i}" for i in cl_cnt.index], y=cl_cnt.values,
                        marker_color=[CLUSTER_PALETTE[int(i) % len(CLUSTER_PALETTE)] for i in cl_cnt.index],
                        text=cl_cnt.values, textposition="outside",
                    ))
                    style_plotly(fig_bc, "Sebaran Cluster pada Data Batch", height=310, show_legend=False)
                    fig_bc.update_yaxes(title="Jumlah Siswa")
                    st.plotly_chart(fig_bc, width="stretch", theme=None)

        # Filter & Tabel Hasil
        f1, f2 = st.columns([4, 4], gap="medium")
        view_df = result.copy()
        if "Status_Risiko" in view_df.columns:
            with f1:
                filt_risk = st.selectbox("Filter status risiko", ["Semua", "Berisiko", "Aman"], key="batch_f_risk")
                if filt_risk != "Semua":
                    view_df = view_df[view_df["Status_Risiko"] == filt_risk]
        if "Cluster" in view_df.columns:
            with f2:
                cl_opts = ["Semua"] + [f"Cluster {i}" for i in sorted(result["Cluster"].unique())]
                filt_cl = st.selectbox("Filter cluster", cl_opts, key="batch_f_cl")
                if filt_cl != "Semua":
                    view_df = view_df[view_df["Cluster"] == int(filt_cl.split()[-1])]

        st.dataframe(view_df, width="stretch")
        st.download_button(
            "⬇️ Unduh Hasil Prediksi Lengkap (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_siswa.csv",
            mime="text/csv",
        )


# =============================================================================
#  TAB 4 - EVALUASI REGRESI (TERSTRUKTUR DENGAN SUB-TAB)
# =============================================================================
with tab_evr:
    R = RES_R
    rep = R["report"]
    r2_t, mae_t, rmse_t = rep.loc["Test", "R2"], rep.loc["Test", "MAE"], rep.loc["Test", "RMSE"]
    acc2_t = rep.loc["Test", "Akurasi ±2 poin"] * 100
    overfit_ok = R["gap_r2"] <= 0.15

    # KPI Strip Evaluasi Regresi
    ek1, ek2, ek3, ek4 = st.columns(4, gap="medium")
    ek1.markdown(kpi_card("Model Terbaik (CV)", R["schema"]["model_name"], f"{R['n_train']} train / {R['n_test']} test · {R['n_features']} fitur", accent="cyan"), unsafe_allow_html=True)
    ek2.markdown(kpi_card("Test R²", f"{r2_t:.3f}", f"Selisih Train–Test: {R['gap_r2']:+.3f} ({'Stabil' if overfit_ok else 'Waspada Overfit'})", accent="emerald" if overfit_ok else "amber"), unsafe_allow_html=True)
    ek3.markdown(kpi_card("Test RMSE / MAE", f"{rmse_t:.2f} / {mae_t:.2f}", "Kesalahan rata-rata pada skala 0–20", accent="indigo"), unsafe_allow_html=True)
    ek4.markdown(kpi_card("Akurasi ±2 Poin", f"{acc2_t:.0f}%", f"Akurasi ±1 poin: {rep.loc['Test', 'Akurasi ±1 poin']*100:.0f}%", accent="cyan"), unsafe_allow_html=True)

    # Ringkasan Interpretasi
    top3_lbl = [f"{LABELS.get(c, c)} ({c})" for c in R["importance"]["Fitur"].head(3).tolist()]
    ev_notes = [
        f"Model terbaik: <strong>{html.escape(R['schema']['model_name'])}</strong> ({'dengan' if use_prev_grades else 'tanpa'} G1/G2; absences {'dikecualikan' if exclude_absences else 'disertakan'}).",
        f"Menjelaskan <strong>{r2_t:.1%}</strong> variansi nilai akhir pada data uji (R² = {r2_t:.3f}, MAE = {mae_t:.2f}, RMSE = {rmse_t:.2f}).",
        f"Fitur paling berpengaruh: <strong>{html.escape(', '.join(top3_lbl))}</strong>.",
    ]
    if R["mae_zero"] is not None and R["mae_other"] is not None:
        ev_notes.append(
            f"Pemisahan error G3 = 0 vs G3 &gt; 0 (sesuai rekomendasi EDA): "
            f"G3 = 0 (n={R['n_test_zero']}) → MAE <strong>{R['mae_zero']:.2f}</strong>, RMSE <strong>{R['rmse_zero']:.2f}</strong> | "
            f"G3 &gt; 0 (n={R['n_test_other']}) → MAE <strong>{R['mae_other']:.2f}</strong>, RMSE <strong>{R['rmse_other']:.2f}</strong>."
        )
    if "Benchmark G3 = G2 (Test)" in rep.index:
        b_rmse = rep.loc["Benchmark G3 = G2 (Test)", "RMSE"]
        b_mae = rep.loc["Benchmark G3 = G2 (Test)", "MAE"]
        ev_notes.append(f"Pembanding naif <code>G3 = G2</code> pada data uji menghasilkan MAE = {b_mae:.2f} dan RMSE = {b_rmse:.2f}.")
    st.markdown(insight_box("Ringkasan Evaluasi & Interpretasi Regresi", "<ul>" + "".join(f"<li>{x}</li>" for x in ev_notes) + "</ul>", tone="cyan"), unsafe_allow_html=True)

    sub_r1, sub_r2, sub_r3, sub_r4 = st.tabs([
        "📊 Benchmark & Tuning", "🎯 Diagnostik Prediksi", "🚨 Deteksi Risiko (G3 < 10)", "🧠 Interpretasi Fitur"
    ])

    with sub_r1:
        b_sorted = R["bench"].sort_values("CV RMSE", ascending=False)
        bar_cols = [COLORS["emerald"] if m == R["schema"]["model_name"] else COLORS["indigo"] for m in b_sorted["Model"]]
        fig_bench = go.Figure(go.Bar(
            x=b_sorted["CV RMSE"], y=b_sorted["Model"], orientation="h",
            marker_color=bar_cols,
            text=[f"{v:.2f}" for v in b_sorted["CV RMSE"]], textposition="outside",
        ))
        style_plotly(fig_bench, "Perbandingan Algoritma Regresi (CV RMSE lebih kecil = lebih baik)", height=390, show_legend=False)
        fig_bench.update_xaxes(title="CV RMSE")
        st.plotly_chart(fig_bench, width="stretch", theme=None)

        m_col1, m_col2 = st.columns([6, 6], gap="large")
        with m_col1:
            section_heading("Metrik Train, Test & Baseline")
            st.dataframe(rep.round(3), width="stretch")
        with m_col2:
            section_heading(f"Hasil Hyperparameter Tuning ({', '.join(R['tunable'])})")
            st.dataframe(R["tune"].round(3), width="stretch")

        with st.expander("Lihat tabel lengkap hasil Cross-Validation semua algoritma"):
            st.dataframe(R["bench"].round(3), width="stretch")

    with sub_r2:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=R["y_test"], y=R["pred_test"], mode="markers",
                marker=dict(
                    size=9, color=np.abs(R["resid"]), colorscale="Turbo", showscale=True,
                    colorbar=dict(title="|Error|", thickness=12),
                    line=dict(width=0.8, color="#0F172A"),
                ),
                hovertemplate="Aktual: %{x:.1f}<br>Prediksi: %{y:.2f}<extra></extra>",
                name="Data Uji",
            ))
            fig_avp.add_trace(go.Scatter(
                x=[0, 20], y=[0, 20], mode="lines",
                line=dict(color="#94A3B8", dash="dash"), name="Garis Ideal (y=x)",
            ))
            style_plotly(fig_avp, f"Aktual vs Prediksi ({R['schema']['model_name']})", height=350)
            fig_avp.update_xaxes(title="G3 Aktual", range=[-0.5, 20.5])
            fig_avp.update_yaxes(title="G3 Prediksi", range=[-0.5, 20.5])
            st.plotly_chart(fig_avp, width="stretch", theme=None)

        with d2:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=R["pred_test"], y=R["resid"], mode="markers",
                marker=dict(size=9, color=COLORS["cyan"], opacity=0.82, line=dict(width=0.7, color="#0F172A")),
                hovertemplate="Prediksi: %{x:.2f}<br>Residual: %{y:+.2f}<extra></extra>",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color=COLORS["rose"], line_width=2)
            style_plotly(fig_res, "Plot Residual vs Prediksi (pola acak di sekitar 0 = baik)", height=350, show_legend=False)
            fig_res.update_xaxes(title="Prediksi G3")
            fig_res.update_yaxes(title="Residual (Aktual − Prediksi)")
            st.plotly_chart(fig_res, width="stretch", theme=None)

        d3, d4 = st.columns(2, gap="large")
        with d3:
            fig_rh = go.Figure(go.Histogram(
                x=R["resid"], nbinsx=20,
                marker=dict(color=COLORS["emerald"], opacity=0.75, line=dict(color="#0F172A", width=1)),
            ))
            fig_rh.add_vline(x=0, line_dash="dash", line_color="#F8FAFC")
            style_plotly(fig_rh, "Distribusi Residual (Data Uji)", height=330, show_legend=False)
            fig_rh.update_xaxes(title="Residual")
            fig_rh.update_yaxes(title="Frekuensi")
            st.plotly_chart(fig_rh, width="stretch", theme=None)

        with d4:
            if R["lc_data"] is not None:
                lc = R["lc_data"]
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["train_rmse"], mode="lines+markers", name="Train RMSE", line=dict(color=COLORS["cyan"], width=2.5)))
                fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["val_rmse"], mode="lines+markers", name="Validasi RMSE", line=dict(color=COLORS["amber"], width=2.5)))
                style_plotly(fig_lc, "Learning Curve (Mode Lengkap)", height=330)
                fig_lc.update_xaxes(title="Jumlah Data Latih")
                fig_lc.update_yaxes(title="RMSE")
                st.plotly_chart(fig_lc, width="stretch", theme=None)
            else:
                grp_lbl = [f"G3 = 0 (n={R['n_test_zero']})", f"G3 > 0 (n={R['n_test_other']})"]
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(name="MAE", x=grp_lbl, y=[R["mae_zero"] or 0, R["mae_other"] or 0], marker_color=COLORS["cyan"]))
                fig_err.add_trace(go.Bar(name="RMSE", x=grp_lbl, y=[R["rmse_zero"] or 0, R["rmse_other"] or 0], marker_color=COLORS["rose"]))
                fig_err.update_layout(barmode="group")
                style_plotly(fig_err, "Perbandingan Error: Kasus G3 = 0 vs G3 > 0", height=330)
                fig_err.update_yaxes(title="Error (poin)")
                st.plotly_chart(fig_err, width="stretch", theme=None)

    with sub_r3:
        rk = R["risk"]
        rk1, rk2, rk3 = st.columns(3, gap="medium")
        rk1.markdown(kpi_card("Recall Siswa Berisiko", f"{rk['recall']:.0%}", f"{rk['tp']} dari {rk['tp'] + rk['fn']} siswa berisiko terdeteksi", accent="emerald" if rk["recall"] >= 0.75 else "amber"), unsafe_allow_html=True)
        rk2.markdown(kpi_card("Precision Siswa Berisiko", f"{rk['precision']:.0%}", f"Salah alarm (FP): {rk['fp']} siswa", accent="cyan"), unsafe_allow_html=True)
        rk3.markdown(kpi_card("F1-Score Deteksi Risiko", f"{rk['f1']:.2f}", f"Batas risiko: G3 < {PASS_THRESHOLD}", accent="indigo"), unsafe_allow_html=True)

        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)
        cm_z = [[rk["tp"], rk["fn"]], [rk["fp"], rk["tn"]]]
        cm_txt = [
            [f"TP: {rk['tp']}<br>(Terdeteksi benar)", f"FN: {rk['fn']}<br>(Terlewat)"],
            [f"FP: {rk['fp']}<br>(Salah alarm)", f"TN: {rk['tn']}<br>(Benar lulus)"],
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_z,
            x=["Prediksi Berisiko (<10)", "Prediksi Aman (≥10)"],
            y=["Aktual Berisiko (<10)", "Aktual Aman (≥10)"],
            text=cm_txt, texttemplate="%{text}",
            textfont=dict(size=14, color="#F8FAFC"),
            colorscale=[[0, "#1E293B"], [1, "#0284C7"]],
            showscale=False,
        ))
        style_plotly(fig_cm, f"Matriks Konfusi Deteksi Siswa Berisiko (Data Uji, n = {R['n_test']})", height=340, show_legend=False)
        fig_cm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cm, width="stretch", theme=None)

    with sub_r4:
        imp_top = R["importance"].head(12).iloc[::-1]
        i_col1, i_col2 = st.columns([7, 5], gap="large")
        with i_col1:
            fig_imp = go.Figure(go.Bar(
                x=imp_top["Kenaikan RMSE bila diacak"],
                y=[f"{r['Label']} ({r['Fitur']})" for _, r in imp_top.iterrows()],
                orientation="h",
                error_x=dict(type="data", array=imp_top["Std"], color="#94A3B8", thickness=1.5),
                marker_color=COLORS["cyan"],
            ))
            style_plotly(fig_imp, "Permutation Importance — Top 12 Fitur (Data Uji)", height=420, show_legend=False)
            fig_imp.update_xaxes(title="Kenaikan RMSE saat fitur diacak (makin besar = makin penting)")
            st.plotly_chart(fig_imp, width="stretch", theme=None)

        with i_col2:
            section_heading("Tabel Top 10 Fitur")
            st.dataframe(R["importance"][["Label", "Fitur", "Kenaikan RMSE bila diacak", "Std"]].head(10).round(4), width="stretch")
            if R["coef"] is not None:
                section_heading("Koefisien Terbesar (Terstandarisasi)")
                st.dataframe(R["coef"].round(3).rename("Koefisien").to_frame(), width="stretch")

        if R["pdp_data"]:
            section_heading("Partial Dependence Plot — Pengaruh Rata-rata 3 Fitur Numerik Teratas")
            pdp_cols = st.columns(len(R["pdp_data"]), gap="medium")
            for idx, (feat_p, pdata) in enumerate(R["pdp_data"].items()):
                fig_p = go.Figure(go.Scatter(
                    x=pdata["grid"], y=pdata["avg"], mode="lines+markers",
                    line=dict(color=COLORS["indigo"], width=2.8),
                    marker=dict(size=6, color=COLORS["cyan"]),
                    fill="tozeroy", fillcolor="rgba(129, 140, 248, 0.14)",
                ))
                style_plotly(fig_p, f"{LABELS.get(feat_p, feat_p)} ({feat_p})", height=280, show_legend=False)
                fig_p.update_xaxes(title=feat_p)
                fig_p.update_yaxes(title="Rata-rata Prediksi G3")
                pdp_cols[idx].plotly_chart(fig_p, width="stretch", theme=None)


# =============================================================================
#  TAB 5 - EVALUASI CLUSTERING (TERSTRUKTUR DENGAN SUB-TAB)
# =============================================================================
with tab_evc:
    C = RES_C
    ev = C["ev"]
    sil_t, ari_m = float(ev.loc["Test", "silhouette"]), float(C["ari"].mean())
    sil_txt = "Kuat" if sil_t > 0.5 else ("Sedang" if sil_t >= 0.25 else "Lemah (tumpang-tindih)")
    stab_txt = "Sangat stabil" if ari_m > 0.8 else ("Cukup stabil" if ari_m >= 0.6 else "Kurang stabil")

    ck1, ck2, ck3, ck4 = st.columns(4, gap="medium")
    ck1.markdown(kpi_card("Algoritma & k Terpilih", f"{clu_schema['algorithm']} (k={clu_schema['k']})", f"PCA {C['n_pc']} PC ({C['var_pc']:.0%} varians)", accent="indigo"), unsafe_allow_html=True)
    ck2.markdown(kpi_card("Silhouette Test", f"{sil_t:.3f}", f"{sil_txt} (acak: {C['rand_sil']:.3f})", accent="emerald" if sil_t >= 0.25 else "amber"), unsafe_allow_html=True)
    ck3.markdown(kpi_card("Stabilitas Bootstrap ARI", f"{ari_m:.2f}", stab_txt, accent="cyan"), unsafe_allow_html=True)
    ck4.markdown(kpi_card("Kruskal-Wallis G3", f"H = {C['kruskal_H']:.1f}", f"p = {C['kruskal_p']:.1e} ({'Signifikan' if C['kruskal_p'] < 0.05 else 'Tidak signifikan'})", accent="emerald" if C["kruskal_p"] < 0.05 else "amber"), unsafe_allow_html=True)

    cl_lines = [
        f"Data dibagi <strong>{C['n_train']} train / {C['n_test']} test</strong> ({'dengan' if include_grades else 'tanpa'} G1–G3 sebagai fitur).",
        "Acuan silhouette: &gt; 0,5 kuat, 0,25–0,5 sedang, &lt; 0,25 lemah (baca sebagai segmentasi deskriptif).",
    ]
    for c_id in C["prof"].index:
        cl_lines.append(
            f"<strong>{html.escape(clu_schema['cluster_names'][int(c_id)])}</strong> · "
            f"n = {int(C['size'][c_id])}, rata-rata G3 = {C['prof'].loc[c_id, 'G3']:.1f}, "
            f"absen = {C['prof'].loc[c_id, 'absences']:.1f}, failures = {C['prof'].loc[c_id, 'failures']:.2f}"
        )
    st.markdown(insight_box("Ringkasan Evaluasi Clustering", "<ul>" + "".join(f"<li>{x}</li>" for x in cl_lines) + "</ul>", tone="cyan"), unsafe_allow_html=True)

    sub_c1, sub_c2, sub_c3 = st.tabs(["📈 Pemilihan Algoritma & k", "🗺️ Peta PCA 2D & Silhouette", "🔥 Heatmap Profil & Kategori"])

    with sub_c1:
        sel = C["sel"]
        fig_sel = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Silhouette (makin besar makin baik)",
                "Calinski-Harabasz (makin besar makin baik)",
                "Davies-Bouldin (makin kecil makin baik)",
                "Elbow KMeans (Inertia)",
            ),
            vertical_spacing=0.18, horizontal_spacing=0.10,
        )
        algo_colors = {"KMeans": COLORS["cyan"], "GMM": COLORS["emerald"], "Agglomerative": COLORS["amber"]}
        for algo_name, grp in sel.groupby("Algoritma"):
            col_a = algo_colors.get(algo_name, COLORS["indigo"])
            fig_sel.add_trace(go.Scatter(x=grp["k"], y=grp["silhouette"], mode="lines+markers", name=algo_name, line=dict(color=col_a, width=2.5), legendgroup=algo_name), row=1, col=1)
            fig_sel.add_trace(go.Scatter(x=grp["k"], y=grp["calinski"], mode="lines+markers", name=algo_name, line=dict(color=col_a, width=2.5), legendgroup=algo_name, showlegend=False), row=1, col=2)
            fig_sel.add_trace(go.Scatter(x=grp["k"], y=grp["davies"], mode="lines+markers", name=algo_name, line=dict(color=col_a, width=2.5), legendgroup=algo_name, showlegend=False), row=2, col=1)
        km_df = sel[sel["Algoritma"] == "KMeans"]
        fig_sel.add_trace(go.Scatter(x=km_df["k"], y=km_df["inertia"], mode="lines+markers", name="KMeans Inertia", line=dict(color=COLORS["rose"], width=2.5)), row=2, col=2)
        style_plotly(fig_sel, "", height=540)
        st.plotly_chart(fig_sel, width="stretch", theme=None)

        sc_l, sc_r = st.columns([5, 7], gap="large")
        with sc_l:
            section_heading("Metrik Train vs Test")
            st.dataframe(ev.round(3), width="stretch")
        with sc_r:
            section_heading("Top 10 Kandidat Algoritma & k")
            st.dataframe(sel.sort_values("rank_score").head(10)[["Algoritma", "k", "silhouette", "calinski", "davies"]].round(3), width="stretch")

        with st.expander("Pembanding DBSCAN (tidak digunakan untuk deployment karena tanpa .predict())"):
            st.dataframe(C["dbscan"].round(3), width="stretch")

    with sub_c2:
        p_col1, p_col2 = st.columns([7, 5], gap="large")
        with p_col1:
            pca_df = pd.DataFrame({
                "PC1": C["P2"][:, 0],
                "PC2": C["P2"][:, 1],
                "Cluster": [f"Cluster {c}" for c in C["df_all"]["cluster"]],
                "G3": C["df_all"]["G3"].values,
                "G2": C["df_all"]["G2"].values,
                "Failures": C["df_all"]["failures"].values,
                "Absences": C["df_all"]["absences"].values,
            })
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                hover_data=["G3", "G2", "Failures", "Absences"],
                color_discrete_sequence=CLUSTER_PALETTE,
            )
            fig_pca.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="#0B1120")))
            style_plotly(fig_pca, f"Sebaran Cluster ({clu_schema['algorithm']}, k={clu_schema['k']}) pada PCA 2D", height=430)
            fig_pca.update_xaxes(title=f"PC1 ({C['pca2_var'][0]:.1%} varians)")
            fig_pca.update_yaxes(title=f"PC2 ({C['pca2_var'][1]:.1%} varians)")
            st.plotly_chart(fig_pca, width="stretch", theme=None)

        with p_col2:
            fig_sil = go.Figure()
            y_cursor = 0
            for c_id in range(clu_schema["k"]):
                vals = np.sort(C["sample_sil"][C["df_all"]["cluster"].values == c_id])
                y_pos = np.arange(y_cursor, y_cursor + len(vals))
                fig_sil.add_trace(go.Bar(
                    x=vals, y=y_pos, orientation="h",
                    name=f"Cluster {c_id}",
                    marker=dict(color=CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)], line=dict(width=0)),
                    hovertemplate=f"Cluster {c_id}<br>Silhouette: %{{x:.3f}}<extra></extra>",
                ))
                y_cursor += len(vals) + 12
            fig_sil.add_vline(x=float(C["sample_sil"].mean()), line_dash="dash", line_color=COLORS["rose"], line_width=2)
            style_plotly(fig_sil, "Distribusi Silhouette per Observasi", height=430)
            fig_sil.update_xaxes(title="Nilai Silhouette")
            fig_sil.update_yaxes(showticklabels=False, title="Observasi Siswa")
            fig_sil.update_layout(bargap=0)
            st.plotly_chart(fig_sil, width="stretch", theme=None)

    with sub_c3:
        z_df = C["z_prof"].T.copy()
        z_df.index = [f"{LABELS.get(c, c)} ({c})" for c in z_df.index]
        z_df.columns = [f"Cluster {c}" for c in z_df.columns]
        fig_hm = go.Figure(go.Heatmap(
            z=z_df.values,
            x=z_df.columns,
            y=z_df.index,
            colorscale="RdBu_r", zmid=0,
            text=np.round(z_df.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Z-Score", thickness=14),
        ))
        style_plotly(fig_hm, "Deviasi Rata-rata Fitur per Cluster (Z-Score: merah = di atas rata-rata, biru = di bawah)", height=560, show_legend=False)
        fig_hm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_hm, width="stretch", theme=None)

        if len(C["notable"]):
            section_heading("Kategori Menonjol per Cluster (lift ≥ 1,3 dan proporsi ≥ 20%)")
            st.dataframe(C["notable"].round(2), width="stretch")


# =============================================================================
#  TAB 6 - DATA, EDA & UNDUHAN
# =============================================================================
with tab_data:
    n_zero = int((df_raw[TARGET] == 0).sum())
    n_miss = int(df_raw.isna().sum().sum())
    pass_rate_all = float((df_raw[TARGET] >= PASS_THRESHOLD).mean() * 100)

    dk1, dk2, dk3, dk4 = st.columns(4, gap="medium")
    dk1.markdown(kpi_card("Dimensi Dataset", f"{df_raw.shape[0]} × {df_raw.shape[1]}", f"Sumber: {html.escape(raw_name)}", accent="cyan"), unsafe_allow_html=True)
    dk2.markdown(kpi_card("Kualitas Data", f"{n_miss} Missing", "0 duplikasi baris penuh", accent="emerald"), unsafe_allow_html=True)
    dk3.markdown(kpi_card("Rata-rata / Lulus G3", f"{df_raw[TARGET].mean():.2f} · {pass_rate_all:.0f}%", f"Median: {df_raw[TARGET].median():.1f} (skala 0–20)", accent="indigo"), unsafe_allow_html=True)
    dk4.markdown(kpi_card("Kasus G3 = 0", f"{n_zero} siswa", f"{n_zero / len(df_raw):.1%} populasi (dipertahankan)", accent="amber"), unsafe_allow_html=True)

    section_heading("📊 Eksplorasi Data Interaktif (EDA)")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        g3_grp = np.where(df_raw[TARGET] == 0, "G3 = 0 (Putus ujian/nol)", "G3 > 0")
        fig_e1 = px.histogram(
            df_raw.assign(Kelompok=g3_grp), x=TARGET, color="Kelompok", nbins=21,
            color_discrete_map={"G3 = 0 (Putus ujian/nol)": COLORS["rose"], "G3 > 0": COLORS["cyan"]},
        )
        style_plotly(fig_e1, "Distribusi Nilai Akhir (G3) & Lonjakan Nilai Nol", height=340)
        fig_e1.update_xaxes(title="Nilai Akhir (G3)")
        fig_e1.update_yaxes(title="Jumlah Siswa")
        st.plotly_chart(fig_e1, width="stretch", theme=None)

    with e2:
        corr = df_raw.select_dtypes(include=np.number).corr()[TARGET].drop(TARGET).sort_values()
        fig_e2 = go.Figure(go.Bar(
            x=corr.values,
            y=[f"{LABELS.get(c, c)} ({c})" for c in corr.index],
            orientation="h",
            marker_color=[COLORS["emerald"] if v > 0 else COLORS["rose"] for v in corr.values],
            text=[f"{v:+.2f}" for v in corr.values], textposition="outside",
        ))
        style_plotly(fig_e2, "Korelasi Pearson Fitur Numerik terhadap G3", height=340, show_legend=False)
        fig_e2.update_xaxes(title="Koefisien Korelasi (r)", range=[-0.5, 1.05])
        st.plotly_chart(fig_e2, width="stretch", theme=None)

    e3, e4 = st.columns(2, gap="large")
    with e3:
        box_feat = st.selectbox(
            "Eksplorasi G3 berdasarkan fitur:",
            ["failures", "higher", "studytime", "Medu", "schoolsup", "address", "internet", "goout", "sex"],
            format_func=lambda c: f"{LABELS.get(c, c)} ({c})",
            key="eda_box_feat",
        )
        fig_e3 = px.box(
            df_raw, x=box_feat, y=TARGET, color=box_feat,
            color_discrete_sequence=CLUSTER_PALETTE, points="outliers",
        )
        style_plotly(fig_e3, f"Sebaran Nilai Akhir (G3) menurut {LABELS.get(box_feat, box_feat)}", height=330, show_legend=False)
        st.plotly_chart(fig_e3, width="stretch", theme=None)

    with e4:
        st.markdown("<div style='height:4.2rem'></div>", unsafe_allow_html=True)
        fig_e4 = px.scatter(
            df_raw.assign(Status=np.where(df_raw[TARGET] == 0, "G3 = 0", "G3 > 0")),
            x="G2", y=TARGET, color="Status",
            color_discrete_map={"G3 = 0": COLORS["rose"], "G3 > 0": COLORS["cyan"]},
            hover_data=["G1", "failures", "absences"],
        )
        fig_e4.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode="lines", name="G3 = G2", line=dict(color="#94A3B8", dash="dash")))
        style_plotly(fig_e4, "Hubungan G2 vs G3 (Korelasi r = 0,905 & Anomali G3 = 0)", height=330)
        st.plotly_chart(fig_e4, width="stretch", theme=None)

    with st.expander("📝 Catatan Metodologi, Cleaning & Risiko Data (sesuai Brief Proyek)"):
        st.markdown(
            """
            - **Sumber data**: UCI Student Performance (`student-mat`, 395 siswa, 33 kolom, 2 sekolah di Portugal: GP & MS, 2005–2006).
            - **Keutuhan data**: 0 missing value, 0 duplikasi baris. Sebanyak 38 baris `G3 = 0` (9,6%) dan outlier tetap dipertahankan karena valid secara domain.
            - **Risiko informasi masa depan (`absences`)**: Seluruh 38 siswa dengan `G3 = 0` tercatat `absences = 0`, dan waktu pencatatan `absences` tidak terinci. Oleh karena itu, secara default fitur `absences` dikeluarkan dari model regresi (dapat diaktifkan kembali melalui *Pengaturan Model* untuk pembandingan).
            - **Kode ordinal vs besaran asli**: Fitur seperti `studytime`, `traveltime`, `Medu`, `Fedu`, `famrel`, `goout`, `Dalc`, `Walc`, dan `health` adalah kode kategori berurutan sehingga ditampilkan dengan label deskriptif pada antarmuka.
            """
        )

    section_heading("🔍 Pratinjau Dataset Mentah")
    st.dataframe(df_raw.head(100), width="stretch")

    section_heading("⬇️ Pusat Unduhan Model & Artefak")
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        st.download_button("📦 Model Regresi (.joblib)", model_bytes(reg_model), file_name="model_regresi.joblib", width="stretch")
        st.download_button("📄 Skema Regresi (.json)", json.dumps(reg_schema, indent=2).encode("utf-8"), file_name="schema_regresi.json", mime="application/json", width="stretch")
    with d2:
        st.download_button("📦 Model Clustering (.joblib)", model_bytes(clu_model), file_name="model_cluster.joblib", width="stretch")
        st.download_button("📄 Skema Clustering (.json)", json.dumps(clu_schema, indent=2).encode("utf-8"), file_name="schema_cluster.json", mime="application/json", width="stretch")
    with d3:
        st.download_button("📊 Data + Label Cluster (.csv)", RES_C["df_all"].to_csv(index=False).encode("utf-8"), file_name="data_dengan_cluster.csv", mime="text/csv", width="stretch")
        st.caption("File `.joblib` memerlukan versi `scikit-learn` yang kompatibel saat dimuat ulang.")
