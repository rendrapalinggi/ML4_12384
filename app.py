# ==============================================================
# Aplikasi Streamlit: Klasifikasi Wine dengan XGBoost
# ==============================================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Konfigurasi Halaman
# -------------------------------------------------------------
st.set_page_config(page_title="Wine Classifier XGBoost", page_icon="🍷", layout="wide")

st.title("🍷 Prediksi Kelas Anggur (Wine Classification)")
st.markdown("""
Aplikasi ini menggunakan **model XGBoost Classifier** yang telah dilatih
menggunakan **2 fitur utama** dari dataset Wine UCI.
Model ini memprediksi apakah sampel anggur termasuk **Kelas 2 (0)** atau **Kelas 3 (1)**
berdasarkan kadar alkohol dan rasio OD280/OD315.
""")

# -------------------------------------------------------------
# Muat Model
# -------------------------------------------------------------
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------------------
# Fitur yang Digunakan
# -------------------------------------------------------------
features = [
    'Alcohol',
    'OD280/OD315 of diluted wines'
]

feature_info = {
    'Alcohol': {
        'min': 11.0,
        'max': 15.0,
        'default': 12.5,
        'help': 'Kadar alkohol dalam anggur (biasanya antara 11–15%)'
    },
    'OD280/OD315 of diluted wines': {
        'min': 1.0,
        'max': 4.0,
        'default': 2.5,
        'help': 'Rasio absorbansi optik OD280/OD315 (ukuran kandungan protein)'
    }
}

# -------------------------------------------------------------
# Input Data Pengguna
# -------------------------------------------------------------
st.subheader("Masukkan Nilai Fitur Anggur")

cols = st.columns(2)
input_data = []

for i, feat in enumerate(features):
    info = feature_info[feat]
    with cols[i % 2]:
        val = st.number_input(
            label=f"**{feat}**",
            min_value=float(info['min']),
            max_value=float(info['max']),
            value=float(info['default']),
            step=0.01,
            format="%.4f",
            help=info['help']
        )
        input_data.append(val)

# -------------------------------------------------------------
# Visualisasi Input (Radar/Bar sederhana)
# -------------------------------------------------------------
with st.expander("📊 Lihat Ringkasan Nilai Input"):
    df_input = pd.DataFrame({
        "Fitur": features,
        "Nilai Input": input_data,
        "Min": [feature_info[f]['min'] for f in features],
        "Max": [feature_info[f]['max'] for f in features],
    })
    st.dataframe(df_input, use_container_width=True)

# -------------------------------------------------------------
# Prediksi
# -------------------------------------------------------------
if st.button("🔍 Prediksi Kelas Anggur"):
    input_array = np.array(input_data).reshape(1, -1)

    # Prediksi kelas dan probabilitas
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    st.write("---")

    # Tampilkan hasil utama
    if prediction == 0:
        st.success("**Hasil Prediksi: Kelas 2 (Label Asli: 2 → Encoded: 0)**")
    else:
        st.error("**Hasil Prediksi: Kelas 3 (Label Asli: 3 → Encoded: 1)**")

    # Tampilkan probabilitas dengan bar chart
    st.markdown("### 📈 Probabilitas Prediksi")

    df_proba = pd.DataFrame({
        "Kelas": ["Kelas 2 (0)", "Kelas 3 (1)"],
        "Probabilitas": proba
    })

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        df_proba["Kelas"],
        df_proba["Probabilitas"],
        color=["#4C72B0", "#C44E52"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probabilitas", fontsize=11)
    ax.set_title("Distribusi Probabilitas Model XGBoost", fontsize=12, fontweight="bold")
    for i, v in enumerate(df_proba["Probabilitas"]):
        ax.text(i, v + 0.03, f"{v:.2%}", ha="center", fontsize=11, fontweight="bold")
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Tampilkan tabel probabilitas
    st.markdown("#### Detail Probabilitas")
    st.dataframe(
        df_proba.style.format({"Probabilitas": "{:.4f}"}),
        use_container_width=True
    )

    st.write("---")

# -------------------------------------------------------------
# Info Tambahan
# -------------------------------------------------------------
with st.expander("ℹ️ Tentang Model & Dataset"):
    st.markdown("""
    - **Model:** XGBoost Classifier  
    - **Dataset:** Wine Dataset dari UCI Machine Learning Repository  
    - **Kelas yang Diprediksi:** Kelas 2 dan Kelas 3 (kelas 1 dihapus → klasifikasi biner)  
    - **Fitur Input:**  
        - `Alcohol` — Kadar alkohol dalam sampel anggur  
        - `OD280/OD315 of diluted wines` — Rasio absorbansi optik (berkorelasi dengan kandungan protein)  
    - **Hyperparameter Model:**  
        - `n_estimators` = 1000  
        - `learning_rate` = 0.01  
        - `max_depth` = 4  
        - `random_state` = 1  
    - **Label Encoding:**  
        - Kelas 2 → `0`  
        - Kelas 3 → `1`  
    """)

with st.expander("📋 Panduan Penggunaan"):
    st.markdown("""
    1. Masukkan nilai **Alcohol** (rentang tipikal: 11.0 – 15.0)
    2. Masukkan nilai **OD280/OD315** (rentang tipikal: 1.0 – 4.0)
    3. Klik tombol **Prediksi Kelas Anggur**
    4. Lihat hasil prediksi dan distribusi probabilitas model
    """)