"""
App Streamlit - Prédiction de l'état d'un téléphone (Telephone_data)

Fichiers attendus dans le MÊME dossier que ce script :
    - xgb_model.json   (le modèle XGBoost, au format natif JSON)
    - scaler.joblib
    - encoders.joblib
    - uniques.joblib

Ce sont tes fichiers d'origine (encoders__1_.joblib, scaler__1_.joblib,
uniques__1_.joblib -> renommés sans le "__1_"). Le modèle xgb_model__1_.joblib
a été reconverti en xgb_model.json (format natif XGBoost, plus fiable entre
systèmes que le pickle/joblib).

Lancer en local :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
import os

st.set_page_config(page_title="Prédiction état d'un téléphone", page_icon="📱")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    # format natif XGBoost (JSON) : portable entre systèmes, contrairement au pickle/joblib
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    scaler = jb.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = jb.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    uniques = jb.load(os.path.join(MODEL_DIR, "uniques.joblib"))
    return model, scaler, encoders, uniques


try:
    model, scaler, encoders, uniques = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Fichiers .joblib introuvables dans le dossier de l'app. "
        "Copie xgb_model.json, scaler.joblib, encoders.joblib et uniques.joblib "
        "à côté de app.py, puis relance.\n\nDétail : " + str(e)
    )
    st.stop()

# uniques / encoders : [Adresse, Marque, Etat]
adresses = uniques[0]
marques = uniques[1]
class_names = uniques[2]  # ["D'occasion", "Neuf", "Réconditionné", "Venant"]

st.title("📱 Prédire l'état d'un téléphone (occasion / neuf / reconditionné / venant)")
st.caption("Modèle entrainé sur des annonces de téléphones (coinafrique.com).")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction par fichier CSV"])

FEATURE_ORDER = ["prix", "adresse", "marque", "dim_ecr", "ram", "stockage"]

# ---------------------------------------------------------------------------
# Prédiction simple
# ---------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        col1, col2 = st.columns(2)
        with col1:
            prix = st.number_input("Prix (FCFA)", min_value=0, value=200_000, step=5_000)
            adresse = st.selectbox("Adresse / Quartier", adresses)
            marque = st.selectbox("Marque", marques)
        with col2:
            dim_ecr = st.number_input("Dimension de l'écran (pouces)", min_value=3.0, max_value=15.0,
                                       value=6.0, step=0.1)
            ram = st.number_input("RAM (Go)", min_value=1, max_value=256, value=6, step=1)
            stockage = st.number_input("Stockage (Go)", min_value=4, max_value=1024, value=128, step=4)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        adresse_e = encoders[0].transform([adresse])[0]
        marque_e = encoders[1].transform([marque])[0]

        # même ordre que dans le notebook : prix, adresse, marque, dim_ecr, ram, stockage
        x_new = np.array([[prix, adresse_e, marque_e, dim_ecr, ram, stockage]])
        x_new = scaler.transform(x_new)

        y_pred = model.predict(x_new)[0]
        proba = model.predict_proba(x_new)[0] if hasattr(model, "predict_proba") else None

        st.success(f"Prédiction : **{class_names[y_pred]}**")
        if proba is not None:
            st.write(pd.DataFrame({"classe": class_names, "probabilité": proba}).set_index("classe"))

# ---------------------------------------------------------------------------
# Prédiction par lot (CSV)
# ---------------------------------------------------------------------------
with tab2:
    st.write("Le fichier doit contenir les colonnes suivantes, dans cet ordre : `" + ", ".join(FEATURE_ORDER) + "`.")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        try:
            df_enc = df_in.copy()
            df_enc["adresse"] = encoders[0].transform(df_enc["adresse"])
            df_enc["marque"] = encoders[1].transform(df_enc["marque"])

            x_batch = df_enc[FEATURE_ORDER].values
            x_batch = scaler.transform(x_batch)
            preds = model.predict(x_batch)

            df_out = df_in.copy()
            df_out["etat_predit"] = [class_names[p] for p in preds]
            st.dataframe(df_out)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name="predictions_telephone_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur pendant le traitement du fichier : {e}")
