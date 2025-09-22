import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

st.title("Application d'analyse de données") 

import os
import streamlit as st
import base64

# Récupère le dossier du script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construit les chemins absolus vers les images
image_stats_path = os.path.join(script_dir, "les_stats.jpeg")
image_moi_path = os.path.join(script_dir, "Mohamed_ISSOUMEÏLA.png")

# Convertir en URL base64 pour injecter dans le CSS
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Image de fond
bg_img = get_base64_of_bin_file(image_stats_path)
# Photo personnelle
moi_img = get_base64_of_bin_file(image_moi_path)

# Injecter CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f0ff 0%, #1a4d8f 100%);
    }
    /* Image personnelle collée en haut à droite, repoussée vers le bas */
    .moi-photo {
        position: fixed;
        top: 40px;
        right: 0px;
        width: 120px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ...existing code...

# Afficher la photo perso avec balise HTML

# Afficher la photo perso avec un léger espace en haut
st.markdown(
    f'''<div style="position:fixed; top:40px; right:0px; width:140px; z-index:9999; text-align:center;">
        <img src="data:image/png;base64,{moi_img}" style="width:120px; border-radius:15px; box-shadow:0px 4px 10px rgba(0,0,0,0.5);" />
        <div style="margin-top:18px;">
            <span style="font-weight:bold; color:#e85d04; font-size:19px; text-shadow:1px 1px 6px #fff;">Mohamed ISSOUMEÏLA</span><br>
            <span style="font-style:italic; color:#0077b6; font-size:15px; text-shadow:1px 1px 6px #fff;">Économiste Statisticien<br>et Ingénieur Démographe</span>
        </div>
    </div>''',
    unsafe_allow_html=True
)

# Carte centrale d'accueil avant le téléversement
st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; height: 40vh;'>
        <div style='background: rgba(26,77,143,0.12); border-radius: 20px; box-shadow: 0 4px 24px rgba(26,77,143,0.18); padding: 40px 60px; text-align: center;'>
            <img src='https://img.icons8.com/fluency/96/database.png' width='80' style='margin-bottom: 20px;'/>
            <h2 style='color: #1a4d8f; margin-bottom: 10px;'>Bienvenue sur l'application d'analyse de données</h2>
            <p style='font-size: 18px; color: #333; margin-bottom: 20px;'>Avant de commencer, téléversez votre base de données CSV.<br>Votre analyse commence ici !</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Affiche le nom et le titre en dehors de la carte et au-dessus du bouton


# 1. Téléversement de la base de données (bouton stylisé)
uploaded_file = st.file_uploader(
    "Téléchargez votre fichier CSV",
    type="csv",
    label_visibility="visible"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1", sep=";")
    st.write("Aperçu des données :", df.head())

    # 2. Analyse descriptive
    st.subheader("Analyse descriptive")
    st.write("Variables :", df.columns.tolist())

    selected_var = st.selectbox("Choisissez une variable à explorer", df.columns)

    if pd.api.types.is_numeric_dtype(df[selected_var]):
        if st.button("Afficher les statistiques numériques"):
            st.write({
                "Moyenne": df[selected_var].mean(),
                "Médiane": df[selected_var].median(),
                "Écart-type": df[selected_var].std(),
                "Min": df[selected_var].min(),
                "Max": df[selected_var].max(),
                "Valeurs manquantes": df[selected_var].isnull().sum()
            })
    else:
        if st.button("Afficher les fréquences"):
            st.write("Fréquences :")
            st.write(df[selected_var].value_counts())
            st.write("Valeurs manquantes :", df[selected_var].isnull().sum())

    # 3. Analyse bivariée
    st.subheader("Analyse bivariée")
    col1 = st.selectbox("Variable 1", df.columns)
    col2 = st.selectbox("Variable 2", [c for c in df.columns if c != col1])
    if st.button("Afficher l'analyse bivariée"):
        st.write("Tableau croisé :")
        st.write(pd.crosstab(df[col1], df[col2]))
        st.write("Graphique en barres :")
        st.bar_chart(pd.crosstab(df[col1], df[col2]))

    # 4. Analyse multivariée et explicative (régression logistique)
    st.subheader("Régression logistique")
    target = st.selectbox("Choisissez la variable cible", df.columns)
    features = st.multiselect("Choisissez les variables explicatives", [col for col in df.columns if col != target])
    # ...existing code...
    if st.button("Lancer la régression"):
        if len(features) == 0:
            st.error("Veuillez sélectionner au moins une variable explicative.")
        else:
            X = df[features]
            y = df[target]
            # Encodage des variables qualitatives
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype == 'O':
                y = pd.factorize(y)[0]
            n_classes = len(np.unique(y))
            if n_classes < 2:
                st.error("La variable cible doit avoir au moins 2 modalités.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                if n_classes == 2:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.success("Régression logistique binomiale effectuée.")
                else:
                    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.success("Régression logistique multinomiale effectuée.")

                st.text_area("Rapport de classification :", classification_report(y_test, y_pred))

                # 5. Recommandations simples
                st.subheader("Recommandations")
                if n_classes == 2:
                    coef_df = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_[0]})
                else:
                    coef_df = pd.DataFrame(model.coef_, columns=X.columns)
                    coef_df['Classe'] = model.classes_
                st.write(coef_df)
                st.write("Les variables avec des coefficients élevés influencent le plus la variable cible.")
    # ...existing code...
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # ...après la régression logistique binomiale...
    if 'n_classes' in locals() and n_classes == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)

