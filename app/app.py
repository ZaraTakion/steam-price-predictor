from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.features import FEATURES, TARGET, current_reference_year, make_prediction_row, prepare_training_data

MODEL_PATH = ROOT / "models/final_steam_model.pkl"

st.set_page_config(page_title="Steam Price Predictor", page_icon="🎮", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None, None
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict) and "estimator" in saved:
        return saved["estimator"], int(saved["reference_year"])
    # Old artifacts have no reference-year metadata; retrain to avoid silent
    # feature drift and ensure the saved model matches the current pipeline.
    return None, None


model, reference_year = load_model()
st.title("Steam Price Predictor")
st.caption("Estimativa experimental baseada em dados históricos. Não representa o preço atual da loja Steam.")

if model is None:
    st.warning("Modelo ausente ou antigo. No terminal, na pasta do projeto, treine novamente:")
    st.code("python -m scripts.train --reference-year 2026")
    st.stop()

st.caption(f"Modelo treinado com ano de referência {reference_year}.")
prediction_tab, explore_tab, about_tab = st.tabs(["Prever preço", "Explorar dados", "Sobre o modelo"])

with prediction_tab:
    st.sidebar.header("Características do jogo")
    owners = st.sidebar.number_input("Estimativa de proprietários", min_value=0, max_value=100_000_000, value=500_000, step=10_000)
    positive = st.sidebar.number_input("Avaliações positivas", min_value=0, max_value=10_000_000, value=1_000, step=100)
    negative = st.sidebar.number_input("Avaliações negativas", min_value=0, max_value=10_000_000, value=100, step=10)
    playtime = st.sidebar.number_input("Tempo médio jogado (minutos)", min_value=0, max_value=2_000_000, value=200, step=30)
    release_year = st.sidebar.slider("Ano de lançamento", min_value=1995, max_value=reference_year, value=min(2020, reference_year))
    genre = st.sidebar.selectbox("Gênero", ["Action", "Indie", "Adventure", "Strategy", "RPG", "Simulation"])

    input_row = make_prediction_row(
        owners_mean=owners,
        positive_ratings=positive,
        negative_ratings=negative,
        average_playtime=playtime,
        release_year=release_year,
        genre=genre,
        reference_year=reference_year,
    )
    assert list(input_row.columns) == FEATURES
    prediction = float(model.predict(input_row)[0])
    st.metric("Preço estimado", f"US$ {max(0, prediction):.2f}")
    st.caption("O resultado depende do recorte, das variáveis disponíveis e do desempenho observado no teste.")

with explore_tab:
    st.subheader("Exploração do dataset")
    upload = st.file_uploader("Envie o CSV de jogos (steam.csv ou steam_clean.csv)", type=["csv"])
    if upload is not None:
        try:
            raw = pd.read_csv(upload, low_memory=False)
            data = prepare_training_data(raw, reference_year=reference_year)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Não consegui preparar esse arquivo: {exc}")
        else:
            st.dataframe(data.sample(min(500, len(data)), random_state=42), use_container_width=True)
            st.plotly_chart(
                px.scatter(data, x="review_ratio", y=TARGET, color="genres", title="Preço e proporção de avaliações positivas"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.box(data, x="genres", y=TARGET, title="Distribuição de preços por gênero"),
                use_container_width=True,
            )

with about_tab:
    st.markdown(
        """
        O projeto treina um `RandomForestRegressor` com `RandomizedSearchCV`,
        valida as combinações por validação cruzada e reporta R² e RMSE em um
        conjunto de teste separado. A previsão é uma demonstração educacional,
        não uma cotação da Steam.

        **Variáveis:** estimativa de proprietários, proporção e total de avaliações,
        tempo médio jogado transformado por log, idade do jogo e gênero.
        """
    )
