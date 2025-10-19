import streamlit as st, pandas as pd, numpy as np, joblib, plotly.express as px, os

st.set_page_config(page_title="Steam Price Predictor", layout="wide")

# === Carregar modelo ===
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    p1 = os.path.join(base, "steam_price_model.pkl")
    p2 = os.path.join(base, "..", "models", "final_steam_model.pkl")
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        st.error("Modelo não encontrado. Coloque 'steam_price_model.pkl' em /app ou 'final_steam_model.pkl' em /models.")
        st.stop()
    return joblib.load(path)

model = load_model()

# === Layout ===
st.title("Steam Game Price Predictor")
tabs = st.tabs(["Predição", "Exploração", "Sobre"])

# === Aba Predição ===
with tabs[0]:
    st.sidebar.header("Entradas")
    owners = st.sidebar.number_input("Owners (estimativa de jogadores)", 1000, 50_000_000, 500_000, 1000)
    pos = st.sidebar.number_input("Avaliações Positivas", 0, 5_000_000, 1000)
    neg = st.sidebar.number_input("Avaliações Negativas", 0, 5_000_000, 100)
    play = st.sidebar.number_input("Tempo médio (minutos)", 0, 200000, 200)
    year = st.sidebar.slider("Ano de lançamento", 1995, 2025, 2020)
    genre = st.sidebar.selectbox("Gênero", ["Action", "Indie", "Adventure", "Strategy", "RPG", "Simulation"])

    # === Feature engineering idêntico ao treino ===
    positive_ratio = pos / (pos + neg + 1)
    log_playtime = np.log1p(play)

    x = pd.DataFrame([{
        "owners_mean": owners,
        "positive_ratio": positive_ratio,
        "log_playtime": log_playtime,
        "release_year": year,
        "genres": genre
    }])

    # === Previsão ===
    price = float(model.predict(x)[0])
    st.metric("Preço estimado", f"US$ {price:.2f}")

    st.plotly_chart(
        px.bar(
            x=["Preço Previsto"],
            y=[price],
            title="Estimativa de Preço",
            color_discrete_sequence=["#38BDF8"]
        ),
        use_container_width=True
    )

    # === Importância das variáveis ===
    if hasattr(model.named_steps["model"], "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
        feats = model.named_steps["preprocessor"].get_feature_names_out()
        imp = pd.DataFrame({"Feature": feats, "Importância": importances}).sort_values("Importância", ascending=False).head(10)
        st.subheader("Importância das variáveis")
        st.plotly_chart(
            px.bar(imp, x="Importância", y="Feature", orientation="h", color="Importância", color_continuous_scale="blues"),
            use_container_width=True
        )

# === Aba Exploração ===
with tabs[1]:
    st.subheader("Exploração de dados")
    up = st.file_uploader("Carregar steam_clean.csv", type=["csv"])
    if up:
        data = pd.read_csv(up)
        needed = {"price", "owners_mean", "positive_ratio", "log_playtime", "release_year", "genres"}
        if needed.issubset(set(data.columns)):
            st.dataframe(data.sample(min(500, len(data))), use_container_width=True)
            st.plotly_chart(px.scatter(data, x="positive_ratio", y="price", color="genres",
                                       title="Preço vs. proporção de avaliações positivas"), use_container_width=True)
            st.plotly_chart(px.box(data, x="genres", y="price", title="Distribuição de preços por gênero"),
                            use_container_width=True)
        else:
            st.warning("O CSV precisa conter as colunas: " + ", ".join(sorted(needed)))

# === Aba Sobre ===
with tabs[2]:
    st.markdown("""
**Sobre**  
Modelo de regressão baseado em Random Forest, treinado com dados de mais de 24 mil jogos da Steam.  
Fatores considerados: número estimado de donos, proporção de avaliações positivas, tempo médio de jogo, ano de lançamento e gênero.  
O pipeline é o mesmo usado no treinamento, garantindo consistência nas previsões.
""")
