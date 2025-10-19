import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
np.random.seed(42)

# === Caminhos ===
BASE = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
DATA = os.path.join(BASE, "data", "steam.csv")
MODEL_OUT = os.path.join(BASE, "models", "final_steam_model.pkl")

print("Carregando dataset...")
df = pd.read_csv(DATA, low_memory=False)

# === Criação de features ===
print("Gerando features...")

# 1. Donos médios (faixa -> média numérica)
def parse_owners(x):
    try:
        a, b = x.split('-')
        return (int(a) + int(b)) / 2
    except:
        return np.nan

df["owners_mean"] = df["owners"].apply(parse_owners) if "owners" in df.columns else np.nan

# 2. Log do tempo médio jogado
if "average_playtime" in df.columns:
    df["log_playtime"] = np.log1p(df["average_playtime"])
elif "average_playtime_forever" in df.columns:
    df["log_playtime"] = np.log1p(df["average_playtime_forever"])
else:
    df["log_playtime"] = 0

# 3. Avaliações e data de lançamento
df["review_total"] = df["positive_ratings"] + df["negative_ratings"]
df["review_ratio"] = df["positive_ratings"] / (df["review_total"] + 1)

# Corrige extração do ano
if "release_year" in df.columns:
    df["years_since_release"] = 2025 - df["release_year"]
elif "release_date" in df.columns:
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["years_since_release"] = 2025 - df["release_year"]
else:
    df["years_since_release"] = np.nan

# === Seleção de colunas ===
features = [
    "owners_mean",
    "review_ratio",
    "review_total",
    "log_playtime",
    "years_since_release",
    "genres"
]

dfm = df[features + ["price"]].dropna()
print(f"Dataset pronto com {dfm.shape[0]} linhas e {dfm.shape[1]} colunas.")

# === Split ===
X = dfm.drop("price", axis=1)
y = dfm["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Pré-processamento ===
num_features = ["owners_mean", "review_ratio", "review_total", "log_playtime", "years_since_release"]
cat_features = ["genres"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# === Modelo e tuning ===
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
params = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(pipe, params, n_iter=10, cv=cv, scoring="r2", random_state=42, n_jobs=-1)

print("Treinando modelo...")
search.fit(X_train, y_train)

# === Avaliação ===
y_pred = search.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nResultados:")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# === Salvar modelo ===
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
joblib.dump(search.best_estimator_, MODEL_OUT)
print(f"Modelo salvo em: {MODEL_OUT}")
