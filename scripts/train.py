"""Train and evaluate the Steam game price model."""

from __future__ import annotations

import argparse
from pathlib import Path
from math import sqrt
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/steam.csv"
MODEL_PATH = ROOT / "models/final_steam_model.pkl"
sys.path.insert(0, str(ROOT))

from app.features import FEATURES, NUMERIC_FEATURES, TARGET, current_reference_year, prepare_training_data


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH, reference_year: int | None = None):
    if reference_year is None:
        reference_year = current_reference_year()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    print(f"Carregando {data_path}...")
    raw = pd.read_csv(data_path, low_memory=False)
    data = prepare_training_data(raw, reference_year=reference_year)
    print(f"Dataset preparado: {len(data):,} jogos; ano de referência: {reference_year}.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["genres"]),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    param_distributions = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )
    print("Executando busca de hiperparâmetros (10 combinações, 5-fold CV)...")
    search.fit(X_train, y_train)

    predictions = search.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print(f"Melhores parâmetros: {search.best_params_}")
    print(f"R² no conjunto de teste: {r2:.3f}")
    print(f"RMSE no conjunto de teste: {rmse:.2f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": search.best_estimator_, "reference_year": reference_year},
        model_path,
    )
    print(f"Modelo salvo em: {model_path}")
    return search.best_estimator_, {"r2": r2, "rmse": rmse, "reference_year": reference_year}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-year",
        type=int,
        default=current_reference_year(),
        help="Ano usado para calcular a idade dos jogos (padrão: ano atual).",
    )
    args = parser.parse_args()
    train(reference_year=args.reference_year)


if __name__ == "__main__":
    main()
