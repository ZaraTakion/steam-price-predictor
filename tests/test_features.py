import unittest

import pandas as pd

from app.features import (
    FEATURES,
    current_reference_year,
    make_prediction_row,
    prepare_training_data,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class FeatureTests(unittest.TestCase):
    def test_reference_year_is_used_for_game_age(self):
        raw = pd.DataFrame(
            {
                "owners": ["1000-2000"],
                "positive_ratings": [90],
                "negative_ratings": [10],
                "average_playtime": [60],
                "release_date": ["2020-03-01"],
                "genres": ["Action"],
                "price": [9.99],
            }
        )
        ready = prepare_training_data(raw, reference_year=2026)
        self.assertEqual(ready.iloc[0]["years_since_release"], 6)
        self.assertEqual(ready.iloc[0]["review_total"], 100)

    def test_app_input_matches_training_feature_schema(self):
        row = make_prediction_row(1500, 90, 10, 60, 2020, "Action", 2026)
        self.assertEqual(list(row.columns), FEATURES)
        self.assertEqual(row.iloc[0]["years_since_release"], 6)
        self.assertEqual(row.iloc[0]["review_total"], 100)

    def test_default_reference_year_is_current_calendar_year(self):
        from datetime import date

        self.assertEqual(current_reference_year(), date.today().year)

    def test_prediction_row_runs_through_a_fitted_pipeline(self):
        training = pd.DataFrame(
            {
                "owners_mean": [100, 200, 300, 400],
                "review_ratio": [0.8, 0.7, 0.9, 0.6],
                "review_total": [10, 20, 30, 40],
                "log_playtime": [1.0, 2.0, 3.0, 4.0],
                "years_since_release": [1, 2, 3, 4],
                "genres": ["Action", "Indie", "Action", "RPG"],
            }
        )
        preprocessor = ColumnTransformer(
            [
                ("num", "passthrough", [name for name in FEATURES if name != "genres"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["genres"]),
            ]
        )
        estimator = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
            ]
        ).fit(training, [2.0, 5.0, 8.0, 10.0])
        prediction = estimator.predict(make_prediction_row(250, 20, 3, 90, 2022, "Simulation", 2026))
        self.assertEqual(len(prediction), 1)


if __name__ == "__main__":
    unittest.main()
