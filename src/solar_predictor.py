"""Helper functions for the Solar Supply Predictor project."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "cleaned" / "FINAL_DATASET_2022_23_24.csv"
MODEL_FILE = PROJECT_ROOT / "model" / "solar_rf_model.joblib"
METRICS_FILE = PROJECT_ROOT / "model" / "model_metrics.json"
LEGACY_MODEL_FILE = PROJECT_ROOT / "solar_rf_model.pkl"

TARGET = "GHI"
NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "Cloud_Coverage",
    "Wind_Speed",
    "Precipitation",
    "Pressure",
    "Corrected_Wind_Speed",
]
DATE_FEATURES = ["month_sin", "month_cos", "day_sin", "day_cos"]
CITY_FEATURES = [
    "City_Chennai",
    "City_Delhi",
    "City_Guwahati",
    "City_Jaipur",
    "City_Kolkata",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + DATE_FEATURES + CITY_FEATURES
LEGACY_FEATURE_COLUMNS = NUMERIC_FEATURES + CITY_FEATURES


def load_data(path=DATA_FILE, year=None):
    """Load the merged dataset and apply the small cleaning steps it needs."""
    data = pd.read_csv(path)

    required_columns = {"Date", "City", TARGET, *NUMERIC_FEATURES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_columns)}")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in [TARGET, *NUMERIC_FEATURES]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Date", "City", TARGET, *NUMERIC_FEATURES])
    data = data[data[TARGET] >= 0]

    # A few weather rows contain small negative/over-100 values from measurement noise.
    data["Precipitation"] = data["Precipitation"].clip(lower=0)
    data["Cloud_Coverage"] = data["Cloud_Coverage"].clip(lower=0, upper=100)
    data["Wind_Speed"] = data["Wind_Speed"].clip(lower=0)
    data["Corrected_Wind_Speed"] = data["Corrected_Wind_Speed"].clip(lower=0)

    if year is not None:
        data = data[data["Date"].dt.year == int(year)]

    return data.sort_values(["Date", "City"]).reset_index(drop=True)


def prepare_features(data, feature_columns=FEATURE_COLUMNS):
    """Create the same numeric and one-hot encoded feature layout used by the model."""
    data = data.copy()
    dates = pd.to_datetime(data.get("Date", pd.Timestamp.today()), errors="coerce")
    day_of_year = dates.dt.dayofyear.fillna(1)
    month = dates.dt.month.fillna(1)

    features = data[NUMERIC_FEATURES + ["City"]].copy()
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features = pd.get_dummies(features, columns=["City"], dtype=float)
    return features.reindex(columns=feature_columns, fill_value=0).astype(float)


def split_data_by_time(data, test_size=0.2):
    """Split complete dates so future observations never enter the training set."""
    dates = sorted(data["Date"].drop_duplicates())
    if len(dates) < 2:
        raise ValueError("At least two dates are needed for a time-based split")
    split_index = min(len(dates) - 1, max(1, int(len(dates) * (1 - test_size))))
    split_date = dates[split_index]
    train = data[data["Date"] < split_date]
    test = data[data["Date"] >= split_date]
    return train, test


def _model_metrics(actual, predicted):
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    non_zero = actual != 0
    mape = (
        ((actual[non_zero] - predicted[non_zero]).abs() / actual[non_zero]).mean() * 100
        if non_zero.any()
        else 0.0
    )
    return {
        "mean_absolute_error": round(float(mean_absolute_error(actual, predicted)), 4),
        "root_mean_squared_error": round(
            float(mean_squared_error(actual, predicted) ** 0.5), 4
        ),
        "mean_absolute_percentage_error": round(float(mape), 2),
        "r2_score": round(float(r2_score(actual, predicted)), 4),
    }


def tune_random_forest(X_train, y_train):
    """Tune a small Random Forest search using time-series cross-validation."""
    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        parameter_grid,
        n_iter=8,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_params_, round(float(-search.best_score_), 4)


def train_model(
    data=None,
    model_path=MODEL_FILE,
    metrics_path=METRICS_FILE,
    n_estimators=100,
    training_year=None,
    tune=False,
):
    """Train the Random Forest model, save it, and return the model and metrics."""
    if data is None:
        data = load_data()
    if training_year is not None:
        data = data[data["Date"].dt.year == int(training_year)].copy()
    data = data.sort_values(["Date", "City"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows are available for the selected training period")

    X = prepare_features(data)
    y = data[TARGET]
    train_data, test_data = split_data_by_time(data)
    X_train = prepare_features(train_data)
    X_test = prepare_features(test_data)
    y_train = train_data[TARGET]
    y_test = test_data[TARGET]

    model_params = {"n_estimators": n_estimators}
    tuning_mae = None
    if tune:
        model_params, tuning_mae = tune_random_forest(X_train, y_train)

    test_model = RandomForestRegressor(random_state=42, n_jobs=1, **model_params)
    test_model.fit(X_train, y_train)
    predictions = test_model.predict(X_test)
    metrics = _model_metrics(y_test, predictions)
    metrics.update(
        {
            "rows_used": int(len(data)),
            "training_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "training_year": training_year,
            "validation": "chronological 80/20 split",
            "features_used": FEATURE_COLUMNS,
            "best_params": model_params,
        }
    )
    if tuning_mae is not None:
        metrics["cross_validation_mae"] = tuning_mae

    # Train the saved model on all available rows after measuring it on the holdout set.
    model = RandomForestRegressor(random_state=42, n_jobs=1, **model_params)
    model.fit(X, y)

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def load_saved_model(model_path=MODEL_FILE):
    """Load the new model, or fall back to the model committed in the original project."""
    model_path = Path(model_path)
    if model_path.exists():
        return joblib.load(model_path)
    if LEGACY_MODEL_FILE.exists():
        return joblib.load(LEGACY_MODEL_FILE)
    raise FileNotFoundError(
        "No trained model was found. Run `python train_model.py` first."
    )


def predict_ghi(model, values):
    """Predict GHI from one dictionary of weather values and a city."""
    values = dict(values)
    model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    if any(column in DATE_FEATURES for column in model_features) and "Date" not in values:
        values["Date"] = pd.Timestamp.today()
    row = pd.DataFrame([values])
    prediction = model.predict(prepare_features(row, model_features))[0]
    return max(0.0, float(prediction))


def get_city_defaults(data, city, selected_date):
    """Return a matching historical row for convenient dashboard defaults."""
    city_data = data[data["City"] == city]
    if city_data.empty:
        return data[NUMERIC_FEATURES].median().to_dict()

    selected_date = pd.Timestamp(selected_date)
    exact_row = city_data[city_data["Date"] == selected_date]
    if not exact_row.empty:
        return exact_row.iloc[0][NUMERIC_FEATURES].to_dict()

    nearest_index = (city_data["Date"] - selected_date).abs().idxmin()
    return city_data.loc[nearest_index, NUMERIC_FEATURES].to_dict()


def get_feature_importance(model):
    """Return model feature importance values in a dashboard-friendly DataFrame."""
    feature_columns = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return (
        pd.DataFrame(
            {"Feature": feature_columns, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def get_feature_range_warnings(data, values):
    """Report live inputs that fall outside the values seen during training."""
    warnings = []
    for column in NUMERIC_FEATURES:
        value = float(values[column])
        minimum = float(data[column].min())
        maximum = float(data[column].max())
        if value < minimum or value > maximum:
            warnings.append(
                f"{column}={value:.2f} is outside the training range "
                f"({minimum:.2f} to {maximum:.2f})"
            )
    return warnings
