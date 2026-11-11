
import argparse

from src.solar_predictor import DATA_FILE, METRICS_FILE, MODEL_FILE, load_data, train_model


def main():
    parser = argparse.ArgumentParser(description="Train the Solar Supply Predictor model")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to the CSV dataset")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional year to train on; by default all balanced years are used",
    )
    parser.add_argument(
        "--all-years", action="store_true", help="Use all available years instead of 2024"
    )
    parser.add_argument("--tune", action="store_true", help="Run time-series hyperparameter tuning")
    parser.add_argument("--model", default=str(MODEL_FILE), help="Where to save the model")
    parser.add_argument(
        "--metrics", default=str(METRICS_FILE), help="Where to save evaluation metrics"
    )
    args = parser.parse_args()

    data = load_data(args.data)
    training_year = None if args.all_years else args.year
    _, metrics = train_model(
        data, args.model, args.metrics, training_year=training_year, tune=args.tune
    )
    print(f"Rows used: {metrics['rows_used']}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']}")
    print(f"RMSE: {metrics['root_mean_squared_error']}")
    print(f"MAPE: {metrics['mean_absolute_percentage_error']}%")
    print(f"R-squared score: {metrics['r2_score']}")
    print(f"Model saved to: {args.model}")


if __name__ == "__main__":
    main()
