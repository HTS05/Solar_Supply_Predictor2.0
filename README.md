# Solar Supply Predictor2.0

This is the extension of my previous project, this predicts solar irradiance and gives an estimated idea of how much electricity a solar farm can produce.

## My idea

I use historical weather data from six Indian cities to train a Random Forest model. The model learns the relationship between weather conditions and solar irradiance (GHI).

In the dashboard, a user selects a city and a future date. The app gets the weather forecast from the Open-Meteo API. Then the trained model predicts the expected GHI range.

The user also enters solar farm details:

- Daily electricity requirement
- Solar panel area
- Panel efficiency
- System performance

The app uses these details to estimate solar energy generation and shows how much of the electricity requirement may be fulfilled.

## Project flow

```
Historical data -> Clean and merge data -> Train Random Forest model -> Save trained model -> Get future weather from Open-Meteo -> Predict GHI range -> Estimate solar farm electricity
```

## Main files

- `app.py` - Streamlit dashboard & Irradiance to electricity conversion(Logic).
- `train_model.py` - trains and saves the model.
- `src/solar_predictor.py` - machine learning functions.
- `src/weather_api.py` - Open-Meteo forecast API connection.
- `data/raw/` - original data files.
- `data/cleaned/` - cleaned city files and the merged dataset.
- `model/solar_rf_model.joblib` - trained Random Forest model.
- `docs/notebooks/data_preprocessing.ipynb` - notebook explaining the preprocessing.

## Technologies used

- Python
- pandas
- NumPy
- scikit-learn
- Random Forest Regression
- Streamlit
- Open-Meteo API
- joblib
- Jupyter Notebook

## Run the project

Install the required packages

Start the dashboard:

```
streamlit run app.py
```

## Model result

The model is evaluated using chronological validation instead of randomly mixing dates.

The current model achieved approximately:

- MAE: 0.47 kWh/m²/day
- RMSE: 0.62 kWh/m²/day
- R²: 0.777

The dashboard displays a range instead of one exact value because weather and model predictions are uncertain.

## Limitation

The model predicts GHI, not exact electricity generation. The electricity result is an estimate based on the farm details entered by the user. Actual production can change because of shading, panel angle, temperature, inverter losses, battery losses, and other system conditions.
