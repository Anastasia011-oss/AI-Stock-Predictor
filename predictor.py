import os
import time

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "stock_predictions.csv"
)

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)


MODEL_IDS = {

    "Chronos-T5 Small":
        "amazon/chronos-t5-small",

    "Chronos-Bolt Small":
        "amazon/chronos-bolt-small"

}


MODEL_CACHE = {}

MODEL_LOAD_TIMES = {}


def get_model(model_name):

    """
    Load a pretrained model from Hugging Face.
    """

    if model_name not in MODEL_CACHE:

        start_time = time.perf_counter()

        model_id = MODEL_IDS[model_name]

        pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32
        )

        MODEL_CACHE[model_name] = pipeline

        MODEL_LOAD_TIMES[model_name] = (
            time.perf_counter() - start_time
        )

    return MODEL_CACHE[model_name]


def predict_values(values, horizon, model_name):

    """
    Generate a forecast using a selected Chronos model.
    """

    pipeline = get_model(model_name)

    context = torch.tensor(
        np.asarray(values, dtype=np.float32),
        dtype=torch.float32
    )

    start_time = time.perf_counter()

    with torch.no_grad():

        forecast = pipeline.predict(
            context,
            prediction_length=int(horizon)
        )

    inference_time = (
        time.perf_counter() - start_time
    )

    forecast = forecast[0].detach().cpu().numpy()

    forecast_type = pipeline.forecast_type.value

    if forecast_type == "samples":


        predicted = np.median(
            forecast,
            axis=0
        )

    else:

        quantiles = np.asarray(
            pipeline.quantiles,
            dtype=float
        )

        median_index = int(
            np.argmin(
                np.abs(quantiles - 0.5)
            )
        )

        predicted = forecast[median_index]

    return predicted, inference_time


def forecast_prices(
    close_series,
    horizon,
    model_name
):

    """
    Forecast future closing prices.
    """

    series = pd.Series(
        close_series
    ).dropna().astype(float)

    if len(series) < 30:

        raise ValueError(
            "At least 30 historical values are required."
        )

    if model_name not in MODEL_IDS:

        raise ValueError(
            "Unknown forecasting model."
        )

    values = series.values

    predicted, inference_time = predict_values(
        values,
        horizon,
        model_name
    )

    last_date = pd.Timestamp(
        series.index[-1]
    )

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon
    )

    result = pd.Series(
        predicted,
        index=future_dates,
        name="Forecast"
    )

    return result


def backtest_model(
    close_series,
    model_name,
    test_size=10
):

    """
    Test the model using the last known observations.
    """

    series = pd.Series(
        close_series
    ).dropna().astype(float)

    if len(series) <= test_size + 30:

        raise ValueError(
            "Not enough data for backtesting."
        )


    train = series.iloc[:-test_size]

    actual = series.iloc[-test_size:].values


    get_model(model_name)

    predicted, inference_time = predict_values(
        train.values,
        test_size,
        model_name
    )

    predicted = np.asarray(
        predicted
    ).reshape(-1)

    if len(predicted) != len(actual):

        raise ValueError(
            "Forecast length does not match test data."
        )

    # Calculate MAE.

    mae = mean_absolute_error(
        actual,
        predicted
    )

    # Calculate RMSE.

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # Calculate MAPE.

    nonzero = actual != 0

    if np.any(nonzero):

        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100

    else:

        mape = np.nan

    return {

        "Компанія": "",

        "Модель": model_name,

        "MAE": round(
            float(mae),
            4
        ),

        "RMSE": round(
            float(rmse),
            4
        ),

        "MAPE, %": round(
            float(mape),
            4
        ),

        "Час прогнозу, сек": round(
            float(inference_time),
            2
        ),

        "Час завантаження, сек": round(
            float(MODEL_LOAD_TIMES.get(model_name, 0)),
            2
        )

    }


def compare_models(
    close_series,
    ticker
):

    """
    Compare two AI models for one company.
    """

    results = []

    for model_name in MODEL_IDS:

        result = backtest_model(
            close_series,
            model_name,
            test_size=10
        )

        result["Компанія"] = ticker

        results.append(result)

    comparison = pd.DataFrame(
        results
    )

    if os.path.exists(COMPARISON_FILE):

        old_results = pd.read_csv(
            COMPARISON_FILE
        )

        saved_results = pd.concat(
            [old_results, comparison],
            ignore_index=True
        )

    else:

        saved_results = comparison

    saved_results.to_csv(
        COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return comparison


def save_forecast(
    ticker,
    last_price,
    forecast,
    model_name
):

    """
    Save forecast results to stock_predictions.csv.
    """

    rows = []

    prediction_date = pd.Timestamp.now()

    for date, predicted_price in forecast.items():

        rows.append({

            "Компанія": ticker,

            "Модель": model_name,

            "Дата прогнозування":
                prediction_date.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "Остання фактична ціна":
                round(float(last_price), 4),

            "Дата прогнозу":
                date.strftime("%Y-%m-%d"),

            "Прогнозована ціна":
                round(float(predicted_price), 4),

            "Фактична ціна": "",

            "Похибка": ""

        })

    new_data = pd.DataFrame(
        rows
    )

    if os.path.exists(PREDICTIONS_FILE):

        old_data = pd.read_csv(
            PREDICTIONS_FILE
        )

        all_data = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

    else:

        all_data = new_data

    all_data.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig"
    )
