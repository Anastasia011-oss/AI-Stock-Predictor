import os
import pandas as pd
import yfinance as yf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)


def download_stock_data(ticker, period="2y"):

    """
    Download historical stock data from Yahoo Finance.
    """

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:

        raise ValueError(
            "No stock data received. Check internet connection or ticker."
        )

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    data = data[required_columns].copy()

    data.index = pd.to_datetime(data.index)

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(keep="last")
    ]

    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    data = data.dropna(
        subset=required_columns
    )

    if len(data) < 100:

        raise ValueError(
            "Not enough historical data."
        )

    file_path = os.path.join(
        DATA_DIR,
        ticker + "_history.csv"
    )

    data.to_csv(
        file_path,
        index_label="Date"
    )

    update_actual_prices(
        ticker,
        data
    )

    return data


def update_actual_prices(ticker, history):

    """
    Update actual prices and errors for old forecasts.
    """

    file_path = os.path.join(
        BASE_DIR,
        "stock_predictions.csv"
    )

    if not os.path.exists(file_path):
        return

    predictions = pd.read_csv(file_path)

    if predictions.empty:
        return

    ticker_rows = (
        predictions["Компанія"] == ticker
    )

    for index in predictions[ticker_rows].index:

        date = pd.to_datetime(
            predictions.loc[index, "Дата прогнозу"]
        )

        if date in history.index:

            actual = float(
                history.loc[date, "Close"]
            )

            predicted = float(
                predictions.loc[index, "Прогнозована ціна"]
            )

            predictions.loc[
                index,
                "Фактична ціна"
            ] = actual

            predictions.loc[
                index,
                "Похибка"
            ] = abs(actual - predicted)

    predictions.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )
