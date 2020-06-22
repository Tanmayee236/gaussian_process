import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pytest
import ta
from loguru import logger
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from darwin_ml import __version__
from darwin_ml.technical import (fibonacci, fibonacci_rsi,
                                 super_hyper_mega_average_true_range)
from darwin_ml.technical.momentum import rsi_positions
from darwin_ml.technical.signals import fib_intensity_signal
from darwin_ml.technical.volume import fibonacci_boll_bands
from darwin_ml.utils import Windowing
from darwin_ml.utils.preprocessing import (format_look_ahead,
                                           format_timeseries_dataframe)
from typing import List
from copy import copy, deepcopy
from creme import stats
from creme import datasets
from creme import linear_model
from creme import metrics
from creme import preprocessing


def boolean_flip(item):
    if item == True:
        return 1
    else:
        return -1


def roll_dataframe_stats(frame: pd.DataFrame,
                         window=14,
                         min_steps: int = 1,
                         callback: Optional[Callable] = None,
                         model=None,
                         metric: metrics.ClassificationReport = metrics.ClassificationReport()):
    windower = Windowing(frame,
                         window_size=window,
                         adaptive_window=True,
                         adapted_window_size=0)

    step_count = 0
    history = []

    model_copy = copy(model)

    _mean_down = stats.Mean()
    _mean_up = stats.Mean()
    while windower.has_next_observation:
        res = windower.step()
        x = res.to_dict(orient="record")[0]
        y = x.pop("y")
        if model_copy is not None:
            y_pred = boolean_flip(model.predict_one(x))
            model.fit_one(x, y)
            if y_pred != y:
                prob_up = model.predict_proba_one(x)
                prob_values = list(prob_up.values())
                is_false_pct = _mean_down.update(prob_values[0]).get()
                is_true_pct = _mean_up.update(prob_values[1]).get()
                down_msg = f"Probability going DOWNWARDS for incorrect classifications: {is_false_pct}"
                up_msg = f"Probability going UPWARDS for incorrect classifications: {is_true_pct}"
                logger.error(up_msg)
                logger.warning(down_msg)
            metric.update(y_pred, y)
            mod_acc = metric.accuracy
            logger.debug(f"Overall model accuracy: {mod_acc} \n\n")
        if callback is not None:
            history.append(callback(res))
        step_count += 1

    return step_count >= min_steps, history


def stock_data():
    BASE_DIR = Path(
        __file__).resolve().parent.parent.cwd() / 'data' / 'stock_data.csv'
    BASE_DIR_STR = str(BASE_DIR)
    return pd.read_csv(BASE_DIR_STR)


X_y = datasets.TrumpApproval()


def main():
    df = stock_data()
    df = ta.utils.dropna(df)
    df = format_timeseries_dataframe(df, "Timestamp")
    df = format_look_ahead(df, "Close", size=-4)
    df.dropna()
    df['log_returns'] = 0
    df['log_returns'] = np.where(df["Close_future"] > df["Close"], 1, 1)
    df['log_returns'] = np.where(df["Close_future"] < df["Close"], -1,
                                 df['log_returns'])
    df = fibonacci(df)
    df = fibonacci_rsi(df)
    # df = super_hyper_mega_average_true_range(df)
    df = df.drop(columns=[
        'Open', 'High', 'Low', 'Volume_Currency', 'Weighted_Price',
        'Volume_BTC', 'Close', 'above_below_close', 'Close_future'
    ])
    df = df.rename(columns={"log_returns": "y"})
    model = (preprocessing.MinMaxScaler()
             | linear_model.PAClassifier(C=0.01, mode=1))
    report = metrics.ClassificationReport()

    roll_dataframe_stats(df, model=model, metric=report)
    # print(df[['y']].tail(20))


if __name__ == "__main__":
    main()
