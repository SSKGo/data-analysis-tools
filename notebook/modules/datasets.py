import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd


def make_time_series_data(
    n_samples: int = 100,
    n_features: int = 10,
    random_state: Optional[int] = None,
    start: Optional[str] = None,
    freq: str = "min",
    drop_microsecond: bool = True,
    drop_second: bool = True,
) -> pd.DataFrame:
    def num2alpha(num: int) -> str:
        # 65 = A
        if num <= 25:
            return chr(65 + num)
        else:
            return num2alpha(num // 26 - 1) + chr(65 + num % 26)

    random_generator = np.random.RandomState(random_state)
    if start:
        start = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    else:
        start = dt.datetime.now()
    if drop_microsecond:
        start = start.replace(microsecond=0)
    if drop_second:
        start = start.replace(second=0)
    index_time = pd.date_range(start=start, periods=n_samples, freq=freq)
    columns_alphabet = [num2alpha(num) for num in range(n_features)]
    x = random_generator.randn(n_samples, n_features)
    df_x = pd.DataFrame(x, index=index_time, columns=columns_alphabet)
    return df_x


def add_bias_noise(
    y: pd.DataFrame,
    bias: float = 0.0,
    noise: float = 0.0,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> pd.DataFrame:
    random_generator = np.random.RandomState(random_state)
    if replace:
        y_new = y
    else:
        y_new = y.copy()
    y_new += bias
    if noise > 0.0:
        y_new += random_generator.normal(scale=noise, size=y_new.shape)
    return y_new
