from collections import OrderedDict
from typing import Callable, Dict, List, Union

import creme
import numpy as np
import pandas as pd
from creme.base.estimator import Estimator
from creme.stream import iter_pandas
from loguru import logger
from creme import metrics

class Windowing:
    """Roll through a dataframe manually
    """
    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 10,
        adaptive_window: bool = False,
        adapted_window_size: int = 1,
    ):
        self._current_step: int = 0
        self._window_size: int = window_size
        self._lower_range: int = 0
        self._upper_range: int = len(frame)
        self._frame: pd.DataFrame = frame

        self._is_adapted = adaptive_window
        self._adapted_window_size = adapted_window_size

    @property
    def columns(self) -> list:
        """ Get Columns
            ---
            Return a list of columns from the dataframe.
        """
        return list(self.frame.columns)

    @property
    def window(self):
        """Get Window Size

        Returns:
            int -- The window size
        """
        if self.is_adaptive and self._current_step > self._window_size:
            return self.adapted_window
        return self._window_size

    @property
    def adapted_window(self) -> int:
        """Get the adaptive window size

        Returns:
            int -- Get window size
        """
        return self._adapted_window_size

    @property
    def is_adaptive(self) -> bool:
        return self._is_adapted

    @property
    def frame(self) -> pd.DataFrame:
        """Get the dataframe we're iterating through.

        Returns:
            pd.DataFrame -- The current dataframe
        """
        return self._frame

    @property
    def lower_bounds(self):
        return max((self._current_step - self.window, 0))

    @property
    def upper_bounds(self):
        return min(self._current_step + 1, len(self.frame))

    @property
    def has_next_observation(self) -> bool:
        """Returns if the dataframe has another observation

        Returns
        -------
        bool
            True if there is another observation.
        """
        return self._current_step < len(self.frame) - self.window - 1

    @property
    def next_observation(self) -> Union[np.ndarray, pd.DataFrame]:
        obs = self.frame[self.lower_bounds:self.upper_bounds]
        return obs

    def step(self):
        if self.has_next_observation:
            self._current_step += 1
            return self.next_observation
        raise IndexError("Why are you on the wrong index?")

    def reset(self):
        self._current_step = 0


def roll_dataframe_stats(frame: pd.DataFrame,
                         window=14,
                         min_steps: int = 1,
                         callback: Optional[Callable] = None,
                         model=None,
                         metric:metrics.ClassificationReport  = None):
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
            y_pred = booleanify(model.predict_one(x))
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


if __name__ == "__main__":
    roll_dataframe_stats()


# class AdaptiveCremeTraining(Windowing):
#     def __init__(
#         self,
#         model: Union[Estimator, OrderedDict],
#         frame: pd.DataFrame,
#         window_size: int = 10,
#         adaptive_window: bool = False,
#         adapted_window_size: int = 1,
#     ):
#         super().__init__(frame=frame,
#                          window_size=window_size,
#                          adaptive_window=adaptive_window,
#                          adapted_window_size=adapted_window_size)
#         """Adaptively train on a dataframe. 

#         Will Stream through all elements in a dataframe if it's greater than 1. Otherwise, it'll train on the single item.
#         The intention is to use this as a prototypical solution to training a single asset.

#         Arguments:
#             model {Union[Estimator, OrderedDict]} -- The model or pipeline we intend to use.
#         """
#         self.model: Union[Estimator, OrderedDict] = model
#         self.metric_set = []

#     def _is_valid_frame(self, frame: pd.DataFrame):
#         if len(frame) > 0: return True
#         return False
    
#     def is_model(self):
#         return self.model is not None

#     def add_metric(self, metric):
#         self.metric_set.append(metric)

#     def step(self, frame: pd.DataFrame, target_column: Union[str, List[str]]):
#         if not self._is_valid_frame(frame):
#             return np.ndarray([[]])

#         logger.debug(frame)
