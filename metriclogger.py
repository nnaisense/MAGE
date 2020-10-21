import warnings
from pathlib import Path

import pandas as pd
import torch
import numpy as np

# TODO: Get rid of ex from here


class MetricLogger:
    def __init__(self, ex, neptune_ex):
        self.data = {}  # Dictionary with key=metric_name and value=list of rows
        self.columns = {}  # Dictionary with key=metric_name and value=column names
        self.index = {}  # Dictionary which keeps auto-increments index for some metrics
        self.ex = ex
        self.neptune_ex = neptune_ex

    def add_scalars(self, meta_metric, metric_dict, *, step_i=None, **context_i):
        single_metric = meta_metric is None
        if single_metric:
            assert len(metric_dict) == 1
            meta_metric = list(metric_dict.keys())[0]
        meta_metric = meta_metric.rstrip('/')  # Remove trailing '/' (a common error)

        if step_i is None:
            step_i = self.ex.step_i

        auto_increment = len(context_i) > 0  # extra indices where provided

        # Save data_dict precisely
        data_dict = context_i
        for metric, value in metric_dict.items():
            name = (meta_metric + '/' + metric) if not single_metric else metric
            if isinstance(value, torch.Tensor):  # A common error
                raise TypeError("Convert to numpy first")
            data_dict[name] = value
        data_dict['step_i'] = step_i

        if meta_metric not in self.data:
            self.data[meta_metric] = []
            self.columns[meta_metric] = list(data_dict.keys())
            self.index[meta_metric] = 1

        if list(data_dict.keys()) != self.columns[meta_metric]:
            raise ValueError(
                f"Inconsistent keys for metric {meta_metric}. Got {data_dict.keys()} but previously have had {self.columns[meta_metric]}")
            # TODO: Maybe I could remove this requirement
        self.data[meta_metric].append(list(data_dict.values()))

        if not auto_increment:
            step_index = step_i
        else:
            step_index = self.index[meta_metric]
            self.index[meta_metric] += 1

        for metric, value in metric_dict.items():
            if not single_metric:
                metric = meta_metric + '/' + metric
            if auto_increment:  # for observer, each metric will have _ai suffix
                metric += '_ai'  # ai = auto-increment

            if self.neptune_ex is not None and np.isfinite(value):  # Neptune does not support nans/infs
                self.neptune_ex.log_metric(metric, x=step_index, y=value)

    def add_scalar(self, metric, value, *, step_i=None, **context_i):
        """
        Args:
            metric: name of the metric
            value: metric value
            step_i:
                if None, ex.step_i is used.
            **context_i: other than step_i indices, e.g. update_i=20. If any is provided, then metrics are logged with auto-increment
        """
        return self.add_scalars(meta_metric=None, metric_dict={metric: value}, step_i=step_i, **context_i)

    def pandas(self):
        """ Get dict of DataFrames"""
        dfs = {}
        for keys in self.data.keys():
            df = pd.DataFrame(data=self.data[keys], columns=self.columns[keys])
            dfs[keys] = df
        return dfs

    def save_artifacts(self):
        dump_dir = self.ex.current_run.config.get('dump_dir', None)
        if dump_dir is None:
            return
        for df_name, df in self.pandas().items():
            fn_name = Path(dump_dir, df_name.replace('/', '_')).with_suffix('.feather')  # The quickest format according to some internet sources
            df.to_feather(fn_name)
            if self.neptune_ex is not None:
                self.neptune_ex.log_artifact(fn_name)

    def __getstate__(self):
        state = dict({k: v for k, v in self.__dict__.items() if k not in ['ex', 'neptune_ex']})
        return state

    def __setstate__(self, other):
        self.__dict__ = other
        self.__dict__['neptune_ex'] = None
        self.__dict__['ex'] = None
