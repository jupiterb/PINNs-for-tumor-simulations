import torch as th
from torch.utils.data import Dataset
from typing import Sequence

from phynn.data.series.interface import TimeSeriesDataInterface
from phynn.physics import ParametrizedEquation, EquationSimulation


TimeSeriesDataSample = tuple[
    th.Tensor,  # start
    th.Tensor,  # result
    th.Tensor,  # duration
]


class TimeSeriesDataset(Dataset):
    def __init__(self, data: TimeSeriesDataInterface) -> None:
        super().__init__()
        self._data = data

        series_length = data.series_length
        self._series_samples = (series_length - 1) * series_length // 2

        start_ixs = []
        result_ixs = []

        for i in range(series_length - 1):
            for j in range(i, series_length):
                start_ixs.append(i)
                result_ixs.append(j)

        self._start_ixs = th.Tensor(start_ixs).type(th.int)
        self._result_ixs = th.Tensor(result_ixs).type(th.int)

    def __len__(self) -> int:
        return self._data.series_number * self._series_samples

    def __getitem__(self, index: int) -> TimeSeriesDataSample:
        start, result, duration = self.__getitems__([index])
        return start[0], result[0], duration[0]

    def __getitems__(self, indices: Sequence[int]) -> TimeSeriesDataSample:
        tensor_indices: th.Tensor = th.Tensor(indices)
        series_ixs = tensor_indices // self._series_samples
        sample_ixs = tensor_indices % self._series_samples
        start_ixs = self._start_ixs[sample_ixs]
        result_ixs = self._result_ixs[sample_ixs]

        start, start_time = self._data.get_batch(series_ixs, start_ixs)
        result, result_time = self._data.get_batch(series_ixs, result_ixs)

        return start, result, result_time - start_time


PhyTimeSeriesDataSample = tuple[
    th.Tensor,  # start
    th.Tensor,  # result from data
    th.Tensor,  # result based om PDE
    th.Tensor,  # duration
]


class PhyInformedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        time_series_dataset: TimeSeriesDataset,
        diff_eq: ParametrizedEquation,
        params: Sequence[float],
    ) -> None:
        super().__init__()
        self._data = time_series_dataset
        self._simulation = EquationSimulation(diff_eq)
        self._params = th.Tensor(params)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> PhyTimeSeriesDataSample:
        starts, results_data, results_pde, durations = self.__getitems__([index])
        return starts[0], results_data[0], results_pde[0], durations[0]

    def __getitems__(self, indices: Sequence[int]) -> PhyTimeSeriesDataSample:
        starts, results_data, durations = self._data.__getitems__(indices)
        results_pde = self._simulation(starts, self._params, durations)
        return starts, results_data, results_pde, durations
