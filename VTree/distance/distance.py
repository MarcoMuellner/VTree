import numpy as np
from numba import jit, float64
from VTree.timeseries.Timeseries import Timeseries
from itertools import islice,combinations
from scipy.spatial.distance import squareform

from typing import List

#https://github.com/cmackenziek/tsfl

@jit(float64[:, :](float64[:, :], float64[:], float64[:], float64[:],
    float64[:], float64, float64), nopython=True)
def pairwise_tweds(TWED, A, A_times, B, B_times, lam=0.5, nu=1e-5):
    for i in range(1, len(A)):
        for j in range(1, len(B)):
            TWED[i, j] = min(
                # insertion
                (TWED[i - 1, j] + abs(A[i - 1] - A[i]) +
                 nu*(A_times[i] - A_times[i - 1]) + lam),
                # deletion
                (TWED[i, j - 1] + abs(B[j - 1] - B[j]) +
                 nu*(B_times[j] - B_times[j - 1]) + lam),
                # match
                (TWED[i - 1, j - 1] + abs(A[i] - B[j]) +
                 nu*(A_times[i] - B_times[j]) +
                 abs(A[i - 1] - B[j - 1]) +
                 nu*(A_times[i - 1] - B_times[j - 1]))
            )
    return TWED


def twed(A, A_times, B, B_times, lam=0.5, nu=1e-5):
    n, m = len(A), len(B)

    A, A_times = np.append(0.0, A), np.append(0.0, A_times)
    B, B_times = np.append(0.0, B), np.append(0.0, B_times)

    TWED = np.zeros((n + 1, m + 1))
    TWED[:, 0] = np.finfo(np.float).max
    TWED[0, :] = np.finfo(np.float).max
    TWED[0, 0] = 0.0

    TWED = pairwise_tweds(TWED, A, A_times, B, B_times, lam=lam, nu=nu)
    return TWED[n, m]

def complexity_twed(A, A_times, B, B_times, lam=0.5, nu=1e-5):
    n, m = len(A), len(B)

    A, A_times = np.append(0.0, A), np.append(0.0, A_times)
    B, B_times = np.append(0.0, B), np.append(0.0, B_times)

    TWED = np.zeros((n + 1, m + 1))
    TWED[:, 0] = np.finfo(np.float).max
    TWED[0, :] = np.finfo(np.float).max
    TWED[0, 0] = 0.0

    TWED = pairwise_tweds(TWED, A, A_times, B, B_times, lam=lam, nu=nu)
    return TWED[n, m]

def complexity_coeff(lc_a, times_a, lc_b, times_b):
    complexity_1 = np.sum(np.sqrt(np.power(np.diff(lc_a), 2)) +
                          np.sqrt(np.power(np.diff(times_a), 2)))
    complexity_2 = np.sum(np.sqrt(np.power(np.diff(lc_b), 2)) +
                          np.sqrt(np.power(np.diff(times_b), 2)))
    return max(complexity_1, complexity_2)/min(complexity_1, complexity_2)


def time_series_twed(ts1 : Timeseries, ts2 : Timeseries, lam=0.5, nu=1e-5):
    return twed(ts1.flux, ts1.time, ts2.flux, ts2.time, lam, nu)

def complexity_time_series_twed(ts1 : Timeseries, ts2 : Timeseries, lam=0.5, nu=1e-5):
    return complexity_twed(ts1.flux, ts1.time, ts2.flux, ts2.time, lam, nu)

def distance_matrix(ts : List[Timeseries]) -> np.ndarray:
    distances = []
    for i, j in islice(combinations(ts, 2), None):
        distances.append(time_series_twed(i, j))
    distance_matrix = squareform(distances)
    return distance_matrix

def distances(ts : List[Timeseries]) -> List[float]:
    return [time_series_twed(i, j) for i, j in islice(combinations(ts, 2), None)]