import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

from MultiplexVisibilityGraph import MultiplexVisibilityGraph

companies = ['AAPL', 'ADBE', 'AMGN', 'AMZN', 'CMCSA', 'CSCO', 'GILD', 'INTC', 'INTU', 'MSFT', 'NVDA', 'PEP', 'QCOM',
             'SBUX', 'TXN']

START_YEAR = 2000
END_YEAR = 2018
SEGMENT_LENGTH = 3  # In months
N_SEGMENTS = (END_YEAR - START_YEAR + 1) * (12 / SEGMENT_LENGTH)


def extract_time_series():
    __time_series = None

    for company in companies:
        dataset = pd.read_csv('data/daily_' + company + '.csv')
        series = dataset.iloc[:, 4].values
        if __time_series is not None:
            __time_series = np.vstack((__time_series, series))
        else:
            __time_series = series
    return __time_series


def get_segment_dates():
    dates_total = np.array_split(first_dataset.iloc[:, 0].values, N_SEGMENTS)
    dates = np.flip([date[0] for date in dates_total])
    return dates


def compute_average_mutual_information():
    __average_mutual_information = []
    for ts in time_series_segments:
        mvg = MultiplexVisibilityGraph(ts, companies)
        __average_mutual_information.append(mvg.average_mutual_information())
    return __average_mutual_information


if __name__ == '__main__':
    first_dataset = pd.read_csv('data/daily_AAPL.csv')

    # Setup x axis for dates
    segment_dates = get_segment_dates()
    x_axis = mdates.datestr2num(segment_dates)
    register_matplotlib_converters()

    # Get time series in numpy format
    # Split the time series based on the segments length
    time_series_full = extract_time_series()
    time_series_segments = np.array_split(time_series_full, N_SEGMENTS, axis=1)

    average_mutual_information = compute_average_mutual_information()

    plt.plot_date(x_axis, average_mutual_information, linestyle='solid', marker='None')
    plt.show()
