import collections
import math

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

companies = ['AAPL', 'ADBE', 'AMGN', 'AMZN', 'CMCSA', 'CSCO', 'GILD', 'INTC', 'INTU', 'MSFT', 'NVDA', 'PEP', 'QCOM',
             'SBUX', 'TXN']


def horizontal_visibility_graph(series):
    current = 0
    series_size = len(series)

    G = nx.Graph()
    G.add_nodes_from(range(series_size))
    while current < series_size - 1:
        new_bar = current + 1
        max_bar_value = -math.inf
        while new_bar < series_size:
            if series[new_bar] > max_bar_value:
                G.add_edge(current, new_bar)
                max_bar_value = series[new_bar]

            if series[new_bar] > series[current]:
                break

            new_bar += 1
        current += 1
    return G


def mutual_information(series1, series2):
    N = len(series1)
    g1 = horizontal_visibility_graph(series1)
    degrees1 = g1.degree()
    degrees_dict1 = collections.defaultdict(list)
    for node, degree in degrees1:
        degrees_dict1[degree].append(node)
    degreeCount1 = collections.Counter([d for n, d in degrees1])

    g2 = horizontal_visibility_graph(series2)
    degrees2 = g2.degree()
    degrees_dict2 = collections.defaultdict(list)
    for node, degree in degrees2:
        degrees_dict2[degree].append(node)
    degreeCount2 = collections.Counter([d for n, d in degrees2])

    mutual_information = 0
    for k1 in degreeCount1.keys():
        for k2 in degreeCount2.keys():
            joint_probability = len(np.intersect1d(degrees_dict1[k1], degrees_dict2[k2])) / N
            marginal_probability_1 = len(degrees_dict1[k1]) / N
            marginal_probability_2 = len(degrees_dict2[k2]) / N

            if joint_probability == 0:
                mutual_information = 0
            else:
                mutual_information += joint_probability * math.log2(
                    (joint_probability / (marginal_probability_1 * marginal_probability_2)))
    return mutual_information


def average_mutual_information(series_list):
    mutual_informations = []
    for i in range(len(series_list)):
        for j in range(i + 1, len(series_list)):
            mutual_informations.append(
                mutual_information(series_list[i], series_list[j]))
    return sum(mutual_informations) / len(mutual_informations)



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


if __name__ == '__main__':
    time_series_full = extract_time_series()
    time_series_segments = np.array_split(time_series_full, 72, axis=1)

    mutual_information_avgs = []
    for ts in time_series_segments:
        mutual_information_avgs.append(average_mutual_information(ts))

    plt.plot(range(len(time_series_segments)), mutual_information_avgs)
    plt.show()
