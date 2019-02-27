import networkx as nx
import math
import collections
import numpy as np


class HorizontalVisibilityGraph(object):

    def __init__(self, company, series):
        self.company = company
        self.series = series
        self.graph = nx.Graph()

        self.generate()

    def generate(self):
        current = 0
        series_size = len(self.series)

        self.graph.add_nodes_from(range(series_size))
        while current < series_size - 1:
            new_bar = current + 1
            max_bar_value = -math.inf
            while new_bar < series_size:
                if self.series[new_bar] > max_bar_value:
                    self.graph.add_edge(current, new_bar)
                    max_bar_value = self.series[new_bar]

                if self.series[new_bar] > self.series[current]:
                    break

                new_bar += 1
            current += 1
        return self.graph

    def degree_distribution(self):
        degrees = self.graph.degree()
        degrees_dict = collections.defaultdict(list)
        for node, degree in degrees:
            degrees_dict[degree].append(node)
        degrees_count = collections.Counter([d for n, d in degrees])

        return degrees_dict, degrees_count

    def mutual_information(self, other):

        N = self.series.size

        degrees_dict_self, degrees_count_self = self.degree_distribution()
        degrees_dict_other, degrees_count_other = other.degree_distribution()

        mutual_information = 0
        for k1 in degrees_count_self.keys():
            for k2 in degrees_count_other.keys():
                joint_probability = len(np.intersect1d(degrees_dict_self[k1], degrees_dict_other[k2])) / N
                marginal_probability_1 = len(degrees_dict_self[k1]) / N
                marginal_probability_2 = len(degrees_dict_other[k2]) / N

                if joint_probability == 0:
                    mutual_information = 0
                else:
                    mutual_information += joint_probability * math.log2(
                        (joint_probability / (marginal_probability_1 * marginal_probability_2)))
        return mutual_information

