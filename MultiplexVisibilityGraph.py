from HorizontalVisibilityGraph import HorizontalVisibilityGraph
from itertools import combinations
import numpy as np


class MultiplexVisibilityGraph(object):

    def __init__(self, series_segments, companies):
        self.series_segments = series_segments
        self.companies = companies
        self.graphs = dict()

        self.generate()

    def generate(self):
        for idx, series in enumerate(self.series_segments):
            self.graphs[self.companies[idx]] = HorizontalVisibilityGraph(self.companies[idx], series)
        return self.graphs

    def average_mutual_information(self):
        mutual_informations = []

        for graph_1, graph_2 in combinations(self.graphs.values(), 2):
            mutual_informations.append(graph_1.mutual_information(graph_2))

        return np.mean(mutual_informations)

