from torch_geometric.data import Data

class LineGraphData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_edge_index = kwargs.get('line_edge_index', None)
        self.line_edge_attr = kwargs.get('line_edge_attr', None)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_edge_index':
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)
