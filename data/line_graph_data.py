from torch_geometric.data import Data

class LineGraphData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual_edge_index = kwargs.get('dual_edge_index', None)
        self.dual_edge_attr = kwargs.get('dual_edge_attr', None)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'dual_edge_index':
            return self.num_edges
        if key == 'dual_edge_attr':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
