# models

Contains different GNN model architectures.

### Overview

| Filename | Class Name | Description |
|---|---|---|
| \_\_init\_\_.py | N/A | Contains the model_factory function which loads the proper model class based on the given arguments. |
| base_model.py | BaseModel | Base class for all models. Defines important instance fields used by all model classes. |
| dual_flood_gnn.py | DUALFloodGNN | Node and edge prediction model. |
| node_edge_gnn_attn.py | NodeEdgeGNNAttn | Node and edge prediction model with attention mechanism (prototype). |
| node_edge_gnn_transformer.py | NodeEdgeGNNTransformer | Graph transformer with node and edge prediction model (prototype). |
| node_gnn.py | NodeGNN | Node only prediction model based on DUALFloodGNN. |
| edge_gnn.py | EdgeGNN | Edge only prediction model based on DUALFloodGNN. |
| gcn.py | GCN | [GCN](https://arxiv.org/abs/1609.02907) with encoder and decoder. |
| edge_gcn.py | GCN (for edge) | GCN model modified for edge prediction. |
| gat.py | GAT | [GAT](https://arxiv.org/abs/1710.10903v3) with encoder and decoder. |
| edge_gat.py | GAT (for edge) | GAT model modified for edge prediction. |
