# models

Contains different GNN model architectures.

### Overview

| Filename | Class Name | Description |
|---|---|---|
| \_\_init\_\_.py | N/A | Contains the model_factory function which loads the proper model class based on the given arguments. |
| base_model.py | BaseModel | Base class for all models. Defines important instance fields used by all model classes. |
| node_edge_gnn_attn.py | NodeEdgeGNNAttn | Node and edge prediction model with attention mechanism. |
| node_edge_gnn.py | NodeEdgeGNN | Node and edge prediction model. |
| node_gnn_attn.py | NodeGNNAttn | Node prediction model based on NodeEdgeGNNAttn. |
| edge_gnn_attn.py | EdgeGNNAttn | Edge prediction model based on NodeEdgeGNNAttn. |
| gcn.py | GCN | [GCN](https://arxiv.org/abs/1609.02907) with encoder and decoder. |
| gat.py | GAT | [GAT](https://arxiv.org/abs/1710.10903v3) with encoder and decoder. |
