from .base_trainer import BaseTrainer
from .dual_autoregressive_trainer import DualAutoRegressiveTrainer
from .dual_regression_trainer import DualRegressionTrainer
from .edge_regression_trainer import EdgeRegressionTrainer
from .node_autoregressive_trainer import NodeAutoRegressiveTrainer
from .node_regression_trainer import NodeRegressionTrainer

def trainer_factory(model_name: str, autoregressive: bool, *args, **kwargs) -> BaseTrainer:
    if 'NodeEdgeGNN' in model_name:
        if autoregressive:
            return DualAutoRegressiveTrainer(*args, **kwargs)
        return DualRegressionTrainer(*args, **kwargs)

    if model_name in ['EdgeGNNAttn']:
        # if autoregressive:
        #     return EdgeAutoRegressiveTrainer(*args, **kwargs)
        return EdgeRegressionTrainer(*args, **kwargs)

    if autoregressive:
        return NodeAutoRegressiveTrainer(*args, **kwargs)
    return NodeRegressionTrainer(*args, **kwargs)

__all__ = [
    'DualAutoRegressiveTrainer',
    'DualRegressionTrainer',
    'EdgeRegressionTrainer',
    'NodeAutoRegressiveTrainer',
    'NodeRegressionTrainer',
    'trainer_factory',
]
