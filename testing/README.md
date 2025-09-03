# training

Contains Tester classes used to test the model.

### Overview

| Filename | Class Name | Description |
|---|---|---|
| base_tester.py | BaseTester | Base class used by all tester classes. |
| node_regression_tester.py | NodeRegressionTester | Tests a node prediction model in a supervised manner where timesteps from the data are given as input. |
| edge_regression_tester.py | EdgeRegressionTester | Tests an edge prediction model in a supervised manner where timesteps from the data are given as input. |
| dual_regression_tester.py | DualRegressionTester | Tests a node and edge prediction model in a supervised manner where timesteps from the data are given as input. |
| node_autoregressive_tester.py | NodeAutoregressiveTester | Tests a node prediction in an autoregressive manner where the previous model output is used as the input for the next timestep. |
| edge_autoregressive_tester.py | EdgeAutoregressiveTester | Tests an edge prediction in an autoregressive manner where the previous model output is used as the input for the next timestep. |
| dual_autoregressive_tester.py | DualAutoregressiveTester | Tests a node and edge prediction in an autoregressive manner where the previous model output is used as the input for the next timestep. |
