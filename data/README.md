# data

Contains the raw data and Dataset classes for accessing this data.

### Overview

| Filename | Class Name | Description |
|---|---|---|
| \_\_init\_\_.py | N/A | Contains the dataset_factory function which loads the proper dataset class based on the given arguments. |
| flood_event_dataset.py | FloodEventDataset | Base dataset class for accessing flood events. Data points are read from disk during access. |
| in_memory_flood_dataset.py | InMemoryFloodDataset | Similar to FloodEventDataset but loads all data points in memory for faster access. |
| autoregressive_flood_dataset.py | AutoregressiveFloodDataset | Dataset which loads multiple timesteps for node features, edge features and labels. Used for autoregressive training. |
| in_memory_autoregressive_flood_dataset.py | InMemoryAutoregressiveFloodDataset | Similar to AutoregressiveFloodDataset but loads all data points in memory for faster access. |
| hecras_data_retrieval.py | N/A | Functions used to retrieve data from HEC-RAS simulation files (.hdf). |
| shp_data_retrieval.py | N/A | Functions used to retrieve data from shape files (.shp). |
| boundary_condition.py | BoundaryCondition | Class used in FloodEventDataset. Handles creation of boundary conditions/cells and removal of ghost cells. |
| dataset_normalizer.py | DatasetNormalizer | Class used in FloodEventDataset. Handles normalization of dataset features. |

![Overview of Classes in the data Folder](../docs/data_overview.png)
