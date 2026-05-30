# DUALFloodGNN

This repository contains the code for "DUALFloodGNN: Physics-informed Graph Neural Networks for Operational Flood Modeling." DUALFloodGNN is a physics-informed flood GNN architecture comprised of three main components: (1) a model that performs shared message passing to predict both node and edge features, (2) a physics-informed loss function that enforces global and local mass conservation between consecutive predictions, and (3) an autoregressive training strategy utilizing dynamic curriculum learning. This paper was accepted at the IJCAI-ECAI 2026 AI4Tech track. Read more about the paper [here](https://arxiv.org/abs/2512.23964).


![DUALFloodGNN Overview](/docs/overview.png)

## Setup

### Environment
1. Create a virtual environment (with either conda or venv). This repository has been tested on Python 3.12.3.
```bash
python -m venv venv

source venv/bin/activate # Linux
venv/Scripts/Activate.ps1 # Windows
```
2. Install PyTorch based on your CUDA version. This repository has been tested on PyTorch version 2.5.1. Replace `${CUDA}` with the apporpriate CUDA version for your machine (ex. `${CUDA}` -> 124 for CUDA 12.4).
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA}
```

3. Install PyTorch Geometric based on your PyTorch and CUDA version. Again, replace `${CUDA}` with the apporpriate CUDA version for your machine.

```bash
# Main library
pip install torch_geometric

# Additional libraries
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu${CUDA}.html
```

4. Install the remaining dependencies.

```bash
pip install -r requirements.txt
```

### Dataset

The following are the instructions to setup the dataset needed for the repository.

#### HEC-RAS simulation files, Summary files

1. Download the following files from [DOI: 10.25910/9xav-0s86](https://doi.org/10.25910/9xav-0s86).
- HEC-RAS Flood Simulation Files in HDF Format (`HDF_FILES.zip`)
- Dummy train dataset summary file (`train.csv`)
- Dummy test dataset summary file (`test.csv`)

**Note**: DO NOT download the Geometry files from here as they are not updated.

2. Extract the files from `HDF_FILES.zip`. Rename the extracted folder to `HEC-RAS Results`.

#### Updated Geometry files

1. Download the updated geometry files from [this Google Drive](https://drive.google.com/drive/folders/1JrIhOoCzYDMZQVtuuKDYvo3wASihmpTt?usp=sharing).
2. Extract the downloaded files. Rename the extracted folder to `GEOMETRY`

#### File Structure

You should now have all all the important files needed for the dataset. These should include the following:
| File | Extension | Source |
|------|-----------|--------|
| Node shape file | .shp | Updated Geometry Files |
| Links shape file | .shp | Updated Geometry Files |
| DEM file | .tif | Updated Geometry Files |
| HEC-RAS simulation files | .hdf | HEC-RAS Simulation Files |
| Summary file for training events | .csv | Summary Files |
| Summary file for testing events | .csv | Summary Files |

1. Create a `raw` folder in the `data/datasets` directory.
2. Place the `HEC-RAS Results` and `GEOMETRY` in the raw data folder created in step 1. Transfer the train.csv and test.csv files to the raw data folder as well. The folder structure should look like this:
```
data/
├── datasets/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── GEOMETRY/
│   │   ├── cell_centers_with_ele.shp
│   │   ├── links_with_slope.shp
│   │   ├── DEM.tif
│   │   ...
│   └── HEC-RAS Results/
│       ├── Model_01.p22.hdf
│       ├── Model_01.p23.hdf
│       ...
```

For more information, refer to the `README.pdf` documentation file [here](https://doi.org/10.25910/9xav-0s86).

#### Dataset Features

The following table provides an overview of the features used in the dataset.

| Feature Class | Feature Type | Name | Description | Source |
|-------|------|------|-------------|--------|
| Graph | Static | Edge index | Describe which nodes are connected to each other in the format [from_node, to_node] | Node Shape |
| Node | Static | Position | X and Y coordinates of a node | Node Shape |
| Node | Static | Area | Area of the mesh cell | HEC-RAS |
| Node | Static | Roughness | Manning's coefficient in mesh cell | HEC-RAS |
| Node | Static | Elevation | Elevation from sea level | Node Shape |
| Node | Static | Aspect | Slope orientation in degrees clockwise from north | DEM |
| Node | Static | Curvature | Total curvature of topographic surface | DEM |
| Node | Static | Flow Accumulation | Upslope areas that drain into location. Computed using the D8 Algorithm. | DEM |
| Node | Dynamic | Rainfall | Rainfall at the cell for a specific time step. Note: HEC-RAS specifies this as cumulative rainfall but we convert it to interval rainfall for our use case. | HEC-RAS |
| Node | Dynamic | Water Volume | Volume at the cell for a specific time step. Note: We clip water volume values to 100,000 at a maximum. | HEC-RAS |
| Edge | Static | Relative Position | X and Y coordinates of a node relative to its neighbors | Edge Shape |
| Edge | Static | Face Length | Length of the border of a mesh cell | HEC-RAS |
| Edge | Static | Length | Length of a cell | Edge Shape |
| Edge | Static | Slope | Slope of the cell in a mesh | Edge Shape |
| Edge | Dynamic | Face Flow | Water flow / flux at an edge for a time step | HEC-RAS |

## Running the Code

### Quick Start

To run the training code, use the following command:
```bash
python train.py --config 'configs/config.yaml' --model 'DUALFloodGNN'
```

Similarly, to run the testing code, use the following command:
```bash
python test.py --config 'configs/config.yaml' --model 'DUALFloodGNN' --model_path 'path/to/model_checkpoint.pt'
```
**IMPORANT**: Make sure train before running tests, as the testing code requires a trained model checkpoint and a processed dataset to perform inference.

### Entry Points

Below is the exhaustive list of entry points for the application.

| File | Description | Arguments |
|---|---|---|
| `train.py` | Train the model with the parameters specified in the config file. | `--config`, `--model`, `--with_test` `--seed` `--device` `--debug` |
| `test.py` | Perform inference using the specified model checkpoint with test data. | `--config`, `--model`, `--model_path`, `--seed`, `--device`, `--debug` |
| `hp_search.py` | Perform a Bayesian hyperparameter search with the specified hyperparameters and events. (WARNING: not fully tested.) | `--config`, `--hparam_config`, `--model`, `--seed`, `--device` |
| `eda.ipynb` | Jupyter notebook that gives an overview and analysis of the data. | N/A |
| `view_results.ipynb` | Jupyter notebook where you may view the results of model training and testing. | N/A |

Notes
- .sh files are mainly used for running programs in the slurm cluster.

## Code Structure

The code is categorized in different folder based on their specific purpose. Below is an overview of all the folders.

| Folder | Description |
|---|---|
| [configs](https://github.com/acostacos/flood_pi_gnn/tree/master/configs) | Contains all the config files used to specify training and testing parameters. |
| [constants](https://github.com/acostacos/flood_pi_gnn/tree/master/constants) | Contains constants used throughout the codebase. |
| [data](https://github.com/acostacos/flood_pi_gnn/tree/master/data) | Contains the raw data and Dataset classes for accessing this data. |
| [loss](https://github.com/acostacos/flood_pi_gnn/tree/master/loss) | Contains custom loss functions used for training (ex. physics-informed loss). |
| [models](https://github.com/acostacos/flood_pi_gnn/tree/master/models) | Contains different GNN model architectures. |
| [testing](https://github.com/acostacos/flood_pi_gnn/tree/master/testing) | Contains Tester classes used to test the model. |
| [training](https://github.com/acostacos/flood_pi_gnn/tree/master/training) | Contains Trainer classes used to train the model. |
| [utils](https://github.com/acostacos/flood_pi_gnn/tree/master/utils) | Contains various utility classes and objects. |

## Citation

If you use this code for your research, please cite [our paper](https://arxiv.org/abs/2512.23964):
```
@misc{acosta2026,
      title={DUALFloodGNN: Physics-informed Graph Neural Network for Operational Flood Modeling}, 
      author={Carlo Malapad Acosta and Herath Mudiyanselage Viraj Vidura Herath and Jia Yu Lim and Abhishek Saha and Sanka Rasnayaka and Lucy Marshall},
      year={2026},
      eprint={2512.23964},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.23964}, 
}
```
