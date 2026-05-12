# DUALFloodGNN

This repository contains the code for "DUALFloodGNN: Physics-informed Graph Neural Networks for Operational Flood Modeling." DUALFloodGNN is a physics-informed flood GNN architecture comprised of three main components: (1) a model that performs shared message passing to predict both node and edge features, (2) a physics-informed loss function that enforces global and local mass conservation between consecutive predictions, and (3) an autoregressive training strategy utilizing dynamic curriculum learning. This paper was accepted at the IJCAI-ECAI 2026 AI4Tech track.

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

### Data

1. Download the dataset from [DOI: 10.25910/9xav-0s86](https://doi.org/10.25910/9xav-0s86). There are 4 necessary files for the dataset: train.csv, test.csv, GEOMETRY.zip and HDF_FILES.zip. The important files for the dataset are as follows:
  - Node shape file (.shp)
  - Links shape file (.shp)
  - DEM file (.tif)
  - HEC-RAS simulation files (.hdf)
  - Summary file for training events (.csv)
  - Summary file for testing events (.csv)
2. Create a `raw` folder in the `data/datasets` directory.
3. Unzip the GEOMETRY.zip and HDF_FILES.zip files and place the unzipped files in the raw data folder created in step 2. Transfer the train.csv and test.csv files to the raw data folder as well. The folder structure should look like this:
```
data/
├── datasets/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── GEOMETRY/
│   │   ├── updated_cell_centers.shp
│   │   ├── links.shp
│   │   ├── DEM.tif
│   │   ...
│   └── HDF_FILES/
│       ├── Model_01.p22.hdf
│       ├── Model_01.p23.hdf
│       ...
```

For more information, refer to the `README.pdf` documentation file.

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
