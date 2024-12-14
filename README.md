[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
# Project description
In the original article presenting the [AttentionDCA](https://www.biorxiv.org/content/10.1101/2024.02.06.579080v2.abstract) model, it is shown that a multi-headed attention-block can be used for the Direct Coupling Analysis (DCA) of Multiple Sequence Alignments (MSA) of families of small proteins.

The goal of this project is to modify this multi-headed attention-block in order to include chemical information about the different amino-acids in the form of molecular representations. With these modifications, learning the value matrices is turned into a metric learning task. The resulting metric matrices of each head of the attention-block can then be used to analyze the importance of each feature in the molecular representation used.

The different molecular representations used for this project are one-hot encodings, [SPAHM](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00050k#cit53), [MBDF](https://pubs.aip.org/aip/jcp/article/159/3/034106/2902959), [SLATM](https://arxiv.org/abs/1807.04259) and [Morgan](https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf) representations.

# Creating the Conda environment
A conda environment containing the required conda packages can be created as follows:
```
conda env create -f environment.yml
```
Note: The file `environment.yml` will likely be updated during the project

# Installing pip dependencies
The scripts additionally require to install packages via pip. To install these packages, execute the following commands from the `ml-project-2-pebkac` environment:
```
python -m pip install juliacall
python -m pip install "qstack[all] @ git+https://github.com/lcmd-epfl/Q-stack.git"
python -m pip install "qml2 @ git+https://github.com/qml2code/qml2.git"
```
WARNING: The `qstack` package only works with UNIX-based operating systems. To build this environment with Windows, using WSL is required

# General workflow:
1. XYZ files corresponding to the different amino-acids are generated from their corresponding SDF files
2. A fasta file corresponding to a given MSA is read to generate an alignment matrix containing each sequence of the MSA
3. For each representation, the hyperparameters of the multi-headed attention block are optimized for the MSA of interest with Bayesian optimization
4. The performance of each representation is evaluated based on the PPV curve generated by the associated multi-headed attention-block.
   In this context, a PPV curve illustrates the ability of the model to predict correct contacts between the different amino-acids in the protein
5. For the MSA of interest, the metric matrices as extracted from the multi-headed attention blocks corresponding to each molecule, and feature importance is derived from these metric matrices

# Python scripts
1. sdf_to_xyz.py: Generates XYZ files from SDF files in a given input directory and stores them in a given output directory
2. find_hyperparams.py: Performs hyperparameter optimization for a given MSA and a given representation
## Model
1. AttentionDCA.py: Defines the structure of the multi-headed attention-block
2. Dataset.py: Defines the Dataset object used during model training for mini-batching
3. hyperparameter_search.py: Defines the functions used for Bayesian hyperparameter optimisation
4. train.py: Defines the training loop used to train the model

## Utils
1. ppv_curve_utils.py: Provides the functions used to generate the PPV curves
2. read_fasta_utils.py: Provides the functions used to generate an alignment matrix from a fasta file
3. representation_generator.py: Provides the functions used to generate and pre-process molecular representations of amino-acids

# Shell scripts
1. sdf_to_xyz.sh: Generates XYZ files from SDF files in 3d_struct_aa_sdf and stores them in 3d_struct_aa_xyz
2. find_hyperparams.sh: Performs hyperparameter optimization for each representation on the PF00014 MSA, and stores the optimal hyperparameters in best_hyperparams.json

WARNING: It is not required to run these scripts, as the 3d_struct_aa_xyz directory and the best_hyperparams.json file are given

# Data directories
1. 3d_struct_aa_sdf: Contains the SDF files corresponding to the different amino-acids
2. 3d_struct_aa_xyz: Contains the XYZ files corresponding to the different amino-acids

# JSON files:
1. best_hyperparams.json: Contains the optimal hyperparameters of the multi-headed attention block for each representation when trained on the PF00014 MSA
