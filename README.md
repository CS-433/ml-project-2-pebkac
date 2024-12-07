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


