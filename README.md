[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
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


