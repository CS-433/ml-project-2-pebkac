[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
# Creating the Conda environment
To create the Conda environment required for the scripts in this repository to work, you first need to create a Conda environment and download the juliacall module via pip:
```
conda create -n ml-project-2-pebkac
conda activate ml-project-2-pebkac
pip install juliacall
conda env update --name ml-project-2-pebkac --file environment.yml
```
The packages needed are given in the `environment.yml` file. This file may be updated during this project.


# AttentionDCA.jl
This directory was not created during this project ! It can be found in the GitHub repository [AttentionDCA.jl](https://github.com/pagnani/AttentionDCA.jl). 

The associated article is available [here](https://www.biorxiv.org/content/10.1101/2024.02.06.579080v2.abstract).
