"""
This script performs the hyperparameter optimisation of the model for a given MSA and a given representation
The best hyperparameters are stored in best_hyperparams.json
"""

from Utils.read_fasta_utils import quickread
from representation_generator import GenerateRepresentation
from Model.hyperparameter_search import HyperparameterSearch
import sys
import json

if len(sys.argv) != 5:
    raise ValueError("Molecular representation, fasta file, structure file, directory where xyz files are stored and verbose/not verbose must be specified")

rep_name = sys.argv[1]
fasta_file = sys.argv[2]
struct_file = sys.argv[3]
xyz_dir = sys.argv[4]
verbose = bool(int(sys.argv[5]))

print("Loading fasta file: ")
Z, W = quickread(fasta_file)
data_dict = {
    "Z": Z,
    "W": W,
}
print("Generating {} representations".format(rep_name))
reps_matrix = GenerateRepresentation(rep_name, xyz_dir)
print("Hyperparameter search: ")
best_params = HyperparameterSearch(reps_matrix, data_dict, struct_file, verbose=verbose)

print("Best hyperparameters: ")
print(best_params)

with open("best_hyperparams.json", "r") as f:
    best_params_dict = json.loads(f.read())

best_params_dict[rep_name + "_" + fasta_file] = best_params

with open("best_hyperparams.json", "w") as f:
        f.write(json.dumps(best_params_dict))