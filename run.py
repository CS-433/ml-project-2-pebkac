from read_fasta_utils import quickread
from RepresentationGenerator import GenerateRepresentation
from HyperparameterSearch import HyperparameterSearch
import sys

if len(sys.argv) != 4:
    raise ValueError("Molecular representation, fasta file and verbose/not verbose must be specified")

rep_name = sys.argv[1]
fasta_file = sys.argv[2]
verbose = bool(int(sys.argv[3]))

print("Loading fasta file: ")
Z, W = quickread(fasta_file)
data_dict = {
    "Z": Z,
    "W": W,
}
print("Generating {} representations".format(rep_name))
reps_matrix = GenerateRepresentation(rep_name, "3d_struct_aa_xyz")
print("Hypermarameter search: ")
best_params = HyperparameterSearch(reps_matrix, data_dict, "PF00014_struct.dat", verbose=verbose)

print(best_params)