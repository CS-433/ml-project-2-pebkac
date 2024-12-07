#!/bin/bash

# This script executes the hyperparameter optimisation for each representation on the PF00014.fasta MSA file

script="python find_hyperparams.py"

reps=("onehot" "SLATM" "cMBDF" "SPAHM" "Morgan")
fasta_file="PF00014.fasta"
struct_file="PF00014_struct.dat"
xyz_dir="3d_struct_aa_xyz"
verbose="0"

for rep in "${reps[@]}"; do
	$script "$rep" "$fasta_file" "$struct_file" "$xyz_dir" "$verbose"
	if [ $? -ne 0 ]; then
		echo "Error during execution of $script with $rep"
		exit 1
	fi
done

echo "Hyperparameter optimisation successfully executed for all representations"
