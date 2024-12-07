#!/bin/bash

# This script executes the sdf_to_xyz.py script with the directory names that were used

script="python sdf_to_xyz.py"
input_dir="3d_struct_aa_sdf"
output_dir="3d_struct_aa_xyz"

$script "$input_dir" "$output_dir"

if [ $? -ne 0 ]; then
	echo "Error during execution of $script with arguments $input_dir and $output_dir"
	exit 1
fi

echo "Successfull generation of XYZ files from SDF files"