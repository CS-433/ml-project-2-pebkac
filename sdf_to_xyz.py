"""
This script generates xyz files from the sdf files corresponding to the different amino acids
"""

import os
import glob
from rdkit import Chem
import sys

aa_onelettercode = ["A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y"]

def sdf_to_xyz(input_dir, output_dir):
    """
    Converts .sdf files in .xyz files and stores them in a new directory
    Important : each file contains only one molecule
    inputs :
        input_dir : str, name of directory where .sdf files are stored
        output_dir : str, name of directory where .xyz files will be stored
    returns :
        None
    """
    if output_dir not in os.listdir("."):
        os.mkdir(output_dir)
    sdf_files = glob.glob(input_dir + '/*.sdf')

    for i, sdf_file in enumerate(sdf_files):
        suppl = Chem.rdmolfiles.ForwardSDMolSupplier(sdf_file, removeHs=False)
        label = aa_onelettercode[i]
        for mol in suppl:
            Chem.rdmolfiles.MolToXYZFile(mol, output_dir + "/" + label + ".xyz")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Directory containing the SDF files and directory where the XYZ files are going to be stored need to be specified")
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    sdf_to_xyz(input_dir, output_dir)