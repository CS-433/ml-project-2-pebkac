import os
import glob
from rdkit import Chem

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
        
    """
    if output_dir not in os.listdir("."):
        os.mkdir(output_dir)
    sdf_files = glob.glob(input_dir + '/*.sdf')

    for i, sdf_file in enumerate(sdf_files):
        suppl = Chem.rdmolfiles.ForwardSDMolSupplier(sdf_file, removeHs=False)
        label = aa_onelettercode[i]
        for mol in suppl:
            Chem.rdmolfiles.MolToXYZFile(mol, output_dir + "/" + label + ".xyz")