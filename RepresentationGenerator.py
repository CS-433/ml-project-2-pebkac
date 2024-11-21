import os
import numpy as np
import qml2
from qml2.representations.slatm import get_slatm_mbtypes, generate_slatm
from qml2.representations.cMBDF import generate_cmbdf, get_asize, get_convolutions
from rdkit.Chem import MolFromSmiles, AddHs, rdFingerprintGenerator
from qstack import compound, spahm
from tqdm import tqdm


class RepresentationGenerator:
    """
    This object serves to generate a given representation for the set of amino-acids.
    To generate a given representation, initialize a RepresentationGenerator object :
        input :
            repType : str, representation that is going to be generated - "onehot", "cMBDF", "SLATM", "SPAHM", or "Morgan"
    and call the RepresentationGeneration method :
        inputs : 
            input_dir : str, a directory where the .xyz files are stored
        returns :
            A : np.array of shape (20, D), each row being a representation of an amino acid

    e.g. : reps_matrix = RepresentationGenerator("SLATM").RepresentationGeneration("xyz_directory")
    """

    def __init__(self, repType):
        self.repType = repType
        self.aa_onelettercode = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        self.aa_smiles = {
            "A": "C[C@@H](C(=O)O)N",
            "C": "C([C@@H](C(=O)O)N)S",
            "D": "C([C@@H](C(=O)O)N)C(=O)O",
            "E": "C(CC(=O)O)[C@@H](C(=O)O)N",
            "F": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",
            "G": "C(C(=O)O)N",
            "H": "C1=C(NC=N1)C[C@@H](C(=O)O)N",
            "I": "CC[C@H](C)[C@@H](C(=O)O)N",
            "K": "C(CCN)C[C@@H](C(=O)O)N",
            "L": "CC(C)C[C@@H](C(=O)O)N",
            "M": "CSCC[C@@H](C(=O)O)N",
            "N": "C([C@@H](C(=O)O)N)C(=O)N",
            "P": "C1C[C@H](NC1)C(=O)O",
            "Q": "C(CC(=O)N)[C@@H](C(=O)O)N",
            "R": "C(C[C@@H](C(=O)O)N)CN=C(N)N",
            "S": "C([C@@H](C(=O)O)N)O",
            "T": "C[C@H]([C@@H](C(=O)O)N)O",
            "V": "CC(C)[C@@H](C(=O)O)N",
            "W": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
            "Y": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O"
        }
        self.repDict = {
            "onehot": self.OneHotRepresentation,
            "cMBDF": self.cMBDFRepresentation,
            "SLATM": self.SLATMRepresentation,
            "SPAHM": self.SPAHMRepresentation,
            "Morgan": self.MorganFingerprints,
        }
    
    def OneHotRepresentation(self, input_dir):
        # Retrieves the number of files (molecules)
        N = len(os.listdir(input_dir))
        # Generates the one-hot representation matrix for all amino acids
        one_hot = np.eye(N)
        return one_hot

    def cMBDFRepresentation(self, input_dir):
        molsList = []

        for fileName in os.listdir(input_dir):
            # converts the amino acid contained in each .xyz file to a qml2.Compound object
            mol = qml2.Compound(input_dir + "/" + fileName)
            
            # recovers the number of atoms in each amino acid
            with open(input_dir + "/" + fileName, 'r') as file:
                lines = file.read().split('\n')
                numAtoms = int(lines[0])
            
            # concatenates the amino acid and its number of atoms in a list
            subList = [mol, numAtoms]
            molsList.append(subList)

        # stores the qml2.Compound and the numbers of atoms in two lists
        compoundsList = [subList[0] for subList in molsList]
        sizesList = [subList[1] for subList in molsList]

        # extracts the charges and coordinates of each amino acid from the qml2.Compound objects
        charges = [compound.nuclear_charges for compound in compoundsList]
        coordinates = [compound.coordinates for compound in compoundsList]
        pad = max(sizesList)

        asize = get_asize(charges)
        convs = get_convolutions()
        repsList = []

        # generates the cMBDF representation of each amino acid and stores it in a list
        print("Generating cMBDF representations: ")
        for i in tqdm(range(len(compoundsList))):
            rep = generate_cmbdf(charges[i], coordinates[i], convs, local=False, pad=pad, asize=asize)
            repsList.append(rep)

        # stacks the representation of each amino acid as a row in a np.array
        A = np.row_stack(tuple(repsList))
        return A

    def SLATMRepresentation(self, input_dir):
        molsList = []

        for fileName in os.listdir(input_dir):
            # converts the amino acid contained in each .xyz file to a qml2.Compound objec
            mol = qml2.Compound(input_dir + "/" + fileName)
            
            # recovers the number of atoms in each amino acid
            with open(input_dir + "/" + fileName, 'r') as file:
                lines = file.read().split('\n')
                numAtoms = int(lines[0])
            
            # concatenates the amino acid and its number of atoms in a list
            subList = [mol, numAtoms]
            molsList.append(subList)

        # stores the qml2.Compound and the numbers of atoms in two lists
        compoundsList = [subList[0] for subList in molsList]

        # extracts the charges and coordinates of each amino acid from the qml2.Compound objects
        charges = [compound.nuclear_charges for compound in compoundsList]
        coordinates = [compound.coordinates for compound in compoundsList]

        mbtypes = get_slatm_mbtypes(charges)
        repsList = []

        # generates the SLATM representation of each amino acid and stores it in a list
        print("Generating SLATM representations: ")
        for i in tqdm(range(len(compoundsList))):
            rep = generate_slatm(charges[i], coordinates[i], mbtypes)
            repsList.append(rep)

        # stacks the representation of each amino acid as a row in a np.array
        A = np.row_stack(tuple(repsList))
        return A
    
    def SPAHMRepresentation(self, input_dir):
        reps_list = []
        
        print("Generating SPAHM representations: ")
        for fileName in tqdm(os.listdir(input_dir)):
            # converts the amino acid contained in each .xyz file to a rdkit.Chem.rdchem.Mol object
            mol = compound.xyz_to_mol(input_dir + '/' + fileName, 'def2-svp', charge=0, spin=0)

            # generates the SPAHM represenation of the amino acid and stores it in a list
            rep = spahm.compute_spahm.get_spahm_representation(mol, "lb")[0, :]
            reps_list.append((rep, rep.size))
        
        # retrieves the longest representation
        max_size = max(reps_list, key=lambda t: t[1])[1]
        padded_reps_list = []

        # concatenates each representation vector and an all-zero np.array to make all 
        # representations of equal length and stores them in a list
        for rep, rep_size in reps_list:
            zeros = np.zeros(max_size - rep_size)
            padded_rep = np.concatenate((rep, zeros), axis=None)
            padded_reps_list.append(padded_rep)
        
        A = np.row_stack(tuple(padded_reps_list))
        return A   
    
    
    def MorganFingerprints(self, input_dir):
        molsList = []

        # converts the amino acid contained in each .xyz file to a rdkit.Chem.rdchem.Mol object
        # via its smiles description
        print("Generating Morgan fingerprints: ")
        for fileName in tqdm(os.listdir(input_dir)):
            mol = MolFromSmiles(self.aa_smiles[fileName[0]])
            mol = AddHs(mol)
            molsList.append(mol)

        # initializes a Morgan Fingerprint generator
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
        fpList = []

        # generates the Morgan FP representation of each amino acid and stores it in a list
        fpList = list(map(fpgen.GetFingerprint, molsList))

        A = np.row_stack(tuple(fpList))
        return A
    
    
    def RepresentationGeneration(self, input_dir):

        # Generates the desired representation
        A = self.repDict[self.repType](input_dir)
        return A


def RepsNormalization(A):
    """
    Normalizes each feature of a representation matrix A
    inputs:
        A : np.array of shape (20, D), representation matrix
    returns:
        A_normalized : np.array of shape (20, D), normalized representation matrix
    """

    row_avg = A.mean(axis=0)
    row_std = A.std(axis=0)
    EPS_REG = 1e-10             # regularization param to avoid ZeroDivision
    A_normalized = (A - row_avg) / (row_std + EPS_REG)

    return A_normalized


def RemoveRedundantFeatures(A, remove_threshold=0.9):
    """
    Removes redundant and zero-variance features from a representation matrix A
    inputs:
        A : np.array of shape (20, D), representation matrix
    returns:
        A_noRedundant : np.array of shape (20, D_new), representation matrix with no redundant features
    """

    # calculates the correlation coefficient matrix of the resulting matrix
    corrMat = np.corrcoef(A.T)**2

    # retrieves only the lower triangular part of the symmetric matrix
    lowerTriangular = np.tril(corrMat, k=-1)

    # retrieves one index from all pairs of redundant features
    redundantIndices = np.unique(np.nonzero(lowerTriangular > remove_threshold)[1])

    # removes one feature from each pair of redundant feature
    A_noRedundant = np.delete(A, redundantIndices, axis=1)

    return A_noRedundant


def RemoveZeroVarianceFeatures(A):
    """
    Removes features that are constant across all amino acids, i.e. the ones with zero variance
    inputs :
        A : np.array of shape (20, D), representation matrix
    returns :
        A_noZerovariance : np.array of shape (20, D_new), representation matrix with no zero-variance features
    """

    # calculates the variance of each feature
    feature_std = np.std(A, axis=0)

    # retrieves the indices of all zero-variance features
    zerovarianceIndices = np.nonzero(feature_std == 0)

    # removes all zero-variance features
    A_noZerovariance = np.delete(A, zerovarianceIndices, axis=1)

def GenerateRepresentation(rep_name, input_dir):
    """
    Generates a representation ("onehot", "SLATM", "SPAHM", ""cMBDF, or "Morgan")
    inputs:
        rep_name : str, name of the representation
        input_dir : directory where the .xyz files of the molecules are stored
    returns:
        reps_cleaned_with_gap : np.array of shape (21, D_new), with the "gap representation" included
    """
    
    reps = RepresentationGenerator(rep_name).RepresentationGeneration(input_dir)

    if rep_name in ["SPAHM", "SLATM", "cMBDF"]:
        if rep_name in ["SLATM", "cMBDF"]:
            # removes redundant features from rep. matrix if rep. is "SLATM" or "cMBDF"
            reps = RemoveRedundantFeatures(reps)

        # normalizes the rep. matrix if rep. is "SPAHM", "SLATM" or "cMBDF"
        reps_cleaned = RepsNormalization(reps)
    else:
        # does nothing for other representations
        reps_cleaned = reps

    # removes features with zero variance
    reps_cleaned = RemoveZeroVarianceFeatures(reps_cleaned)

    # retrieves the dimensions of the cleaned rep. matrix
    N = np.shape(reps_cleaned)[0]
    D_new = np.shape(reps_cleaned)[1]

    # adds a row that represents the "gap" and a feature that takes values 0 for 
    # all amino acids and 1 for the "gap"
    reps_cleaned_with_gap = np.zeros((N+1, D_new+1))
    reps_cleaned_with_gap[:N, :D_new] = reps_cleaned
    reps_cleaned_with_gap[-1, -1] = 1

    return reps_cleaned_with_gap




    
