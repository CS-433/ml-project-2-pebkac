import os
import numpy as np
import qml2
from qml2.representations.slatm import get_slatm_mbtypes, generate_slatm
from rdkit.Chem.rdmolfiles import MolFromXYZFile
from rdkit.Chem import rdFingerprintGenerator
from qstack import compound, spahm
from tqdm import tqdm
from rdkit import RDLogger

# Enable RDKit logging for debugging
RDLogger.EnableLog('rdApp.error')


class RepresentationGenerator:
    """
    This object serves to generate a given representation for a set of molecules
    To generate a given representation, initialize a RepresentationGenerator object (argument: "onehot", "SLATM", "SPAHM", or "Morgan"), representation that is going to be generated)
    A directory where the .xyz files are stored
    In any case, the generation of a representation gives as an output a numpy array, each row represents a single molecule
    """
    def __init__(self, repType):
        self.repType = repType
        self.repDict = {
            "onehot": self.OneHotRepresentation,
            "SLATM": self.SLATMRepresentation,
            "SPAHM": self.SPAHMRepresentation,
            "Morgan": self.MorganFingerprints,
        }
    
    def OneHotRepresentation(self, input_dir):
        N = len(os.listdir(input_dir))
        labels = np.arange(0, N, 1)
        unique_labels, indices = np.unique(labels, return_inverse=True)
        one_hot = np.eye(len(unique_labels))[indices]
        return one_hot

    
    def SLATMRepresentation(self, input_dir):
        molsList = []
        for fileName in os.listdir(input_dir):
            mol = qml2.Compound(input_dir + "/" + fileName)
            
            with open(input_dir + "/" + fileName, 'r') as file:
                lines = file.read().split('\n')
                numAtoms = int(lines[0])
            
            subList = [mol, numAtoms]
            molsList.append(subList)

        compoundsList = [subList[0] for subList in molsList]

        charges = [compound.nuclear_charges for compound in compoundsList]
        coordinates = [compound.coordinates for compound in compoundsList]

        mbtypes = get_slatm_mbtypes(charges)
        repsList = []
        print("Generating SLATM representations: ")
        for i in tqdm(range(len(compoundsList))):
            rep = generate_slatm(charges[i], coordinates[i], mbtypes)
            repsList.append(rep)

        A = np.row_stack(tuple(repsList))
        return A
    
    
    def SPAHMRepresentation(self, input_dir):
        reps_list = []
        
        print("Generating SPAHM representations: ")
        for fileName in tqdm(os.listdir(input_dir)):
            print(f"fileName = {fileName}")
            mol = compound.xyz_to_mol(input_dir + '/' + fileName, 'def2-svp', charge=0, spin=0)
            rep = spahm.compute_spahm.get_spahm_representation(mol, "lb")[0, :]
            reps_list.append((rep, rep.size))
        
        max_size = max(reps_list, key=lambda t: t[1])[1]
        padded_reps_list = []

        for rep, rep_size in reps_list:
            zeros = np.zeros(max_size - rep_size)
            padded_rep = np.concatenate((rep, zeros), axis=None)
            padded_reps_list.append(padded_rep)
        
        A = np.row_stack(tuple(padded_reps_list))
        return A   
    
    
    def MorganFingerprints(self, input_dir):
            molsList = []
            print("Generating Morgan fingerprints: ")
            for fileName in tqdm(os.listdir(input_dir)):
                print(fileName)
                mol = MolFromXYZFile(fileName)
                molsList.append(mol)

            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
            print(molsList)
            fpList = []
            fpList = list(map(fpgen.GetFingerprint, molsList))

            A = np.row_stack(tuple(fpList))
            return A
    
    
    def RepresentationGeneration(self, input_dir):
        A = self.repDict[self.repType](input_dir)
        return A
    

def RepsNormalization(A):
    """
    Normalizes the matrix of representation A
    inputs:
        A : np.array of shape (21, D)
    returns:
        A_normalized : np.array of shape (21, D)
    """
    row_avg = A.mean(axis=1)
    row_std = A.std(axis=1)
    EPS_REG = 1e-10
    A_normalized = (A - row_avg[:, np.newaxis]) / (row_std[:, np.newaxis] + EPS_REG)

    return A_normalized


def RemoveRedundantFeatures(A, threshold=0.9):
    """
    Removes redundant features of a data set
    inputs:
        A : np.array of shape (21, D)
    returns:
        A_noredundant : np.array of shape (21, D_new)
    """
    corrMat = np.corrcoef(A.T)**2
    lowerTriangular = np.tril(corrMat, k=-1)
    redundantIndices = np.unique(np.nonzero(lowerTriangular > threshold)[1])
    A_noredundant = np.delete(A, redundantIndices, axis=1)

    return A_noredundant


def GenerateRepresentation(rep_name, input_dir):
    """
    Generates a representation ("onehot", "SLATM", "SPAHM" or "Morgan")
    inputs:
        rep_name : str, name of the representation
        input_dir : directory where the .xyz files of the molecules are stored
    returns:
        reps_cleaned_with_gap : np.array of shape (21, D_new), with the "gap representation" included
    """
    reps = RepresentationGenerator(rep_name).RepresentationGeneration(input_dir)
    if rep_name != "Morgan" and rep_name != "onehot":
        reps_cleaned = RemoveRedundantFeatures(RepsNormalization(reps))
    else:
        reps_cleaned = reps

    N = np.shape(reps_cleaned)[0]
    D_new = np.shape(reps_cleaned)[1]

    reps_cleaned_with_gap = np.zeros((N+1, D_new+1))
    reps_cleaned_with_gap[:N, :D_new] = reps_cleaned
    reps_cleaned_with_gap[-1, -1] = 1

    return reps_cleaned_with_gap


print(f'e.g. : GenerateRepresentation("SLATM", "3d_struct_aa_xyz") : {GenerateRepresentation("SLATM", "3d_struct_aa_xyz")}')



    
