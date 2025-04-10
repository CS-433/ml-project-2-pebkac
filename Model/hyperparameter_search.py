"""
This script defines the functions used for hyperparameter optimization
"""

from functools import partial
import optuna
import torch
from Model.AttentionDCA import AttentionDCA
from Model.train import train_model

def objective_with_params(trial, reps_matrix, data_dict, struct_file, verbose=False, seed=10):
    """
    Objective function that trains the AttentionDCA with different combinations of hyperparameters and returns 
    its average testing loss over the cross-validation

    inputs :
        trial : optuna.trial.Trial object (automatically generated in the optuna study)
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        data_dict : dictionnary generated by the function read_fasta_k_folds in read_fasta_utils.py
        struct_file: str, name of the file where the true structure of the protein family is stored
        params : dict, the pârameters for the dataloader function. looks like 
            params = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 6
            }
        verbose : bool, should the validation loss and running loss be displayed (default: False)
        seed (int): Random seed for the training process (default: 10)

    returns :
        mean_loss_test : float, average testing loss over the cross-validation
    """

    # Suggest hyperparameters :
    num_heads = trial.suggest_int("num_heads", 64, 256, step=16)
    d_k = trial.suggest_int("d_k", 10, 30)
    d_v = trial.suggest_int("d_v", 2, 15)
    kernel_type = "rbf"
    gamma_fact = trial.suggest_float("gamma_fact", 1e-1, 10, log=True)
    lambda_ = trial.suggest_float("lambda_", 1e-5, 1e-1, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  
    num_epochs = trial.suggest_int("num_epochs", 100, 200, step=25) 

    seq_len = data_dict["Z"].shape[0]
    reps_matrix = torch.tensor(reps_matrix, dtype=torch.float32)
    if torch.cuda.is_available():
        reps_matrix = reps_matrix.to(torch.device("cuda"))

    # Definition of dataloader parameters
    dataloader_params = {
        "batch_size": data_dict["Z"].shape[1]//40,
        "shuffle": True,
        "num_workers": 10,
    }

    # Initialize model
    model = AttentionDCA(
        reps_matrix=reps_matrix,
        seq_len=seq_len,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        kernel_type=kernel_type,
        gamma_fact=gamma_fact,
        lambda_=lambda_,
    )

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    # Run the training loop to optimize the model
    _, loss_test, _ = train_model(model, data_dict, struct_file, num_epochs, lr, dataloader_params, verbose=verbose, seed=seed)

    return loss_test

def HyperparameterSearch(reps_matrix, data_dict, struct_file, verbose=False, seed=10):
    """
    Runs the hyper-parameter search over a hyper-parameter space given in objective_with_params

    inputs :
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        data_dict : dictionnary generated by the function read_fasta_k_folds in read_fasta_utils.py
        struct_file: str, name of the file where the true structure of the protein family is stored
    returns :
        best_params : dict of the form {name_param_1 : best_param_1, ..., name_param_D : best_param_D}
    """

    # Define the TPE Optuna sampler and the objective function
    sampler = optuna.samplers.TPESampler(seed=seed)

    objective = partial(objective_with_params, reps_matrix=reps_matrix, data_dict=data_dict, struct_file=struct_file, verbose=verbose)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=200, timeout=10800)
    best_trial = study.best_trial
    best_params = best_trial.params

    return best_params
