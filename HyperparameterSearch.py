import numpy as np
from functools import partial
import optuna
from Transformer import AttentionDCA
from train import train_model
import torch

def objective_with_params(trial, reps_matrix, data_dict):
    """
    Objective function that trains the AttentionDCA with different combinations of hyperparameters and returns 
    its average testing loss over the cross-validation

    inputs :
        trial : optuna.trial.Trial object (automatically generated in the optuna study)
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        weights
        data_dict : dictionnary containing the weight vectors and alignment matrices of training/test sets for each fold of the cross validation
            It has the following structure (N_train/N_test is the number of sequences in the training/test set, M is the number of elements per sequence): 
            {
                "Fold number": {
                    "Train": {
                        "W": Weight vector for the training set (torch.tensor of size (N_train,))
                        "Z": Alignment matrix for the training set (torch.tensor of size (N_train, M))
                    },
                    "Test": {
                        "W": Weight vector for the test set (torch.tensor of size (N_test,))
                        "Z": Alignment matrix for the test set (torch.tensor of size (N_test, M))
                    },
                },
                ...
            }
    returns :
        average_loss : float, average loss over the cross-validation
    """

    # Suggest hyperparameters :

    # Number of heads of the attention block
    num_heads = trial.suggest_int("num_heads", 1, 8, step=1)

    # Dimensionality of the output spaces of Q and K
    d_k = trial.suggest_int("d_k", 32, 128, step=16)

    # Dimensionality of the output space of V_metric 
    d_v = trial.suggest_int("d_v", 1, 20, step=1)

    # Kernel type
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "laplace", "linear"])

    # L2 regularization parameter for the loss
    lambda_ = trial.suggest_loguniform("lambda_", 1e-4, 1e-2)
    
    # Learning rate
    learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    # Number of epochs
    num_epochs = trial.suggest_int("num_epochs", 10, 50, step=10)

    # Calculate the length of the MSA sequence
    seq_len = data_dict[0]["Train"]["Z"].shape()[1]

    # Initialize model
    model = AttentionDCA(
        reps_matrix=reps_matrix,
        seq_len=seq_len,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        kernel_type=kernel_type,
        lambda_=lambda_,
    )

    losses_test = 0

    # Iterate over all folds
    for i in range(len(data_dict)):
        # Extract the training and testing MSAs from the dict
        _, Z_train = data_dict[f"{i}"]["Train"].values()
        weights_test, Z_test = data_dict[f"{i}"]["Test"].values()

        # Run the training loop to obtain the new weights
        model_opt = train_model(model, Z_train, kernel_type, learning_rate, num_epochs)

        # Add the testing loss of each fold 
        losses_test += model_opt.loss(Z_test, weights_test) 

    # Calculate the mean testing loss and returns it
    mean_loss_test = losses_test / len(data_dict)

    return mean_loss_test

def HyperparameterSearch(trial, reps_matrix, data_dict):
    """
    Runs the hyper-parameter search over a hyper-parameter space given in objective_with_params
    inputs :
        trial : optuna.trial.Trial object (automatically generated in the optuna study)
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        data_dict : dictionnary containing the weight vectors and alignment matrices of training/test sets for each fold of the cross validation
            It has the following structure (N_train/N_test is the number of sequences in the training/test set, M is the number of elements per sequence): 
            {
                "Fold number": {
                    "Train": {
                        "W": Weight vector for the training set (torch.tensor of size (N_train,))
                        "Z": Alignment matrix for the training set (torch.tensor of size (N_train, M))
                    },
                    "Test": {
                        "W": Weight vector for the test set (torch.tensor of size (N_test,))
                        "Z": Alignment matrix for the test set (torch.tensor of size (N_test, M))
                    },
                },
                ...
            }
    returns :
        best_params : dict, of 
    """

    objective = partial(objective_with_params(trial, reps_matrix, data_dict))

    # Run the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Find best parameters
    best_params = study.best_params

    return best_params
