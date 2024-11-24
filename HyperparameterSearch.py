from functools import partial
import optuna
from Transformer import AttentionDCA
from train import train_model

"""
Commentaires de Jacques:
- Plutôt pas mal, mais les commentaires en vert donnent l'impression que ChatGPT a tout fait, c'est mieux de les enlever
    j'en ai laissé pour la clarté, dis-moi si tu penses que ça fait trop encore -tim
- Il faut absolument mettre une seed pour optuna, sinon nos résutats ne seront pas reproductibles
"""

def objective_with_params(trial, reps_matrix, sub_data_dict):
    """
    Objective function that trains the AttentionDCA with different combinations of hyperparameters and returns 
    its average testing loss over the cross-validation

    inputs :
        trial : optuna.trial.Trial object (automatically generated in the optuna study)
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        sub_data_dict : dictionnary containing the weight vectors and alignment matrices of training/test sets for each fold of the cross validation
            It has the following structure (N_train/N_test is the number of sequences in the training/test set, M is the number of elements per sequence): 
            {
                "Train": {
                        "W": Weight vector for the training set (torch.tensor of size (N_train,))
                        "Z": Alignment matrix for the training set (torch.tensor of size (N_train, M))
                    },
                "Test": {
                        "W": Weight vector for the test set (torch.tensor of size (N_test,))
                        "Z": Alignment matrix for the test set (torch.tensor of size (N_test, M))
                    },
            }
    returns :
        mean_loss_test : float, average testing loss over the cross-validation
    """

    # Suggest hyperparameters :
    num_heads = trial.suggest_int("num_heads", 1, 8, step=1)
    d_k = trial.suggest_int("d_k", 32, 128, step=16)
    d_v = trial.suggest_int("d_v", 1, 20, step=1)
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "laplace", "linear"])
    lambda_ = trial.suggest_loguniform("lambda_", 1e-4, 1e-2)
    learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 10, 50, step=10)

    # Calculate the length of the MSA sequence
    seq_len = sub_data_dict["Train"]["Z"].shape()[1]

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

    # Run the training loop to optimize the model
    _, loss_test, _ = train_model(model, sub_data_dict, kernel_type, learning_rate, num_epochs)

    return loss_test

def HyperparameterSearch(trial, reps_matrix, data_dict, seed = 10):
    """
    Runs the hyper-parameter search over a hyper-parameter space given in objective_with_params
    inputs :
        trial : optuna.trial.Trial object (automatically generated in the optuna study)
        reps_matrix : np.array of shape (21, D_rep), representations matrix
        data_dict : dictionnary containing the weight vectors and alignment matrices of training/test sets for each fold of the cross validation
            It has the following structure (N_train/N_test is the number of sequences in the training/test set, M is the number of elements per sequence): 
            {
                "Train": {
                    "W": Weight vector for the training set (torch.tensor of size (N_train,))
                    "Z": Alignment matrix for the training set (torch.tensor of size (N_train, M))
                },
                "Test": {
                    "W": Weight vector for the test set (torch.tensor of size (N_test,))
                    "Z": Alignment matrix for the test set (torch.tensor of size (N_test, M))
                },
            }
    returns :
        best_params : dict, of 
    """

    # Define the TPE Optuna sampler and the objective function
    sampler = optuna.samplers.TPESampler(seed=seed)
    objective = partial(objective_with_params(trial, reps_matrix, data_dict))

    # Run the Optuna study
    study = optuna.create_study(sampler = sampler, direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params

    return best_params
