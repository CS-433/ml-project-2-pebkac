import torch
import torch.optim as optim
import torch.utils.data
import Dataset
import tqdm 

"""
params = {'batch_size': 1,
'shuffle': True,
'num_workers': 6}
"""

def train_model(model, list_ids, file_path_training, file_path_validation, num_epoch, lr, verbose, **params):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    """A quoi sert cette ligne ?
    Réponse :
    Si vos tailles d'entrée (par ex. taille des batches) ne changent pas fréquemment au cours de l'entraînement, activer cudnn.benchmark = True permet à cuDNN de rechercher automatiquement la méthode de calcul la plus rapide et de la réutiliser pour optimiser les performances.
Cela est particulièrement utile pour des modèles avec des tailles d'entrée fixes, comme dans des images de taille constante ou des séquences de longueurs identiques."""
    torch.backends.cudnn.benchmark = True
    
    # Loads the data
    training_set = Dataset(list_ids, file_path_training)

    # Generates the batches for the training
    training_generator = torch.utils.data.Dataloader(training_set, **params)

    # Generates the batches for the cross validation
    ### Not necessary if batches already made. Struggling to understand where the batches should be generated
    validation_set = Dataset(list_ids, file_path_validation)
    """Inutile, le but c'est de tester sur tout le validation set, pas sur des mini-batchs du validation set
    Réponse : on peut de toute facon faire un seul batch. Si la validation przend trop de temps on peut faire plus de batch pour parallelisation"""
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)


    # Sets the adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        """A quoi sert running loss ?
        Reponse : keep track. Toujours un truc bon a avoir + pour plot"""
        running_loss = 0.0
        """Quels labels ? C'est de l'apprentissage non-supervisé ??
        Reponse : my bad, c'était local_weights"""
        for local_batch, local_weights in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False):
        
            # Transfer to GPU
            local_batch, local_weights = local_batch.to(device), local_weights.to(device)

            optimizer.zero_grad()

            # Compute loss
            """Si tu ne sais pas comment dataloader fonctionne, le mieux c'est de voir des exemples sur internet ou de tester ton code"""
            loss = model.loss(local_batch, local_weights) ## problem should be local weights but i am not sure about dataloader work

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            running_loss += loss.item()


        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for local_batch, local_weights in tqdm(validation_generator, desc="Validation", leave=False):
                # Transfer to GPU
                local_batch, local_weights = local_batch.to(device), local_weights.to(device)

                # Model computations
                loss = model.loss(local_batch, local_weights)

                validation_loss += loss.item()

        running_loss /= len(training_generator)
        validation_loss /= len(validation_generator)

        if verbose:
            tqdm.write(f"Epoch {epoch+1}: Training Loss = {running_loss:.4f}, Validation Loss = {validation_loss:.4f}")

    return model, validation_loss, running_loss
