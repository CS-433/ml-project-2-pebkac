import torch
import torch.optim as optim
import torch.utils.data
import Dataset

def train_model(model, list_ids, file_path_training, file_path_validation, num_epoch, **params):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Loads the data
    training_set = Dataset(list_ids, file_path_training)

    # Generates the batches for the training
    training_generator = torch.utils.data.Dataloader(training_set, **params)

    # Generates the batches for the cross validation
    ### Not necessary if batches already made. Struggling to understand where the batches should be generated
    validation_set = Dataset(list_ids, file_path_validation)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)


    # Sets the adam optimizer
    ### I don't know to what lr should be set
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epoch):
        running_loss = 0.0
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            optimizer.zero_grad()

            # I took the forward as not useful

            # Compute loss
            loss = model.loss(local_batch, local_labels) ## problem should be local weights but i am not sure about dataloader work

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Track loss
            running_loss += loss

        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                validation_loss = 0.0
            # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                loss = model.loss(local_batch, local_labels)

                validation_loss += loss
    return model, validation_loss, running_loss