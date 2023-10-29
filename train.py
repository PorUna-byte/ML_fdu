import torch
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import numpy as np
import torch.nn as nn
import json
from model import MyModel
args={'batch_size':256, 'num_epochs':150}

def evaluate(model, criterion_class, criterion_seg, dataloader):
    model.eval()
    total_loss = 0.0
    correct_classifications = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels_class, labels_seg in dataloader:
            # Move data to the appropriate device
            inputs, labels_class, labels_seg = inputs.to(model.local_rank), labels_class.to(model.local_rank), labels_seg.to(model.local_rank)

            # Forward pass
            outputs_class, outputs_seg = model(inputs)

            # Calculate loss
            loss_class = criterion_class(outputs_class, labels_class)
            loss_seg = criterion_seg(outputs_seg, labels_seg)
            loss = loss_class + loss_seg
            total_loss += loss.item()

            # Calculate accuracy for classification
            predicted = outputs_class > 0.5
            correct_classifications += (predicted == labels_class).sum().item()
            total_samples += labels_class.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_classifications / total_samples

    return avg_loss, accuracy

X_train, Y_train, Z_train = np.load('data/train/X.npy'), np.load('data/train/Y.npy'), np.load('data/train/Z.npy')
X_val, Y_val, Z_val = np.load('data/val/X.npy'), np.load('data/val/Y.npy'), np.load('data/val/Z.npy')
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Z_train_tensor = torch.tensor(Z_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)
Z_val_tensor = torch.tensor(Z_val, dtype=torch.float32)
# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, Z_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor, Z_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
# Configure DeepSpeed
deepspeed.init_distributed()

# Load DeepSpeed configuration file manually
with open('deepspeed_config.json') as f:
    config_file = json.load(f)

# Instantiate the model, loss functions
model = MyModel()
model_engine, model, _, _ = deepspeed.initialize(model=model, config_params=config_file)
criterion_class = nn.BCELoss()
criterion_seg = nn.BCELoss()
best_accuracy = 0
# Train the model
for epoch in range(args['num_epochs']):
    running_loss = 0.0
    for inputs, labels_class, labels_seg in train_loader:
        inputs, labels_class, labels_seg = inputs.to(model_engine.local_rank), labels_class.to(model_engine.local_rank), labels_seg.to(model_engine.local_rank)
        model_engine.zero_grad()
        outputs_class, outputs_seg = model_engine(inputs)
        loss_class = criterion_class(outputs_class, labels_class)
        loss_seg = criterion_seg(outputs_seg, labels_seg)
        loss = loss_class + loss_seg
        model_engine.backward(loss)
        model_engine.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')
    if epoch%10 == 0:
        _, accuracy = evaluate(model_engine, criterion_class, criterion_seg, val_loader)
        if accuracy > best_accuracy:
            print('save model')
            best_accuracy = accuracy
            model_engine.save_checkpoint('models/', 0)
        model_engine.train()

print('Finished Training')