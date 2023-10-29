import torch
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import torch.nn as nn
import numpy as np
import json
from model import MyModel
import torchvision.transforms as transforms
from PIL import Image

args = {'batch_size':256, 'ck_epoch':0}
def evaluate(model, criterion_class, criterion_seg, dataloader):
    model.eval()
    total_loss = 0.0
    correct_classifications = 0
    total_samples = 0
    label_predictions = []
    mask_predictions = []
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
            label_predictions.append(predicted)
            mask_predictions.append(outputs_seg)
            correct_classifications += (predicted == labels_class).sum().item()
            total_samples += labels_class.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_classifications / total_samples
    label_predictions = torch.cat(label_predictions, dim=0)
    mask_predictions = torch.cat(mask_predictions, dim=0)
    return avg_loss, accuracy, label_predictions, mask_predictions

# Prepare the test data loader
X_test, Y_test, Z_test = np.load('data/test/X.npy'), np.load('data/test/Y.npy'), np.load('data/test/Z.npy')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
Z_test_tensor = torch.tensor(Z_test, dtype=torch.float32)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor, Z_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
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
model_engine.load_checkpoint('models/',args['ck_epoch'])

# Evaluate the model
test_loss, test_accuracy, label_predictions, mask_predictions = evaluate(model_engine, criterion_class, criterion_seg, test_loader)

# Normalize to [0, 1] if your tensor has values outside this range
tensor_img = mask_predictions.reshape(mask_predictions.shape[0], -1)
tensor_img = (tensor_img - torch.min(tensor_img, dim=1, keepdims=True).values) / (torch.max(tensor_img, dim=1, keepdims=True).values - torch.min(tensor_img, dim=1, keepdims=True).values)

# Move to CPU if necessary
tensor_img = tensor_img.cpu()
label_predictions = label_predictions.cpu()
# Save the masks
for i in range(tensor_img.shape[0]):
    # Convert to PIL image
    one_image = tensor_img[i].reshape(256,256)
    pil_img = transforms.ToPILImage()(one_image)
    # Save the image
    pil_img.save(f'results/masks/{i}.jpg')

# Save the labels
np.savetxt('results/label.txt', label_predictions.numpy().squeeze(), fmt='%f')

# Print the results
test_accuracy = f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%'
print(test_accuracy)
with open('results/metric.txt','w+') as f:
    f.write(test_accuracy)