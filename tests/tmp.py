import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import os

# Define the ClassificationModel class (from your provided code)

class ClassificationModel:
    """
    Example classifier
    """
    def __init__(self):
        self.model = None
        
    def load_model_pickle(self, filename, path="../models/"):
        """Load model parameters from a file using pickle."""
        print(f'Loading a classifier model...')
        try:
            model = torch.load(path + filename + '.pkl')
            self.model = model
        except FileNotFoundError:
            print('Model not found')
            self.model = None

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def reset(self, input_size, hidden):
        print(f'Creating a new classifier model...')
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def _training_loop(self, dataloader, n_epochs, learning_rate):
        # use the AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        # use the Binary Cross Entropy loss
        criterion = nn.BCELoss()
        
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            num_elements = 0
            
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()

                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_elements += X_batch.shape[0]
            epoch_loss = running_loss / num_elements
            pbar.set_description(f'Epoch: {epoch+1} | Loss: {epoch_loss:.5f}')
        
    def train(self, dataloader, n_epochs=200, learning_rate=0.1, 
              model_name="classifier_ddpm", path="../models/"):
        self._training_loop(dataloader, n_epochs, learning_rate)

        x = dataloader.dataset.dataset.tensors[0]
        y = dataloader.dataset.dataset.tensors[1]
        # test the model
        y_pred = self.model(x)

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = np.array(y_class == y).astype(float).mean()
        dummy_acc = max(y.mean().item(), 1 - y.mean().item())
        acc = accuracy.item()
        usefulness = max([0, (acc - dummy_acc) / (1 - dummy_acc)])
        print(f'Dummy accuracy = {dummy_acc:.1%}')
        print(f'Accuracy on test data = {acc:.1%}')
        print(f'Usefulness = {usefulness:.1%}')

        if not os.path.exists(path):
            os.makedirs(path)
        # save the model
        torch.save(self.model, path + model_name + '.pkl')

# Generate some dummy data
x = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100, 1)).float()  # 100 samples, binary labels
print(y[0:5])

# Create a TensorDataset and DataLoader
dataset = TensorDataset(x, y)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize, train, and save the model
model = ClassificationModel()
model.reset(input_size=10, hidden=5)
model.train(train_loader, n_epochs=10, learning_rate=0.01, model_name="test_classifier")

# Load the model and make a prediction
model.load_model_pickle('test_classifier')
example_datapoint = np.random.rand(5, 10)
example_datapoint_tensor = torch.tensor(example_datapoint, dtype=torch.float32)#.unsqueeze(0)

model.model.eval()
with torch.no_grad():
    prediction = model(example_datapoint_tensor)
    predicted_class = (prediction > 0.5).float()
    y = np.round(prediction.numpy().flatten())
    percentage_true = y.mean()

print(f"Prediction: {prediction}")
print(f'y: {y} ')
print(f"Percentage of True: {percentage_true:.1%}")
print(f"Predicted class: {predicted_class.numpy().squeeze()}")


# # generate a random numpy array of size 5, 10
# np.random.seed(0)
# x = np.random.rand(5, 10)
# x = torch.tensor(x, dtype=torch.float32)
