import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type, loss_func_type, device, num_classes,
                 epochs=10, batch_size=32, weights=None,
                 lr=1e-6, weight_decay=0.0, train_loader=None, test_loader=None):
        self.model_type = model_type
        self.loss_func_type = loss_func_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.weights = weights
        self.train_loader = train_loader
        self.test_loader = test_loader

        if self.model_type == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        else:
            raise NotImplementedError
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model.to(self.device)

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        if self.loss_func_type == 'CE':
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights) if weights is not None else nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def load_pretrained_weights(self, state_dict_or_path):
        if isinstance(state_dict_or_path, str):
            state_dict = torch.load(state_dict_or_path, map_location=self.device)
        else:
            state_dict = state_dict_or_path
        self.model.load_state_dict(state_dict)

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None):
        if self.train_loader is None:
            assert X_train is not None and y_train is not None, "X_train and y_train must be provided if train_loader is not."
            dataset = TensorDataset(X_train.type(torch.FloatTensor), y_train.type(torch.LongTensor))
            self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None and self.test_loader is None:
            val_dataset = TensorDataset(X_val.type(torch.FloatTensor), y_val.type(torch.LongTensor))
            self.test_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        train_losses, val_losses = [], []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / len(self.train_loader))

            if self.test_loader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in self.test_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        output = self.model(X_batch)
                        loss = self.loss_fn(output, y_batch)
                        total_val_loss += loss.item()
                val_losses.append(total_val_loss / len(self.test_loader))

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1] if val_losses else 'N/A'}")

        self.plot_losses(train_losses, val_losses)

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.title('Losses over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        X_tensor = X.type(torch.FloatTensor).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().detach().numpy()

    def predict_proba(self, X):
        X_tensor = X.type(torch.FloatTensor).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = torch.softmax(self.model(X_tensor), dim=1)
        return outputs.cpu().detach().numpy()

