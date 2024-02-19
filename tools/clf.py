'''
create clf using torch
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

class PyTorchClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, model_type, loss_func_type, device, num_classes,
                 epochs=10, batch_size=32,weights=None,
                 lr = 1e-6, weight_decay=0.0):
        self.model_type = model_type
        self.loss_func_type = loss_func_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.weights = weights

        if self.model_type == 'resnet':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes) 
            self.model.to(device)
        else:
            raise NotImplementedError

        if self.loss_func_type == 'CE':
            if weights is None:
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        else:
            raise NotImplementedError
        
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.init_optimizer()

    def init_optimizer(self,):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay =self.weight_decay)
        self.optimizer = optimizer
        return
    
    def load_pretrained_weights(self, state_dict_or_path):
        """
        Load pretrained model weights.
        
        Parameters:
        - state_dict_or_path: Either a path to a saved state_dict file (str) 
                              or a state_dict object.
        """
        if isinstance(state_dict_or_path, str):  # If the argument is a file path
            state_dict = torch.load(state_dict_or_path, map_location=self.device)
        else:
            state_dict = state_dict_or_path  # Assume it's already a state_dict
        
        self.model.load_state_dict(state_dict)

    def fit(self, X_train, y_train, X_val=None, y_val =None):
        '''
        fit function
        you can write early stop by further split the dataset into train and validation
        '''
        # set up dataloader
        dataset = TensorDataset(X_train.type(torch.FloatTensor), y_train.type(torch.LongTensor))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        X_tensor = X.type(torch.FloatTensor) 
        X_tensor = X_tensor.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().detach().numpy()

    def predict_proba(self, X):
        X_tensor = X.type(torch.FloatTensor) 
        X_tensor = X_tensor.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().detach().numpy()


class PyTorchClassifier2DInput(PyTorchClassifier):
    def __init__(self,model_type, loss_func_type, device, num_classes,
                 epochs=10, batch_size=32,weights=None,
                 lr = 1e-6, weight_decay=0.0, img_size = (3,128,128)):
        super(PyTorchClassifier2DInput, self).__init__(model_type, loss_func_type, device, num_classes,
                 epochs, batch_size,weights,
                 lr, weight_decay)
        self.img_size = img_size

    def fit(self, X_train, y_train):
        assert len(X_train.shape) == 2
        X_train = torch.reshape(X_train,(-1,)+self.img_size)

        dataset = TensorDataset(X_train.type(torch.FloatTensor), y_train.type(torch.LongTensor))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        assert len(X.shape) == 2
        X = torch.reshape(X,(-1,)+self.img_size)

        X_tensor = X.type(torch.FloatTensor) 
        X_tensor = X_tensor.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().detach().numpy()

    def predict_proba(self, X):

        assert len(X.shape) == 2
        X = torch.reshape(X,(-1,)+self.img_size)
        
        X_tensor = X.type(torch.FloatTensor) 
        X_tensor = X_tensor.to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().detach().numpy()