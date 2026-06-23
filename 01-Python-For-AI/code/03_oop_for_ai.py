"""
Module 01: Python for AI
Topic: Object-Oriented Architecture for Machine Learning

This script demonstrates the skeleton of how a Machine Learning system
is built using Object-Oriented Programming (OOP). We abstract the
Dataset, Model, and Optimizer into classes.
"""

import math
import random

class Dataset:
    """
    Abstracts the data loading and storage process.
    In real ML, this might read from an SQL database or load a folder of images.
    """
    def __init__(self, size):
        self.data = [random.uniform(-1, 1) for _ in range(size)]
        self.labels = [1 if x > 0 else 0 for x in self.data]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LinearModel:
    """
    Abstracts a simple linear equation y = mx + c.
    The 'weights' are 'm' and 'c'.
    """
    def __init__(self):
        self.weight = random.uniform(-0.1, 0.1) # m
        self.bias = 0.0 # c
        
    def forward(self, x):
        """Passes data through the model."""
        return self.weight * x + self.bias
        
    def __repr__(self):
        return f"LinearModel(weight={self.weight:.4f}, bias={self.bias:.4f})"

class Optimizer:
    """
    Abstracts the weight update logic.
    """
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        
    def step(self, model, gradient_w, gradient_b):
        """Updates the model weights based on gradients."""
        model.weight -= self.lr * gradient_w
        model.bias -= self.lr * gradient_b

def train():
    print("Initializing Training Pipeline...")
    dataset = Dataset(size=100)
    model = LinearModel()
    optimizer = Optimizer(learning_rate=0.1)
    
    print(f"Initial Model: {model}")
    
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(dataset)):
            x, y_true = dataset[i]
            
            # 1. Forward Pass
            y_pred = model.forward(x)
            
            # 2. Calculate Loss (Mean Squared Error derivative placeholder)
            error = y_pred - y_true
            total_loss += error ** 2
            
            # 3. Calculate Gradients (Chain rule logic explored in Module 07)
            grad_w = 2 * error * x
            grad_b = 2 * error
            
            # 4. Optimizer Step
            optimizer.step(model, grad_w, grad_b)
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.4f}")
        
    print(f"Trained Model: {model}")

if __name__ == "__main__":
    train()
