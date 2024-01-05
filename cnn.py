import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

class CNN(nn.Module):
    def __init__(self, input_channels, output_size, num_conv_layers, num_fc_layers, fc_units, dropout_rate):
        super(CNN, self).__init__()

        # Convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(input_channels, 64, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Fully connected layers
        fc_layers = []
        fc_layers.append(nn.Flatten())
        prev_fc_units = 64 * (input_channels // (2**num_conv_layers))  # Calculate input size for FC layers
        for i in range(num_fc_layers):
            fc_layers.append(nn.Linear(prev_fc_units, fc_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_rate))
            prev_fc_units = fc_units

        fc_layers.append(nn.Linear(prev_fc_units, output_size))
        
        self.model = nn.Sequential(*conv_layers, *fc_layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def main():
    # Generate synthetic data for binary classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model parameters
    input_channels = 1  # Assuming the input is a 1D signal
    output_size = 2  # Binary classification
    num_conv_layers = 2
    num_fc_layers = 2
    fc_units = 64
    dropout_rate = 0.2
    learning_rate = 0.001
    epochs = 10

    # Reshape input for 1D convolution
    X_train_tensor = X_train_tensor.unsqueeze(1)

    # Create model instance
    model = CNN(input_channels, output_size, num_conv_layers, num_fc_layers, fc_units, dropout_rate)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs)

if __name__ == "__main__":
    main()
