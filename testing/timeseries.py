import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------ STEP 1: GENERATE TIME SERIES ------------------------
def generate_time_series(n_points=2000):
    x = np.linspace(0, 50, n_points)
    y = np.sin(x) + 0.1*np.random.randn(n_points)  # sine wave + noise
    return y


# ------------------------ STEP 2: CREATE SLIDING WINDOWS ------------------------
def create_windows(data, window_size=50):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])  # predict next step
    return np.array(X), np.array(y)


# ------------------------ STEP 3: DATASET CLASS ------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # add channel dimension
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------ STEP 4: 1D CNN MODEL WITH AUTO DIMENSION CALC ------------------------
class CNN1DRegressor(nn.Module):
    def __init__(self, window_size=50):
        super(CNN1DRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # calculate final dimension dynamically
        dummy = torch.zeros(1, 1, window_size)
        out = self.pool(self.relu(self.conv2(self.relu(self.conv1(dummy)))))
        self.flatten_dim = out.numel()

        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # (batch, 32, L1)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, L2)
        x = x.view(x.size(0), -1)        # flatten
        return self.fc(x)                # regression output


# ------------------------ STEP 5: TRAINING LOOP ------------------------
def train_model():
    data = generate_time_series()
    X, y = create_windows(data, window_size=50)

    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN1DRegressor(window_size=50)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    for epoch in range(1000):
        for batch_x, batch_y in dataloader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/20  Loss: {loss.item():.6f}")

    print("Training finished.")
    return model


if __name__ == "__main__":
    trained_model = train_model()
