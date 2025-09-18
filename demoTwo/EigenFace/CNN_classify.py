import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

# -----------------------------
# Device setup
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# -----------------------------
# Load dataset
# -----------------------------
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, data_home="data")
X = lfw_people.images
Y = lfw_people.target
n_classes = len(lfw_people.target_names)

print("X_min:", X.min(), "X_max:", X.max())

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# add channel dimension for CNN
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)

# torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)

# -----------------------------
# CNN Model
# -----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            32 * (X_train.shape[2] // 2) * (X_train.shape[3] // 2), 128
        )
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNClassifier(n_classes).to(device)

# -----------------------------
# Optimizer and loss
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training loop
# -----------------------------
epochs = 15
batch_size = 32
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, preds = torch.max(outputs, 1)
    acc = (preds == y_test).float().mean().item()
    print("Test Accuracy:", acc)

# -----------------------------
# Interactive Random Prediction Viewer
# -----------------------------
target_names = lfw_people.target_names

# matplotlib setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplots_adjust(bottom=0.2)

def show_random_prediction(event=None):
    # pick a random test index
    idx = random.randint(0, X_test.shape[0] - 1)
    img = X_test[idx].cpu().numpy().squeeze()
    true_label = y_test[idx].item()

    # run model prediction
    with torch.no_grad():
        output = model(X_test[idx:idx+1])
        pred = torch.argmax(output, 1).item()

    # clear plots
    ax1.cla()
    ax2.cla()

    # show test image + prediction
    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"Predicted: {target_names[pred]}")
    ax1.axis("off")

    # show another real image of actual person
    same_person_indices = np.where(Y == true_label)[0]
    other_idx = random.choice(same_person_indices)
    ax2.imshow(lfw_people.images[other_idx], cmap="gray")
    ax2.set_title(f"Actual: {target_names[true_label]}")
    ax2.axis("off")

    plt.draw()

# button widget
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, "Next Sample")
button.on_clicked(show_random_prediction)

# show first prediction
show_random_prediction()
plt.show()
