import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# set up device at top of file
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# load faces dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, data_home="data")

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print(f"n_samples:  {n_samples}")
print(f"n_features: {X.shape[1]}")
print(f"n_classes:  {n_classes}")

# move to torch right away
X = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.long, device=device)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.25, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)

n_components = 150

# swap mean-centering to torch
mean = torch.mean(X_train, dim=0, keepdim=True)
X_train = X_train - mean
X_test = X_test - mean

# swap SVD to torch
U, S, V = torch.linalg.svd(X_train, full_matrices=False)

# take top components
components = V[:n_components]
eigenfaces = components.reshape(n_components, h, w)

# swap projection into PCA subspace to torch @
X_transformed = X_train @ components.T
print(X_transformed.shape)
X_test_transformed = X_test @ components.T
print(X_test_transformed.shape)

# bring eigenfaces back to numpy for plotting
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces.detach().cpu().numpy(), eigenface_titles, h, w)
plt.show()

# swap explained variance to torch
explained_variance = (S ** 2) / (X_train.shape[0] - 1)
total_var = torch.sum(explained_variance)
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0)
print(ratio_cumsum.shape)

# plotting needs numpy so detach+cpu
eigenvalueCount = np.arange(n_components)
plt.plot(eigenvalueCount, ratio_cumsum[:n_components].detach().cpu().numpy())
plt.title('Compactness')
plt.show()
