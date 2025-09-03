from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Download the data (if not already cached) and load it as NumPy arrays. cache in data dir
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, data_home="data")

# Extract the meaningful parameters of the faces dataset
n_samples, h, w = lfw_people.images.shape

# For machine learning we use the 2D data directly (ignore pixel positions)
X = lfw_people.data
n_features = X.shape[1]

# The label to predict is the ID of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Print dataset summary
print("Total dataset size:")
print(f"n_samples:  {n_samples}")
print(f"n_features: {n_features}")
print(f"n_classes:  {n_classes}")
