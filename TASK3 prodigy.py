import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Define the paths to the datasets
cat_dir = 'train/cats'
dog_dir = 'train/dogs'

# Function to load images and labels
def load_data(cat_dir, dog_dir):
    data = []
    labels = []
    
    for category in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, category)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        data.append(img)
        labels.append(0)  # 0 for cats

    for category in os.listdir(dog_dir):
        img_path = os.path.join(dog_dir, category)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        data.append(img)
        labels.append(1)  # 1 for dogs

    return np.array(data), np.array(labels)

# Load data
data, labels = load_data(cat_dir, dog_dir)

def extract_hog_features(data):
    hog_features = []
    for img in data:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features, hog_image = hog(gray_img, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_features.append(features)
    return np.array(hog_features)

# Extract features
features = extract_hog_features(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Visualize some of the results
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB))
    true_label = 'Cat' if y_test[i] == 0 else 'Dog'
    pred_label = 'Cat' if y_pred[i] == 0 else 'Dog'
    ax.set_title(f'True: {true_label}\nPred: {pred_label}')
    ax.axis('off')
plt.tight_layout()
plt.show()
