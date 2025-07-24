import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data_dir = r"C:\Users\SKILLCRAFT\TASK 4\leapGestRecog\leapGestRecog"

X = []
y = []

for subject in os.listdir(data_dir):
    subject_path = os.path.join(data_dir, subject)
    if os.path.isdir(subject_path):
        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)
            if os.path.isdir(gesture_path):
                for img_name in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))  # Resize to fixed size
                        X.append(img.flatten())
                        y.append(gesture)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
