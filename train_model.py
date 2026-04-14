import os
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import extract_features_from_image

training_folder = "training_images"

all_data = []

print("Reading training images...\n")

# Load Training Images

for file in os.listdir(training_folder):

    path = os.path.join(training_folder, file)

    print("Processing:", file)

    image = Image.open(path)

    features = extract_features_from_image(image)

    features["file"] = file
    features["group"] = 0

    all_data.append(features)

df = pd.DataFrame(all_data)

print("\nExtracted training data:\n")
print(df)

# Dummy ASD Data

dummy_asd = pd.DataFrame([

{"file":"ASD_1","weight":76,"body_fat":28,"inorganic_salt":3.4,"protein":9.8,"body_water":46,"muscle_mass":52,"skeletal_muscle":28,"BMI":26,"body_fat_rate":28,"visceral_fat":9,"BMR":1680,"fat_free_mass":50,"subcutaneous_fat":23,"SMI":7.4,"WHR":0.92,"group":1},

{"file":"ASD_2","weight":82,"body_fat":31,"inorganic_salt":3.6,"protein":10.1,"body_water":44,"muscle_mass":54,"skeletal_muscle":30,"BMI":27,"body_fat_rate":31,"visceral_fat":10,"BMR":1740,"fat_free_mass":52,"subcutaneous_fat":25,"SMI":7.7,"WHR":0.94,"group":1},

{"file":"ASD_3","weight":70,"body_fat":25,"inorganic_salt":3.3,"protein":9.5,"body_water":48,"muscle_mass":50,"skeletal_muscle":27,"BMI":24,"body_fat_rate":25,"visceral_fat":8,"BMR":1600,"fat_free_mass":48,"subcutaneous_fat":20,"SMI":7.1,"WHR":0.90,"group":1},

{"file":"ASD_4","weight":88,"body_fat":34,"inorganic_salt":3.7,"protein":10.5,"body_water":43,"muscle_mass":55,"skeletal_muscle":31,"BMI":29,"body_fat_rate":34,"visceral_fat":11,"BMR":1790,"fat_free_mass":53,"subcutaneous_fat":27,"SMI":7.9,"WHR":0.96,"group":1},

{"file":"ASD_5","weight":79,"body_fat":29,"inorganic_salt":3.5,"protein":10.0,"body_water":45,"muscle_mass":53,"skeletal_muscle":29,"BMI":26,"body_fat_rate":29,"visceral_fat":9,"BMR":1700,"fat_free_mass":51,"subcutaneous_fat":24,"SMI":7.5,"WHR":0.93,"group":1},

{"file":"ASD_6","weight":73,"body_fat":27,"inorganic_salt":3.4,"protein":9.7,"body_water":47,"muscle_mass":51,"skeletal_muscle":28,"BMI":25,"body_fat_rate":27,"visceral_fat":8,"BMR":1650,"fat_free_mass":49,"subcutaneous_fat":22,"SMI":7.3,"WHR":0.91,"group":1},

{"file":"ASD_7","weight":85,"body_fat":33,"inorganic_salt":3.6,"protein":10.4,"body_water":43,"muscle_mass":55,"skeletal_muscle":31,"BMI":28,"body_fat_rate":33,"visceral_fat":11,"BMR":1770,"fat_free_mass":53,"subcutaneous_fat":26,"SMI":7.8,"WHR":0.95,"group":1},

{"file":"ASD_8","weight":68,"body_fat":24,"inorganic_salt":3.2,"protein":9.3,"body_water":49,"muscle_mass":49,"skeletal_muscle":26,"BMI":23,"body_fat_rate":24,"visceral_fat":7,"BMR":1550,"fat_free_mass":47,"subcutaneous_fat":19,"SMI":6.9,"WHR":0.89,"group":1},

{"file":"ASD_9","weight":91,"body_fat":36,"inorganic_salt":3.8,"protein":10.8,"body_water":42,"muscle_mass":56,"skeletal_muscle":32,"BMI":30,"body_fat_rate":36,"visceral_fat":12,"BMR":1850,"fat_free_mass":54,"subcutaneous_fat":29,"SMI":8.1,"WHR":0.98,"group":1},

{"file":"ASD_10","weight":77,"body_fat":28,"inorganic_salt":3.5,"protein":9.9,"body_water":46,"muscle_mass":52,"skeletal_muscle":29,"BMI":26,"body_fat_rate":28,"visceral_fat":9,"BMR":1690,"fat_free_mass":50,"subcutaneous_fat":23,"SMI":7.4,"WHR":0.92,"group":1},

{"file":"ASD_11","weight":83,"body_fat":32,"inorganic_salt":3.6,"protein":10.2,"body_water":44,"muscle_mass":54,"skeletal_muscle":30,"BMI":27,"body_fat_rate":32,"visceral_fat":10,"BMR":1750,"fat_free_mass":52,"subcutaneous_fat":25,"SMI":7.7,"WHR":0.94,"group":1},

{"file":"ASD_12","weight":72,"body_fat":26,"inorganic_salt":3.4,"protein":9.6,"body_water":47,"muscle_mass":51,"skeletal_muscle":28,"BMI":25,"body_fat_rate":26,"visceral_fat":8,"BMR":1630,"fat_free_mass":49,"subcutaneous_fat":21,"SMI":7.2,"WHR":0.91,"group":1},

{"file":"ASD_13","weight":87,"body_fat":34,"inorganic_salt":3.7,"protein":10.5,"body_water":43,"muscle_mass":55,"skeletal_muscle":31,"BMI":29,"body_fat_rate":34,"visceral_fat":11,"BMR":1780,"fat_free_mass":53,"subcutaneous_fat":27,"SMI":7.9,"WHR":0.96,"group":1},

{"file":"ASD_14","weight":75,"body_fat":27,"inorganic_salt":3.5,"protein":9.8,"body_water":46,"muscle_mass":52,"skeletal_muscle":29,"BMI":26,"body_fat_rate":27,"visceral_fat":9,"BMR":1670,"fat_free_mass":50,"subcutaneous_fat":22,"SMI":7.3,"WHR":0.92,"group":1},

{"file":"ASD_15","weight":90,"body_fat":35,"inorganic_salt":3.8,"protein":10.7,"body_water":42,"muscle_mass":56,"skeletal_muscle":32,"BMI":30,"body_fat_rate":35,"visceral_fat":12,"BMR":1830,"fat_free_mass":54,"subcutaneous_fat":28,"SMI":8.0,"WHR":0.97,"group":1},

{"file":"ASD_16","weight":69,"body_fat":25,"inorganic_salt":3.3,"protein":9.4,"body_water":48,"muscle_mass":50,"skeletal_muscle":27,"BMI":24,"body_fat_rate":25,"visceral_fat":7,"BMR":1580,"fat_free_mass":48,"subcutaneous_fat":20,"SMI":7.0,"WHR":0.90,"group":1},

{"file":"ASD_17","weight":84,"body_fat":32,"inorganic_salt":3.6,"protein":10.3,"body_water":44,"muscle_mass":54,"skeletal_muscle":30,"BMI":28,"body_fat_rate":32,"visceral_fat":10,"BMR":1760,"fat_free_mass":52,"subcutaneous_fat":26,"SMI":7.8,"WHR":0.95,"group":1},

{"file":"ASD_18","weight":71,"body_fat":26,"inorganic_salt":3.4,"protein":9.6,"body_water":47,"muscle_mass":51,"skeletal_muscle":28,"BMI":25,"body_fat_rate":26,"visceral_fat":8,"BMR":1620,"fat_free_mass":49,"subcutaneous_fat":21,"SMI":7.2,"WHR":0.91,"group":1},

{"file":"ASD_19","weight":86,"body_fat":33,"inorganic_salt":3.7,"protein":10.4,"body_water":43,"muscle_mass":55,"skeletal_muscle":31,"BMI":29,"body_fat_rate":33,"visceral_fat":11,"BMR":1780,"fat_free_mass":53,"subcutaneous_fat":27,"SMI":7.9,"WHR":0.96,"group":1},

{"file":"ASD_20","weight":74,"body_fat":27,"inorganic_salt":3.5,"protein":9.7,"body_water":46,"muscle_mass":52,"skeletal_muscle":29,"BMI":26,"body_fat_rate":27,"visceral_fat":9,"BMR":1660,"fat_free_mass":50,"subcutaneous_fat":22,"SMI":7.3,"WHR":0.92,"group":1}

])

final_df = pd.concat([df, dummy_asd], ignore_index=True)

print("\nFinal dataset:\n")
print(final_df)

# Machine Learning

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

X = final_df.drop(columns=["file","group"], errors="ignore")
y = final_df["group"].astype(int)

# Handle missing values
X = X.fillna(X.mean())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Feature Selection

coefficients = model.coef_[0]
feature_names = X.columns

selected_features = [
feature for feature, coef in zip(feature_names, coefficients) if coef != 0
]

print("\nSelected Features:", selected_features)

# Training accuracy
train_accuracy = model.score(X_train_scaled, y_train)

# Testing accuracy
test_accuracy = model.score(X_test_scaled, y_test)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

plt.figure()

labels = ["Training Accuracy", "Testing Accuracy"]
values = [train_accuracy, test_accuracy]

plt.bar(labels, values)

plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.ylabel("Accuracy")

plt.show()

# Cross Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)

print("\nCross-validation scores:", scores)
print("Cross-validated accuracy:", scores.mean())

# Plot per-fold accuracy
plt.figure()

folds = range(1, len(scores) + 1)

plt.plot(folds, scores, marker='o')

plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.ylim(0,1)

plt.show()

# Model Performance

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("\nAccuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

print("\nConfusion Matrix:\n", cm)

# Save Model

joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("\nModel and scaler saved.")