import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ---------------------------------------------------------
# Create required folders
# ---------------------------------------------------------

os.makedirs("results", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# ---------------------------------------------------------
# 1) Load Dataset from /data/
# ---------------------------------------------------------
data_path = "data/accepted_2007_to_2018Q4.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found in {data_path}. "
        "Place the CSV file inside the /data/ folder."
    )

df = pd.read_csv(data_path)

df = df[['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_title','emp_length',
         'home_ownership','annual_inc','verification_status','loan_status','dti','open_acc','pub_rec',
         'revol_bal','revol_util','total_acc','initial_list_status','application_type','mort_acc',
         'pub_rec_bankruptcies']]

df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]

# ---------------------------------------------------------
# 2) One-hot encoding
# ---------------------------------------------------------
categorical_cols = ['term','grade','sub_grade','emp_title','emp_length',
                    'home_ownership','verification_status','initial_list_status','application_type']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df['loan_status'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})

X = df.drop('loan_status', axis=1).values
y = df['loan_status'].values

# ---------------------------------------------------------
# 3) Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4) Scaling
# ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 5) Build Model
# ---------------------------------------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------------------------------------
# 6) Train
# ---------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.2
)

# ---------------------------------------------------------
# 7) Save Model
# ---------------------------------------------------------
model.save("saved_model/model.h5")

# ---------------------------------------------------------
# 8) Predictions
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

# ---------------------------------------------------------
# 9) Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# 10) Classification Report
# ---------------------------------------------------------
report = classification_report(y_test, y_pred_class)
with open("results/metrics.txt", "w") as f:
    f.write(report)

# ---------------------------------------------------------
# 11) ROC Curve
# ---------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("results/roc_curve.png")
plt.close()

print("Training complete. Model and results saved successfully.")
