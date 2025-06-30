# Personality Trait Classification with ML Models (IntelliJ-Compatible)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# STEP 1: Load the Dataset (local file)
df = pd.read_csv("data.csv")

# Dataset Summary
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
df.info()

# STEP 2: Data Cleaning
df['Time_spent_Alone'] = df['Time_spent_Alone'].apply(lambda x: 0 if x == -0 else x)
df['Personality'] = df['Personality'].astype("category")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# STEP 3: Exploratory Data Analysis (Violin + KDE + Correlation + Class Count)

# Violin plots for feature distribution
plt.figure(figsize=(18, 12))
for i, col in enumerate(df.drop("Personality", axis=1).columns, 1):
    plt.subplot(3, 3, i)
    sns.violinplot(x="Personality", y=col, data=df, palette="Set3", inner="quartile")
    plt.title(f"{col} Distribution by Personality")
plt.tight_layout()
plt.suptitle("Violin Plots of Features by Personality", y=1.02)
plt.show()

# KDE plots
plt.figure(figsize=(18, 12))
for i, col in enumerate(df.drop("Personality", axis=1).columns, 1):
    plt.subplot(3, 3, i)
    sns.kdeplot(data=df, x=col, hue="Personality", fill=True, alpha=0.5)
    plt.title(f"KDE Plot: {col}")
plt.tight_layout()
plt.suptitle("KDE Distributions by Personality", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of All Features")
plt.show()

# Target distribution barplot
plt.figure(figsize=(6, 4))
sns.countplot(x="Personality", data=df, palette="pastel")
plt.title("Target Class Distribution: Personality")
plt.xticks([0, 1], labels=["Extrovert", "Introvert"])
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show()

# STEP 4: Feature/Target Separation
X = df.drop("Personality", axis=1)
y = df["Personality"]

# STEP 5: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=537, stratify=y)

# STEP 7: Modeling – Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_lr))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# STEP 8: Modeling – Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.show()

# STEP 9: Modeling – SVM
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\nSVM Report:\n", classification_report(y_test, y_pred_svm))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Oranges')
plt.title("SVM Confusion Matrix")
plt.show()

# STEP 10: Model Accuracy Comparison
print("\nModel Accuracies:")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("SVM:", accuracy_score(y_test, y_pred_svm))
