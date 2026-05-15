# Building churn prediction from scratch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


df = pd.read_csv("customer_churn_new.csv")
df = df.drop("CustomerID", axis=1)


df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Recreate X_train and X_test AFter encoding
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)          

print(X_train.select_dtypes(include=["object"]).columns)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
print('Model training complete!")')

print(log_model)


# model's predictions and performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#make predictions
y_pred = log_model.predict(X_test)

#show first 1000000000 predictions
print("First 10 predictions:")
print(y_pred[:10])

# show actual answers for comparison
print("First 10 actual values:")
print(y_test.values[:10])

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#Logistric regression probabilities
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# ROC-AUC Score
log_auc = roc_auc_score(y_test, y_prob_log)
print("Logistic Regression ROC-AUS:", log_auc)

#ROC Curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)


# Random Forest

from sklearn.ensemble import RandomForestClassifier

#create random forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest ROC-AUC:")
print(roc_auc_score(y_test, y_prob_rf))

# i copied this part
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance values
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Print top 10 features
print(feature_importance.head(10))

# Plot top 10 features
top_10 = feature_importance.head(10)

plt.figure(figsize=(8, 5))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances - Random Forest")
plt.gca().invert_yaxis()
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(
    rf_model,
    X_test,
    y_test,
    display_labels=["Stayed", "Churned"]
)

plt.title("Random Forest Confusion Matrix")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_10 = feature_importance.head(10)

plt.figure(figsize=(8, 5))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Driving Customer Churn")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix)
plt.colorbar()
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()










