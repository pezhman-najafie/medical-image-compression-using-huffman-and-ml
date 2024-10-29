import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Load the dataset
df = pd.read_csv('Phone.csv')

# Separate features (X) and target variable (y)
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Explore the dataset
df.describe()

# Check for missing values
df.isnull().sum()

# Data preprocessing steps (you can add more based on your analysis)
# For example, scaling features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualize the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()




# Function to evaluate and print model performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Accuracy
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, X_train_scaled, y_train, X_test_scaled, y_test)

# SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
evaluate_model(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Try different values for K
k_values = [3, 5, 7, 9, 11]
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    print(f"\nKNN (K={k}):")
    evaluate_model(knn_model, X_train_scaled, y_train, X_test_scaled, y_test)


# Try different values for C, gamma, and kernel
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto']
kernel_values = ['linear', 'rbf', 'poly']

for C in C_values:
    for gamma in gamma_values:
        for kernel in kernel_values:
            svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
            print(f"\nSVM (C={C}, gamma={gamma}, kernel={kernel}):")
            evaluate_model(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)


# KFold and Cross Validation for KNN
kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=kf)
print("KNN Cross Validation Scores:", knn_scores)

# KFold and Cross Validation for SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=kf)
print("SVM Cross Validation Scores:", svm_scores)

# Binarize the target variable for ROC curve
y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

# ROC Curve for KNN
knn_classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
knn_classifier.fit(X_train_scaled, y_train)
y_probs_knn = knn_classifier.predict_proba(X_test_scaled)
fpr_knn, tpr_knn, _ = roc_curve(y_bin.ravel(), y_probs_knn.ravel())
roc_auc_knn = auc(fpr_knn, tpr_knn)

# ROC Curve for SVM
svm_classifier = OneVsRestClassifier(SVC(kernel='rbf', C=1, gamma='scale'))
svm_classifier.fit(X_train_scaled, y_train)
y_probs_svm = svm_classifier.decision_function(X_test_scaled)
fpr_svm, tpr_svm, _ = roc_curve(y_bin.ravel(), y_probs_svm.ravel())
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

