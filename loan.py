import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 1: Load datasets from ZIP 
with zipfile.ZipFile("/Users/sirishathiragabattina/Desktop/Mini project/archive.zip") as z:
    with z.open("loan-train.csv") as train_file:
        train_df = pd.read_csv(train_file)
    with z.open("loan-test.csv") as test_file:
        test_df = pd.read_csv(test_file)

# Step 2: Prepare data (use train_df for training)
data = train_df.copy()

# Impute missing numerical values with mean
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
imputer_num = SimpleImputer(strategy='mean')
data[num_cols] = imputer_num.fit_transform(data[num_cols])

# Impute missing categorical values with most frequent
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

# Encode categorical columns
label_enc = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_columns:
    data[col] = label_enc.fit_transform(data[col].astype(str))

# Encode target variable Loan_Status
data['Loan_Status'] = label_enc.fit_transform(data['Loan_Status'])

# Step 3: Define features and target
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# Step 4: Split train/test from train_df
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Validate model
y_pred = model.predict(X_val)
joblib.dump(model, "loan_eligibility_model.pkl")
print("Model saved as loan_eligibility_model.pkl")


print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# Confusion matrix visualization
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("Hi")
# Step 7: Save the trained model

