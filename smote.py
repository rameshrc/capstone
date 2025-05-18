import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# File paths
input_path = "data/credit_record.csv"  # Update with the correct path
output_path = "updated_credit_record_smote.csv"

# Load the dataset
print("Loading dataset...")
credit_df = pd.read_csv(input_path)

# Apply the transformation to map STATUS values
print("Transforming 'STATUS' values...")
credit_df.loc[credit_df["STATUS"] == "C", "STATUS"] = "Good_Debt"
credit_df.loc[credit_df["STATUS"] == "X", "STATUS"] = "Good_Debt"
credit_df.loc[credit_df["STATUS"] == "0", "STATUS"] = "Good_Debt"
credit_df.loc[credit_df["STATUS"] == "1", "STATUS"] = "Bad_Debt"
credit_df.loc[credit_df["STATUS"] == "2", "STATUS"] = "Bad_Debt"
credit_df.loc[credit_df["STATUS"] == "3", "STATUS"] = "Bad_Debt"
credit_df.loc[credit_df["STATUS"] == "4", "STATUS"] = "Bad_Debt"
credit_df.loc[credit_df["STATUS"] == "5", "STATUS"] = "Bad_Debt"

# Verify transformation
print("\nTransformed 'STATUS' Distribution:\n", credit_df['STATUS'].value_counts())

# Separate features and target
X = credit_df.drop(columns=["STATUS"])
y = credit_df["STATUS"]

# Apply SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Combine the balanced dataset
smote_data = pd.concat([X_smote, y_smote], axis=1)

# Verify new class distribution
print("\nNew 'STATUS' Distribution After SMOTE:\n", y_smote.value_counts())

# Save the dataset
print(f"\nSaving balanced dataset to '{output_path}'...")
smote_data.to_csv(output_path, index=False)
print(f"SMOTE applied and dataset saved as '{output_path}' successfully.")
