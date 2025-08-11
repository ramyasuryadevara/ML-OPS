from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("./data")
def load_and_save():
    # Load the data
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # --- Preprocessing ---
    # Check and drop missing values (usually none in this dataset)
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconstruct the processed dataframe
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["MedHouseVal"] = y.reset_index(drop=True)

    # Save preprocessed data
    df_processed.to_csv("data/housing.csv", index=False)
    print("Preprocessed data saved to data/housing.csv")


if __name__ == "__main__":
    load_and_save()
