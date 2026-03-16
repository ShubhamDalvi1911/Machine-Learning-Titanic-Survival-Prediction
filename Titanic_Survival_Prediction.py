import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#-----------------------------------------------------------
# Function name : DisplayInfo
# Description : Display formatted title
#-----------------------------------------------------------
def DisplayInfo(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)


#-----------------------------------------------------------
# Function name : LoadPreservedModel
# Description : Load saved model
#-----------------------------------------------------------
def LoadPreservedModel(filename):

    loaded_model = joblib.load(filename)

    print("Model successfully loaded")

    return loaded_model


#-----------------------------------------------------------
# Function name : PreserveModel
# Description : Save trained model
#-----------------------------------------------------------
def PreserveModel(model, filename):

    joblib.dump(model, filename)

    print("\nModel preserved successfully")


#-----------------------------------------------------------
# Function name : ShowData
# Description : Display dataset information
#-----------------------------------------------------------
def ShowData(df, message):

    DisplayInfo(message)

    print("\nFirst 5 rows of dataset : ")
    print(df.head())

    print("\nShape of dataset : ")
    print(df.shape)

    print("\nColumn names : ")
    print(df.columns.tolist())

    print("\nMissing values in each column : ")
    print(df.isnull().sum())


#-----------------------------------------------------------
# Function name : PlotCorrelationHeatmap
# Description : Display feature correlation heatmap
#-----------------------------------------------------------
def PlotCorrelationHeatmap(df):

    print("\nGenerating Correlation Heatmap...")

    correlation_matrix = df.corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")

    plt.savefig("correlation_heatmap.png")

    plt.show()


#-----------------------------------------------------------
# Function name : CleanTitanicData
# Description : Data preprocessing
#-----------------------------------------------------------
def CleanTitanicData(df):

    DisplayInfo("Step 2 : Original Data")

    print(df.head())

    # Remove unnecessary columns
    drop_columns = ["Passengerid", "zero", "Name", "Cabin"]

    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\nColumns to be dropped : ")
    print(existing_columns)

    df = df.drop(columns=existing_columns)

    DisplayInfo("Step 2 : Data after column removal")

    print(df.head())

    # Handle Embarked column
    if "Embarked" in df.columns:

        print("\nEmbarked column before preprocessing")

        print(df["Embarked"].head(10))

        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        df["Embarked"] = df["Embarked"].replace(['nan','None',''], np.nan)

        embarked_mode = df["Embarked"].mode()[0]

        print("\nMode of Embarked column :", embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\nEmbarked column after preprocessing :")

        print(df["Embarked"].head(10))

    print("\nMissing values after preprocessing")

    print(df.isnull().sum())

    # One Hot Encoding
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    print("\nData after encoding")

    print(df.head())

    print("Shape of dataset :", df.shape)

    # Convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nFinal cleaned dataset")

    print(df.head())

    print("Shape of dataset :", df.shape)

    return df


#-----------------------------------------------------------
# Function name : TrainTitanicModel
# Description : Train ML model
#-----------------------------------------------------------
def TrainTitanicModel(X_train, Y_train):

    DisplayInfo("Step 3 : Training Machine Learning Model")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, Y_train)

    print("\nModel trained successfully")

    print("\nIntercept of model :")

    print(model.intercept_)

    print("\nCoefficients of model :")

    for feature, coef in zip(X_train.columns, model.coef_[0]):
        print(feature, ":", coef)

    return model


#-----------------------------------------------------------
# Function name : EvaluateModel
# Description : Evaluate trained model
#-----------------------------------------------------------
def EvaluateModel(model, X_test, Y_test):

    DisplayInfo("Step 4 : Model Evaluation")

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("\nModel Accuracy :", accuracy * 100)

    cm = confusion_matrix(Y_test, Y_pred)

    print("\nConfusion Matrix :")

    print(cm)


#-----------------------------------------------------------
# Function name : TitanicLogistic
# Description : ML Pipeline Controller
#-----------------------------------------------------------
def TitanicLogistic(DataPath):

    DisplayInfo("Step 1 : Loading the dataset")

    df = pd.read_csv(DataPath)

    ShowData(df, "Initial Dataset")

    df = CleanTitanicData(df)

    PlotCorrelationHeatmap(df)

    DisplayInfo("Step 3 : Feature and Label Separation")

    X = df.drop("Survived", axis=1)

    Y = df["Survived"]

    print("\nFeatures :")

    print(X.head())

    print("\nLabels :")

    print(Y.head())

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("\nTrain Test Split")

    print("X_train shape :", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Y_train shape :", Y_train.shape)
    print("Y_test shape :", Y_test.shape)

    model = TrainTitanicModel(X_train, Y_train)

    PreserveModel(model, "Titanic.pkl")

    loaded_model = LoadPreservedModel("Titanic.pkl")

    EvaluateModel(loaded_model, X_test, Y_test)


#-----------------------------------------------------------
# Function name : main
# Description : Starting point of program
#-----------------------------------------------------------
def main():

    TitanicLogistic("TitanicDataset.csv")


if __name__ == "__main__":
    main()