import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# 1. LOAD DATA
# =========================================================

data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

passenger_ids = test["PassengerId"].copy()


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# Learn ticket group sizes from training data
ticket_counts = data["Ticket"].value_counts()


def feature_engineer(df):

    df = df.copy()

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

    # Group uncommon titles together
    common_titles = ["Mr", "Miss", "Mrs", "Master"]

    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles),
        "Rare"
    )


    # -----------------------------------------------------
    # FAMILY FEATURES
    # -----------------------------------------------------

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["IsChild"] = (df["Age"] < 14).astype(int)


    # -----------------------------------------------------
    # INTERACTION FEATURES
    # -----------------------------------------------------

    df["FemaleFirstClass"] = (
        (df["Sex"] == "female") &
        (df["Pclass"] == 1)
    ).astype(int)

    df["MaleThirdClass"] = (
        (df["Sex"] == "male") &
        (df["Pclass"] == 3)
    ).astype(int)


    # -----------------------------------------------------
    # TICKET
    # -----------------------------------------------------

    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts).fillna(1)

    df["SharedTicket"] = (df["TicketGroupSize"] > 1).astype(int)


    # Extract ticket prefix
    df["TicketPrefix"] = (
        df["Ticket"]
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[\s./]", "", regex=True)
        .replace("", "NONE")
    )


    # -----------------------------------------------------
    # CABIN
    # -----------------------------------------------------

    # First letter roughly represents the deck
    df["Deck"] = df["Cabin"].str[0].fillna("U")


    # -----------------------------------------------------
    # FARE
    # -----------------------------------------------------

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]


    # -----------------------------------------------------
    # DROP UNHELPFUL RAW COLUMNS
    # -----------------------------------------------------

    df = df.drop(
        columns=[
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]
    )

    return df


# Separate target
y = data["Survived"]

X = data.drop("Survived", axis=1)

X = feature_engineer(X)
X_test = feature_engineer(test)


# =========================================================
# 3. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# =========================================================

numeric_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "IsChild",
    "FemaleFirstClass",
    "MaleThirdClass",
    "TicketGroupSize",
    "SharedTicket",
    "FarePerPerson"
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "TicketPrefix",
    "Deck"
]


# =========================================================
# 4. NUMERICAL PREPROCESSING
# =========================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# =========================================================
# 5. CATEGORICAL PREPROCESSING
# =========================================================

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        )
    )
])


# =========================================================
# 6. COMBINE PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])


# =========================================================
# 7. LOGISTIC REGRESSION PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=5000,
            solver="liblinear"
        )
    )
])


# =========================================================
# 8. HYPERPARAMETER SEARCH
# =========================================================

param_grid = {
    "model__C": [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1,
        3,
        10,
        30,
        100
    ],

    "model__penalty": [
        "l1",
        "l2"
    ]
}


cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)


# =========================================================
# 9. TRAIN
# =========================================================

grid.fit(X, y)


print("Best parameters:")
print(grid.best_params_)

print()

print("Best cross-validation accuracy:")
print(grid.best_score_)


# =========================================================
# 10. FINAL MODEL
# =========================================================

model = grid.best_estimator_


# =========================================================
# 11. PREDICT KAGGLE TEST DATA
# =========================================================

predictions = model.predict(X_test)


# =========================================================
# 12. CREATE SUBMISSION
# =========================================================

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions.astype(int)
})

submission.to_csv("submission.csv", index=False)

print()
print(submission.head())