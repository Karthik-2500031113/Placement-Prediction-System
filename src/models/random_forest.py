from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)
from sklearn.ensemble import RandomForestClassifier

def create_model():
    model = RandomForestClassifier(
        n_estimators = 100,
        max_features = 'sqrt',
        random_state = 42,
        oob_score = True
    )
    return model
def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    print("\nDecision Tree Trained Successfully!")
    return model
def evaluate_model(model,X_test,y_test):
    accuracy = model.score(X_test,y_test)
    print("Test Accuracy:" ,accuracy)
    print("OOB Score:",model.oob_score_)

    def main():
        df = load_data()
        print("Original Dataset Shape:")
        print(df.shape)

        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column = "PlacementStatus",
            drop_columns = [
                "StudentID",
                "Salary Package",
                "IsAnomaly"
            ]
        )

        numerical_features, categorical_features = (
            identify_features(X_train)
        )

        one_hot_features = [
            "Gender",
            "City",
            "Stream",

        ]

        ordinal_features = [
            "CollegeTier",
            "CGPA_Tier"
        ]