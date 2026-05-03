import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-tuning")

# 📥 load data
df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

X = df.drop("Species", axis=1)
y = df["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

model = RandomForestClassifier(random_state=42)

grid = GridSearchCV(model, param_grid, cv=3)

with mlflow.start_run():
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

   
    mlflow.log_param("n_estimators", best_model.n_estimators)
    mlflow.log_param("max_depth", best_model.max_depth)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(best_model, "model")

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)