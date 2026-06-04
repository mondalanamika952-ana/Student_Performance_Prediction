import pandas as pd
df = pd.read_csv("student_performance_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Result"] = le.fit_transform(df["Result"])
X = df[[
    "Study_Hours",
    "Attendance",
    "Previous_Score"
]]

y = df["Result"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#Import Model
from sklearn.tree import DecisionTreeClassifier
#Train Model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)
#Make Predictions
predictions = model.predict(X_test)

print(predictions)

#Evaluate Model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)