Student-Performance-Analytics-System/
│
├── data/
│   └── student_data.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── model.py
│   ├── prediction.py
│
├── dashboard/
│   └── powerbi_dashboard.pbix
│
├── outputs/
│   └── predictions.csv
│
├── app.py
├── requirements.txt
└── README.mdDATASET (data/student_data.csv)
Csv
StudentID,Attendance,StudyHours,InternalMarks,Assignments,FinalMarks,Result
1,85,5,78,80,82,Pass
2,60,2,45,50,48,Fail
3,90,6,88,85,91,Pass
4,55,1,40,42,38,Fail
5,75,3,65,70,68,Pass
6,62,2,50,55,52,Fail
7,95,7,92,90,94,Pass
8,70,4,60,65,63,Pass
🧹 3. DATA CLEANING (src/data_cleaning.py)
Python
import pandas as pd

def clean_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna()
    df = df.drop_duplicates()

    return df

if __name__ == "__main__":
    df = clean_data("data/student_data.csv")
    df.to_csv("data/cleaned_student_data.csv", index=False)
    print("Data cleaned successfully")
4. MODEL TRAINING (src/model.py)
Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv("data/student_data.csv")

    df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

    X = df[["Attendance", "StudyHours", "InternalMarks", "Assignments"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Model Accuracy:", acc)

    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
5. PREDICTION (src/prediction.py)
Python
import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict(attendance, study_hours, internal_marks, assignments):
    data = np.array([[attendance, study_hours, internal_marks, assignments]])
    result = model.predict(data)

    return "Pass" if result[0] == 1 else "Fail"

if __name__ == "__main__":
    print(predict(70, 3, 60, 65))
 6. STREAMLIT APP (app.py)
Python
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Student Performance Predictor")

attendance = st.number_input("Attendance")
study_hours = st.number_input("Study Hours")
internal_marks = st.number_input("Internal Marks")
assignments = st.number_input("Assignments Marks")

if st.button("Predict Result"):
    data = np.array([[attendance, study_hours, internal_marks, assignments]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("Student will PASS 🎉")
    else:
        st.error("Student will FAIL ⚠️")
7. REQUIREMENTS (requirements.txt)

pandas
numpy
scikit-learn
joblib
streamlit
