import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
data = pd.read_csv("student_data.csv")

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓"
)

st.title("🎓 Student Grade Predictor")
st.write(
    "Predict a student's final marks using academic performance data."
)

st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.sidebar.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.sidebar.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=65.0
)

assignment_marks = st.sidebar.number_input(
    "Assignment Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Final Marks"):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_marks": [previous_marks],
        "assignment_marks": [assignment_marks]
    })

    result = model.predict(input_data)

    marks = max(0, min(100, result[0]))

    st.success(
        f"Predicted Final Marks: {marks:.2f}"
    )

    if marks >= 90:
        grade = "A+"
    elif marks >= 80:
        grade = "A"
    elif marks >= 70:
        grade = "B"
    elif marks >= 60:
        grade = "C"
    elif marks >= 50:
        grade = "D"
    else:
        grade = "F"

    st.info(f"Predicted Grade: {grade}")


st.header("📊 Data Visualization")

# Study hours vs marks
fig1, ax1 = plt.subplots()

ax1.scatter(
    data["study_hours"],
    data["final_marks"]
)

ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Final Marks")
ax1.set_title("Study Hours vs Final Marks")

st.pyplot(fig1)

# Attendance vs marks
fig2, ax2 = plt.subplots()

ax2.scatter(
    data["attendance"],
    data["final_marks"]
)

ax2.set_xlabel("Attendance (%)")
ax2.set_ylabel("Final Marks")
ax2.set_title("Attendance vs Final Marks")

st.pyplot(fig2)

# Actual marks
st.subheader("Student Dataset")
st.dataframe(data)