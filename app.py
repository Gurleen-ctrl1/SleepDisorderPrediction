from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Define mappings for categorical features
gender_mapping = {"Male": 0, "Female": 1}
bmi_mapping = {"Normal Weight": 0, "Overweight": 1, "Obese": 2}
occupation_mapping = {
    "Accountant": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3, "Manager": 4,
    "Nurse": 5, "Sales Representative": 6, "Salesperson": 7,
    "Scientist": 8, "Software Engineer": 9, "Teacher": 10
}

# Define disorder mapping
disorder_mapping = {0: "No Disorder", 1: "Insomnia", 2: "Sleep Apnea", 3: "Narcolepsy"}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get input values from form
        age = float(request.form["age"])
        gender = gender_mapping[request.form["gender"]]
        occupation = occupation_mapping[request.form["occupation"]]
        sleep_duration = float(request.form["sleep_duration"])
        quality_of_sleep = float(request.form["quality_of_sleep"])
        physical_activity = float(request.form["physical_activity"])
        
        # Validate Physical Activity Level (1-100)
        if not (1 <= physical_activity <= 100):
            return "Error: Physical Activity Level must be between 1 and 100."

        stress_level = float(request.form["stress_level"])
        bmi_category = bmi_mapping[request.form["bmi_category"]]
        heart_rate = float(request.form["heart_rate"])
        daily_steps = float(request.form["daily_steps"])
        systolic_pressure = float(request.form["systolic_pressure"])
        diastolic_pressure = float(request.form["diastolic_pressure"])

        # Prepare features for model
        features = np.array([[age, gender, occupation, sleep_duration, quality_of_sleep,
                              physical_activity, stress_level, bmi_category, heart_rate,
                              daily_steps, systolic_pressure, diastolic_pressure]])

        # Make prediction
        prediction = model.predict(features)[0]
        disorder = disorder_mapping.get(prediction, "Unknown Disorder")

        return render_template("result.html", disorder=disorder)

    except Exception as e:
        return str(e)  # Debugging error message

if __name__ == "__main__":
    app.run(debug=True)
