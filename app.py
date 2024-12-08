import base64
from flask import Flask, request, jsonify
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

app = Flask(__name__)
CORS(app)

# Na��tanie d�t,premenovanie ID
csv_file_path = "data.csv"
df = pd.read_csv(csv_file_path,sep=";")
df = df.rename(columns={"ID": "Id"})

# \----- PREDIKCIA ABSENCIE ZAMESTNANCOV -----/

# Zaradenie predikuj�cich atrib�tov a cie�ov�ho
X = df[['Reason for absence', 'Month of absence', 'Day of the week',
       'Seasons', 'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index']]
y = df["Absenteeism time in hours"]

# Generovanie nov�ch d�t pomocou Bootstrappingu
X_extended = X.sample(n=2000, replace=True, random_state=1)
y_extended = y.sample(n=2000, replace=True, random_state=1)
new_Id = df["Id"].sample(n=2000, replace=True, random_state=1)
new_df = X_extended.copy()
new_df["Id"] = new_Id
new_df["Absenteeism time in hours"] = y_extended

# Skontrolovanie roz��ren�ho datasetu
print(f"New data\nX->{X_extended.shape} | y->{y_extended.shape} | DF->{new_df.shape}")

X = new_df[['Transportation expense', 'Reason for absence', 'Work load Average/day ', 'Day of the week', 'Pet', 'Education']]
y = new_df["Absenteeism time in hours"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model_RF = RandomForestRegressor(random_state=1).fit(X_train, y_train)

y_pred_RF = model_RF.predict(X_test)
r2 = r2_score(y_test, y_pred_RF)
print(f"r2_score = {r2}")

joblib.dump(model_RF, "model_RF.pkl")

# \----- KONIEC PREDIKCIE ABSENCIE ZAMESTNANCOV -----/

labels = ["Very Low Workload","Low Workload","Medium Workload","High Workload","Very High Workload"]
bins = [205,244,264,294,320,379]
rounded_workload = new_df["Work load Average/day "].round(0)
rounded_workload = np.int64(rounded_workload)
new_workload = pd.cut(rounded_workload,bins=bins,labels=labels)
new_df['Workload Category'] = new_workload

labels = ["80-85%", "86-90%", "91-95%", "96-100%"]
bins = [80, 85, 90, 95, 100]
rounded_hitTarget = new_df["Hit target"].round(0)
rounded_hitTarget = np.int64(rounded_hitTarget)
new_hitTarget = pd.cut(rounded_hitTarget, bins=bins, labels=labels, include_lowest=True)
new_df['Hit target Category'] = new_hitTarget

new_df["Workload Category"] = new_df["Workload Category"].astype(str)
new_df["Hit target Category"] = new_df["Hit target Category"].astype(str)

# Zabezpe�enie spr�vneho poradia st�pcov
new_df = new_df[['Id','Transportation expense','Reason for absence','Month of absence','Day of the week','Seasons', 
                 'Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Disciplinary failure', 
                 'Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index','Workload Category', 
                 'Hit target Category','Absenteeism time in hours']]

# Konverzia na JSON
json_file_path = "data.json"
new_df.to_json("data.json", orient="records", indent=4)

# V�pis
print(f"The data has been stored as JSON.: {json_file_path}")

# Na��tanie d�t z .json s?boru
with open("data.json") as f:
    data = json.load(f)

@app.route("/Data", methods=["GET"])
def get_students():
    return jsonify(data), 200


@app.route("/Prediction", methods=["POST"])
def predict():
    print("Request received on /Prediction") 
    try:
        data = request.get_json()  # Skontroluj JSON 
        features = [
            data["Transportation"],
            data["Reason"],
            data["Workload"],
            data["Day"],
            data["Pet"],
            data["Education"]
        ]

        predicted_absenteeism = model_RF.predict([features])[0]
        predicted_absenteeism = int(round(predicted_absenteeism,0))
        return jsonify({"PredictedAbsenteeism": predicted_absenteeism}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


    import base64
from io import BytesIO
from flask import Response

from io import BytesIO

@app.route("/VisualizationWorkload", methods=["GET"])
def prediction_image_base64_workload():
    try:
        sns.set()
        plt.figure(figsize=(15, 10))
        workload_order = ["Very Low Workload", "Low Workload", "Medium Workload", "High Workload", "Very High Workload"]
        ax = sns.countplot(data=new_df, x="Workload Category", order=workload_order)
        plt.title("Daily average workload in categories")
        plt.ylabel("Number of employees")

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='black',
                        xytext=(0, 5), textcoords='offset points')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')
        return jsonify({"image": base64_image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



from io import BytesIO

@app.route("/VisualizationHitTarget", methods=["GET"])
def prediction_image_base64_hit_target():
    try:
        sns.set()
        plt.figure(figsize=(15, 10))

        # Zabezpe�enie spr�vneho poradia kateg�ri�
        hit_target_order = ["80-85%", "86-90%", "91-95%", "96-100%"]
        ax = sns.countplot(data=new_df, x="Hit target Category", order=hit_target_order)
        
        plt.title("Hit target in categories")
        plt.ylabel("Number of tasks")

        # Pridanie hodn�t nad st�pce
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='black',
                        xytext=(0, 5), textcoords='offset points')
        
        # Ulo�enie obr�zka do pam�te ako Base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        # K�dovanie do Base64
        base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')
        return jsonify({"image": base64_image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)