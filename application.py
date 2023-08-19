import pickle
from flask import Flask, render_template, redirect, request, jsonify ,url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen

application = Flask(__name__)
app = application

## import ridge model and stadndard scaler pickel file
ridge_model = pickle.load(open("models/algerian_ridge.pkl","rb"))
Standard_scaler = pickle.load(open("models/algerian_scaler.pkl","rb"))

## route from home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        temp = float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        classes = float(request.form.get("Classes"))
        region = float(request.form.get("Region"))

        new_data = Standard_scaler.transform([[temp,rh,ws,rain,ffmc,dmc,classes,region]])
        resultt = ridge_model.predict(new_data)

        return render_template("home.html",result=resultt[0])

    else :  
        return render_template("home.html")


if __name__=="__main__":
    app.run(host="0.0.0.0")
