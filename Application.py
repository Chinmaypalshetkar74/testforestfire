import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application 

##import ridge regressor and standard scaler pickel
Ridge_model=pickle.load(open('models/Ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('models/scaler .pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data=Standard_Scaler.transform([[ Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=Ridge_model.predict(new_scaled_data)
        
        return render_template('Home.html',results=result[0])
    
    else:
        return render_template('Home.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0")

