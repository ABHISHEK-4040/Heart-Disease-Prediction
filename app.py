from flask import Flask,render_template,request,session,redirect,url_for
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
app.secret_key="abhishek6060"

@app.route("/")
def home():
    return render_template("homepage.html")
@app.route("/prediction",methods=['GET','POST'])
def prediction():
    result=""
    try:
        if request.method=="POST":
            age=request.form['age']
            gender=request.form['gender']
            cp=request.form['cp']
            trestbps=request.form['trestbps']
            chol=request.form['chol']
            fbs=request.form['fbs']
            restecg=request.form['restecg']
            thalach=request.form['thalach']
            exang=request.form['exang']
            oldpeak=request.form['oldpeak']
            slope=request.form['slope']
            ca=request.form['ca']
            thal=request.form['thal']
            data=[age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            d=np.float64(data)
            final_input=np.array([d])
            output=model.predict(final_input)[0]
            if output==1:
                result="Presence of heart disease"
            else:
                result="No presence of heart disease"
           
    except Exception as e:
        if e:
            session["er"]="Error: "+ str(e)
            if session["er"]:
               return  redirect(url_for("Error_handling"))
    return render_template("predictionpage.html",result=result)
@app.route("/error")
def Error_handling():
    error=session["er"]
    return render_template("errorpage.html",error=error)
if __name__=='__main__':
    app.run(debug=True)
