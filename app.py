from flask import Flask, render_template,request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictPipeline



app = Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train',methods=['GET'])
def trainpage():
    os.system("python main.py")
    return "Successfully trained"
@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            print("test")
            nature_mutation =int(request.form['nature_mutation'])
            print("1")
            code_postal =int(request.form['code_postal'])
            print("2")
            nombre_lots =int(request.form['nombre_lots'])
            print("3")
            code_type_local =float(request.form['code_type_local'])
            print("4")
            surface_reelle_bati =float(request.form['surface_reelle_bati'])
            print("5")
            nombre_pieces =float(request.form['nombre_pieces'])
            print("6")
            surface =float(request.form['surface'])
         
            data = [nature_mutation,code_postal,nombre_lots,code_type_local,surface_reelle_bati,nombre_pieces,surface]
            data = np.array(data).reshape(1, 7)

            print(data)
            
            predict = PredictPipeline()
            predict = predict.predict(data)

            return render_template('results.html', prediction = str(round(predict[0])))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",port= 8080, debug=True)
    #app.run(host="0.0.0.0",port= 8080)
