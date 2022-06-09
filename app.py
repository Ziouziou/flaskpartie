import sklearn.utils
from flask import Flask
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin
from flask_restful.utils.cors import crossdomain
from numpy import array
from sklearn import model_selection
from sklearn.svm import SVC
#Scaling our columns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pickle as p
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
@app.route('/api/', methods=['POST'])
def makecalc():
    #data = request.get_json()
    #json_ = request.json
    #response = np.array2string(model.predict(data))
    #return jsonify(response)
    iris = pd.read_csv("IRIS.csv")
    # splitting the dataset
    x = iris.drop("species", axis=1)
    y = iris["species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=0)
    # training the model
    from sklearn.neighbors import KNeighborsClassifier
    model.fit()
    # giving inputs to the machine learning model
    # features = [[sepal_length, sepal_width, petal_length, petal_width]]
    features = np.array([[5, 2.9, 1, 0.2]])
    # using inputs to predict the output
    prediction = knn.predict(features)
    print("Prediction: {}".format(prediction))

if __name__ == '__main__':
    modelfile = 'voting.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')
