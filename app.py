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

import numpy as np
import pickle as p
import pandas as pa
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
    data = request.get_json()
    json_ = request.json
    response = np.array2string(model.predict(data))
    response.headers.add('Access-Control-Allow-Headers', 'x-csrf-token')

    return jsonify(response)


if __name__ == '__main__':
    modelfile = 'voting.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')
