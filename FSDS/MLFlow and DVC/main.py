# Importing Libraries
from flask import Flask, request, Response
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.method == 'POST':
            if request.json is not None:
                test_case = request.json['testCase']
                filename = 'model.sav'
                loaded_model = pickle.load(open(filename, 'rb'))
                prediction = loaded_model.predict(test_case)

                return Response("The prediction for the test case is "+ str(prediction))
        else:
            print('Incorrect Method or Input')
    except Exception as e:
        return Response("Error Occurred! %s" %e)

if __name__ == '__main__':
    app.run(debug=True)
