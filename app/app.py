from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
from utils import PatternTokenizer, toxic_ensemble

app = Flask(__name__)
api = Api(app)

## Load models
model = toxic_ensemble(model_path='lib/models')
tokenizer = PatternTokenizer()

## Parse argument
parser = reqparse.RequestParser()
parser.add_argument('query', action='append')

## Run and output predictions & class probabilities
class Predict(Resource):
    def get(self):
        # use parser to get user's query and conver to Pandas series
        args = parser.parse_args()
        if type(args['query'])!=list:
            user_query = pd.Series([args['query']])
        else:
            user_query = pd.Series(args['query'])

        ## Prediction
        # tokenize the user's query and make a prediction
        user_query = tokenizer.process_ds(user_query).str.join(sep=" ")
        # Make prediction using ensemble model
        preds = model.predict(user_query)
        preds_proba = model.predict_proba(user_query)

        ## Create JSON object as output
        output = {}
        for i,query in enumerate(user_query):
            output['Comment ' + str(i+1)] = {
                'Comment' : query,
                'Prediction' : preds[i],
                'Class Probabilities' : [round(1-preds_proba[i],2), round(preds_proba[i],2)]
            }
        
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=True,
            host="0.0.0.0") # important for remote access (e.g., running inside a docker container)