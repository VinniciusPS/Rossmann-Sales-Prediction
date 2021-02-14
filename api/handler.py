import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load( open( 'C:\\Users\\Usuário\\Documents\\repos\\DataScience_Em_Producao_Treino\\model\\model_rossman.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

# endpoint
@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame( test_json, test=[0] )
            
        else:
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys()) # multiple example
        
        # Instantiate Rossmann Class
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
    else:
        return Response( '{}', status=200, minetype='application/json' )
    

if __name__ == '__main__':
    app.run('localhost')