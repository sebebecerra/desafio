#Install Libraries
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
application = Flask(__name__)
@application.route('/prediction', methods=['POST'])
#define function
def predict():
    if xgboost:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=xgboost_columns, fill_value=0)
            test_keys = ['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada', 'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines', 'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways', 'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans', 'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA', 'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings', 'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas', 'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines', 'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_2', 'MES_3', 'MES_4', 'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9', 'MES_10', 'MES_11', 'MES_12']
            test_values = [0]*len(test_keys)
            print(test_keys, test_values)
            res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
            
            for i in json_:
                aux = res
                for k, v in i.items():
                    llave = f"{k}_{v}"
                    aux[llave] = 1
                    
            valor = pd.DataFrame(aux.items())
            valor = valor.T
            valor.columns = valor.iloc[0]
            valor = valor.drop(0)
            valor = valor.apply(pd.to_numeric)
            
            predict = list(xgboost.predict(valor))
            return jsonify({'prediction': str(predict)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Model not good')
        return ('Model is not good')
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345 
    xgboost = joblib.load("xgboost.pkl") 
    print ('Model loaded')
    xgboost_columns = joblib.load("xgboost_columns.pkl") # Load "xgboost_columns.pkl"
    print ('Model columns loaded')
    application.run(port=port, debug=True)