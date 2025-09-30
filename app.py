from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)

model=None
X_test_scaled=None
y_test=None
scaler=None

@app.route('/')
def home():
    return "Home route"

@app.route('/train', methods=['POST'])
def train():
    global model, X_test_scaled, y_test, scaler
    df=pd.read_csv('Iris.csv')
    df['Species_enc']=df['Species'].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
    
    X=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y=df['Species_enc']
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25)
    
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    model=LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    return "Model is being trained!"
    
@app.route('/test', methods=['GET'])
def test():
    if model is None:
        return "Train the model first by hitting /train"
    score=model.score(X_test_scaled, y_test)
    return jsonify({
        'Score':score,
    })
    
@app.route('/predict', methods=['POST'])
def predict():
    global scaler
    if model is None:
        return "Train the model first by hitting /train"
    
    input_data=request.get_json()
    features=[
        input_data['SepalLengthCm'],
        input_data['SepalWidthCm'],
        input_data['PetalLengthCm'],
        input_data['PetalWidthCm']
    ]
    
    input_df=pd.DataFrame([features], columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    input_scaled=scaler.transform(input_df)
    pred=model.predict(input_scaled)
    species_map = {1:'Iris-setosa', 2:'Iris-versicolor', 3:'Iris-virginica'}
    pred_species = species_map[pred[0]]
    
    return jsonify({'Predicted species': pred_species})

if __name__=='__main__':
    app.run(debug=True)