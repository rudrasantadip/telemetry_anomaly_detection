from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT=Path(__file__).resolve().parent
MODEL=joblib.load(ROOT/'model'/'telemetry_extratrees.joblib')
META=json.load(open(ROOT/'model'/'metadata.json'))
DATA=pd.read_csv(ROOT/'data'/'dataset.csv')
FEATURES=META['features']
app=Flask(__name__)

@app.get('/')
def home():
    return render_template('index.html', metrics=META, channels=sorted(DATA.channel.unique()), features=FEATURES)

@app.get('/health')
def health():
    return jsonify({'status':'ok','model':'ExtraTreesClassifier','threshold':META['threshold']})

def score_rows(rows):
    x=pd.DataFrame(rows)
    missing=[c for c in FEATURES if c not in x.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    p=MODEL.predict_proba(x[FEATURES])[:,1]
    pred=(p>=META['threshold']).astype(int)
    return [{'anomaly_probability':float(a),'prediction':int(b)} for a,b in zip(p,pred)]

@app.post('/predict')
def predict():
    try:
        payload=request.get_json(force=True)
        rows=payload if isinstance(payload,list) else [payload]
        return jsonify({'predictions':score_rows(rows)})
    except Exception as e:
        return jsonify({'error':str(e)}),400

@app.post('/predict-form')
def predict_form():
    try:
        row={c:request.form.get(c) for c in FEATURES}
        for c in FEATURES:
            if c!='channel': row[c]=float(row[c])
        result=score_rows([row])[0]
        return render_template('result.html', result=result, row=row)
    except Exception as e:
        return render_template('result.html', error=str(e), row={}),400

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
