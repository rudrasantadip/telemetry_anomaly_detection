from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MODEL=joblib.load(ROOT/'model'/'telemetry_extratrees.joblib')
ISO=joblib.load(ROOT/'model'/'telemetry_isolation_forest.joblib')
META=json.load(open(ROOT/'model'/'metadata.json'))
DATA=pd.read_csv(ROOT/'data'/'dataset.csv')
FEATURES=META['features']
NUM=[c for c in FEATURES if c!='channel']

st.set_page_config(page_title='Telemetry Anomaly Monitor', page_icon='🛰️', layout='wide')
st.title('🛰️ Telemetry Anomaly Monitor')
st.caption('OPS-SAT telemetry segment classification using ExtraTrees + optional Isolation Forest.')

with st.sidebar:
    st.header('Model')
    st.metric('Decision threshold', f"{META['threshold']:.3f}")
    st.metric('Test ROC-AUC', f"{META['roc_auc']:.3f}")
    st.metric('Test F1', f"{META['f1']:.3f}")
    page=st.radio('Mode',['Dashboard','Single segment','Batch CSV'])

if page=='Dashboard':
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Segments',len(DATA)); c2.metric('Anomalies',int(DATA.anomaly.sum()))
    c3.metric('Channels',DATA.channel.nunique()); c4.metric('Anomaly rate',f"{DATA.anomaly.mean()*100:.1f}%")
    left,right=st.columns(2)
    with left:
        st.subheader('Anomaly rate by channel')
        ch=DATA.groupby('channel').anomaly.mean().sort_values(ascending=False)
        st.bar_chart(ch)
    with right:
        st.subheader('Mean vs standard deviation')
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter(DATA['mean'],DATA['std'],c=DATA['anomaly'],alpha=.55,s=14)
        ax.set_xlabel('Mean'); ax.set_ylabel('Std'); ax.set_title('Telemetry feature space')
        st.pyplot(fig,use_container_width=True)
    st.subheader('Recent / highest-risk segments')
    p=MODEL.predict_proba(DATA[FEATURES])[:,1]
    out=DATA[['segment','channel','anomaly']].copy(); out['risk_probability']=p
    st.dataframe(out.sort_values('risk_probability',ascending=False).head(25),use_container_width=True)

elif page=='Single segment':
    st.subheader('Score one telemetry segment')
    row={}
    row['channel']=st.selectbox('Channel',sorted(DATA.channel.unique()))
    cols=st.columns(3)
    for i,c in enumerate(NUM):
        default=float(DATA[c].median()) if pd.api.types.is_numeric_dtype(DATA[c]) else 0
        with cols[i%3]: row[c]=st.number_input(c,value=default,format='%.8g')
    x=pd.DataFrame([row],columns=FEATURES)
    if st.button('Predict anomaly',type='primary'):
        prob=float(MODEL.predict_proba(x)[0,1]); pred=int(prob>=META['threshold'])
        if pred: st.error(f'ANOMALY DETECTED — probability {prob:.1%}')
        else: st.success(f'NOMINAL — anomaly probability {prob:.1%}')
        iso_score=float(-ISO.decision_function(x)[0])
        st.metric('Anomaly probability',f'{prob:.2%}')
        st.metric('Isolation Forest score',f'{iso_score:.4f}')

else:
    st.subheader('Batch scoring')
    st.write('Upload a CSV containing the model features. Extra columns are ignored; required columns are shown below.')
    st.code(', '.join(FEATURES))
    file=st.file_uploader('CSV file',type=['csv'])
    if file:
        incoming=pd.read_csv(file)
        missing=[c for c in FEATURES if c not in incoming.columns]
        if missing:
            st.error('Missing columns: '+', '.join(missing))
        else:
            scored=incoming.copy()
            prob=MODEL.predict_proba(incoming[FEATURES])[:,1]
            scored['anomaly_probability']=prob
            scored['prediction']=(prob>=META['threshold']).astype(int)
            st.dataframe(scored,use_container_width=True)
            st.download_button('Download scored CSV',scored.to_csv(index=False).encode(),'telemetry_predictions.csv','text/csv')
