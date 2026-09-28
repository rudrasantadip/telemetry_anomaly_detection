from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset.csv'
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)

TARGET='anomaly'
DROP=['segment','anomaly','train']

df=pd.read_csv(DATA)
features=[c for c in df.columns if c not in DROP]
cat=['channel']
num=[c for c in features if c not in cat]

# Respect the dataset's official train/test flag.
train=df[df['train']==1].copy()
test=df[df['train']==0].copy()

# A validation split is used only to choose an operating threshold.
tr, val = train_test_split(train, test_size=0.20, stratify=train[TARGET], random_state=42)
pre=ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat)
])
clf=ExtraTreesClassifier(
    n_estimators=500, min_samples_leaf=2, class_weight='balanced',
    random_state=42, n_jobs=-1
)
pipe=Pipeline([('preprocessor',pre),('classifier',clf)])
pipe.fit(tr[features], tr[TARGET])
val_prob=pipe.predict_proba(val[features])[:,1]
thresholds=np.linspace(0.15,0.85,141)
f1s=[f1_score(val[TARGET], val_prob>=t) for t in thresholds]
best=float(thresholds[int(np.argmax(f1s))])

# Refit on all official training data.
pipe.fit(train[features], train[TARGET])
test_prob=pipe.predict_proba(test[features])[:,1]
pred=(test_prob>=best).astype(int)
metrics={
    'accuracy':float(accuracy_score(test[TARGET],pred)),
    'precision':float(precision_score(test[TARGET],pred)),
    'recall':float(recall_score(test[TARGET],pred)),
    'f1':float(f1_score(test[TARGET],pred)),
    'roc_auc':float(roc_auc_score(test[TARGET],test_prob)),
    'pr_auc':float(average_precision_score(test[TARGET],test_prob)),
    'confusion_matrix':confusion_matrix(test[TARGET],pred).tolist(),
    'threshold':best,
    'train_rows':int(len(train)), 'test_rows':int(len(test)),
    'features':features
}
joblib.dump(pipe, MODEL_DIR/'telemetry_extratrees.joblib')

# Optional unsupervised detector, useful when labels are unavailable.
iso_pre=ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat)
])
iso=Pipeline([('preprocessor',iso_pre),('detector',IsolationForest(
    n_estimators=300, contamination='auto', random_state=42, n_jobs=-1
))])
iso.fit(train[features])
joblib.dump(iso, MODEL_DIR/'telemetry_isolation_forest.joblib')

with open(MODEL_DIR/'metadata.json','w') as f: json.dump(metrics,f,indent=2)
print(json.dumps(metrics,indent=2))
