import time, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import features as F
def log(*a): print(*a,flush=True)
t=time.time()
samples=F.load_jsonl("../../data/train.jsonl")
labels=pd.read_csv("../../data/train_labels.csv")
lm=dict(zip(labels.id,labels.action))
ids,prompts,records=F.build_records(samples)
y=np.array([F.CLASS_ORDER.index(lm[i]) for i in ids])
log("load+build",round(time.time()-t,1),"s")
skf=StratifiedKFold(5,shuffle=True,random_state=42)
tr_idx,va_idx=next(skf.split(np.zeros(len(y)),y))
t=time.time()
ptr=[prompts[i] for i in tr_idx]; rtr=[records[i] for i in tr_idx]; ytr=y[tr_idx]
wv=TfidfVectorizer(analyzer='word',ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True).fit(ptr)
cv=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),max_features=20000,min_df=2,sublinear_tf=True).fit(ptr)
art={"word_vec":wv,"char_vec":cv,"cat_mappings":F.build_cat_mappings(rtr)}
X=F.transform_all(ptr,rtr,art)
log("tfidf+transform",round(time.time()-t,1),"s  X",X.shape,"nnz/row",round(X.nnz/X.shape[0],1))
il=np.arange(len(tr_idx))
tr2,es2=train_test_split(il,test_size=0.1,random_state=42,stratify=ytr)
Xtr,Xes,y2,yes=X[tr2],X[es2],ytr[tr2],ytr[es2]
cw=compute_class_weight('balanced',classes=np.arange(14),y=y2); sw=cw[y2]
for nj in [16, 8]:
    m=lgb.LGBMClassifier(objective='multiclass',num_class=14,n_estimators=50,learning_rate=0.05,
        num_leaves=63,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,
        reg_lambda=1.0,n_jobs=nj,random_state=42,verbose=-1)
    t=time.time()
    m.fit(Xtr,y2,sample_weight=sw)
    dt=time.time()-t
    log(f"n_jobs={nj} 50-iter fit {dt:.1f}s -> per-iter {dt/50*1000:.0f}ms")
