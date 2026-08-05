# -*- coding: utf-8 -*-
"""
留出测试集 + Fold 3 分析 + SHAP 可解释
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import h5py
from scipy.signal import find_peaks
from scipy.stats import skew
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

MAT_ORIG = r'E:\VSCode\BatteryLife-AI\MIT-Stanford Battery Dataset_cleaned.mat'
MAT_PROC = r'E:\VSCode\BatteryLife-AI\battery_analysis_results.mat'

# ============================================================
# 1. 读取 Feature D 全量数据
# ============================================================
print('读取数据...')
f_orig = h5py.File(MAT_ORIG, 'r'); f_proc = h5py.File(MAT_PROC, 'r')
bc, bi = f_orig['batch_combined'], f_proc['battery_info']
bats = sorted(bi.keys())

N_BAT, SEQ_LEN = 124, 100
SOC_BINS = [(0,20),(20,40),(40,60),(60,80)]

X_seq = np.zeros((N_BAT, SEQ_LEN, 18), dtype=np.float32)  # Feature D: 18 dims
y_life = np.zeros(N_BAT)
meta = {}  # 存储策略参数用于 Fold 3 分析

for idx, g in enumerate(bats):
    bg = bi[g]; cl = int(bg['cycle_life'][()]); y_life[idx] = cl
    soh=bg['SOH'][()].flatten(); ir=bg['IR'][()].flatten()
    qc=bg['QCharge'][()].flatten(); qd=bg['QDischarge'][()].flatten(); ct=bg['chargetime'][()].flatten()
    ref_s = bc['summary'][idx,0]; summ = f_orig[ref_s]; tavg = summ['Tavg'][0,:]
    c1 = float(bg['C1'][()]); c1 = c1 if c1>0 else 0
    q1 = float(bg['Q1'][()]); q1 = q1 if q1>0 else 0
    c2 = float(bg['C2'][()]); c2 = c2 if c2>0 else 0
    meta[idx] = {'C1':c1,'Q1':q1,'C2':c2,'cycle_life':cl}

    n_take = min(SEQ_LEN, len(soh))
    for t in range(n_take):
        X_seq[idx,t,0]=soh[t]; X_seq[idx,t,1]=ir[t]; X_seq[idx,t,2]=tavg[t]
        X_seq[idx,t,3]=qc[t]; X_seq[idx,t,4]=qd[t]; X_seq[idx,t,5]=ct[t]
    for t in range(n_take):
        X_seq[idx,t,6]=c1; X_seq[idx,t,7]=q1; X_seq[idx,t,8]=c2

    # SOC bin C-rate
    for j,(lo,hi) in enumerate(SOC_BINS):
        cr=[]; s1l,s1h=max(lo,0),min(hi,q1)
        if s1h>s1l: cr.append(c1)
        s2l,s2h=max(lo,q1),min(hi,80)
        if s2h>s2l: cr.append(c2)
        if max(lo,80)<min(hi,100): cr.append(1.0)
        crate = np.mean(cr) if cr else 1.0
        for t in range(n_take): X_seq[idx,t,9+j]=crate

    # dQ/dV stats
    ref_c = bc['cycles'][idx,0]; cg = f_orig[ref_c]; dqdv_r = cg['discharge_dQdV']
    for t in range(min(SEQ_LEN, dqdv_r.shape[0])):
        dqdv = f_orig[dqdv_r[t,0]][()].flatten()
        if len(dqdv)<10: continue
        X_seq[idx,t,13]=np.var(dqdv)
        peaks,_=find_peaks(-dqdv, prominence=0.01)
        X_seq[idx,t,14]=len(peaks)
        if len(peaks)>0:
            ph=dqdv[peaks]; mi=np.argmax(np.abs(ph))
            X_seq[idx,t,15]=ph[mi]; X_seq[idx,t,16]=peaks[mi]/1000.0
        X_seq[idx,t,17]=skew(dqdv)

f_orig.close(); f_proc.close()
print(f'  {N_BAT} 电池, shape={X_seq.shape}')

# ============================================================
# 2. 留出测试集: 100 train / 12 val / 12 test
# ============================================================
print('\n' + '='*60)
print('留出测试集 (100/12/12)')
print('='*60)

np.random.seed(42)
idx_shuf = np.random.permutation(N_BAT)
# 按cycle_life分层采样，确保test覆盖范围
bins = np.digitize(y_life, np.percentile(y_life, [20,40,60,80]))
test_idx, val_idx, train_idx = [], [], []
for b in range(6):
    in_bin = idx_shuf[bins[idx_shuf]==b]
    n = len(in_bin)
    n_test = max(1, n//10); n_val = max(1, n//10)
    test_idx.extend(in_bin[:n_test])
    val_idx.extend(in_bin[n_test:n_test+n_val])
    train_idx.extend(in_bin[n_test+n_val:])
train_idx=np.array(train_idx); val_idx=np.array(val_idx); test_idx=np.array(test_idx)
print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
print(f'Test cycle_life range: {y_life[test_idx].min():.0f}-{y_life[test_idx].max():.0f}')

# ============================================================
# 3. Transformer 模型 (复用)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe=torch.zeros(max_len,d_model); pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*-(np.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1),:]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.ip=nn.Linear(input_dim,d_model); self.pe=PositionalEncoding(d_model)
        el=nn.TransformerEncoderLayer(d_model,nhead,dropout=dropout,batch_first=True)
        self.tr=nn.TransformerEncoder(el,num_layers)
        self.fc=nn.Sequential(nn.Linear(d_model,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,1))
    def forward(self,x): x=self.ip(x); x=self.pe(x); x=self.tr(x); return self.fc(x[:,-1,:]).squeeze(-1)

device='cuda' if torch.cuda.is_available() else 'cpu'

def train_transformer(X_tr,y_tr,X_val,y_val, epochs=200, lr=1e-3):
    xm=X_tr.mean(axis=(0,1),keepdims=True); xs=X_tr.std(axis=(0,1),keepdims=True); xs[xs==0]=1.0
    Xt=(X_tr-xm)/xs; Xv=(X_val-xm)/xs
    ym,ys=y_tr.mean(),y_tr.std()
    yt=(y_tr-ym)/ys; yv=(y_val-ym)/ys
    tr_ds=TensorDataset(torch.FloatTensor(Xt),torch.FloatTensor(yt))
    vl_ds=TensorDataset(torch.FloatTensor(Xv),torch.FloatTensor(yv))
    tr_dl=DataLoader(tr_ds,batch_size=16,shuffle=True)
    vl_dl=DataLoader(vl_ds,batch_size=64)

    model=TransformerPredictor(input_dim=X_tr.shape[2]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=20,factor=0.5)
    crit=nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in tr_dl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=crit(model(xb),yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in vl_dl: xb,yb=xb.to(device),yb.to(device); vl+=crit(model(xb),yb).item()
        sch.step(vl)

    model.eval()
    with torch.no_grad():
        yp_s=model(torch.FloatTensor(Xv).to(device)).cpu().numpy()
    yp=yp_s*ys+ym
    return yp, model, (xm,xs,ym,ys)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

yp_val, model, norm = train_transformer(
    X_seq[train_idx],y_life[train_idx], X_seq[val_idx],y_life[val_idx])

yp_test, _, _ = train_transformer(
    np.concatenate([X_seq[train_idx],X_seq[val_idx]]),
    np.concatenate([y_life[train_idx],y_life[val_idx]]),
    X_seq[test_idx],y_life[test_idx])

print(f'\nVal:  MAE={mean_absolute_error(y_life[val_idx],yp_val):.0f}, '
      f'R2={r2_score(y_life[val_idx],yp_val):.3f}')
print(f'Test: MAE={mean_absolute_error(y_life[test_idx],yp_test):.0f}, '
      f'RMSE={np.sqrt(mean_squared_error(y_life[test_idx],yp_test)):.0f}, '
      f'R2={r2_score(y_life[test_idx],yp_test):.3f}')

# ============================================================
# 4. Fold 3 分析
# ============================================================
print('\n'+'='*60)
print('Fold 3 异常分析')
print('='*60)

from sklearn.model_selection import KFold
kf=KFold(n_splits=5,shuffle=True,random_state=42)
splits=list(kf.split(np.arange(N_BAT)))
fold3_val=set(splits[2][1])
fold_others=set(range(N_BAT))-fold3_val

def compare_fold(name, vals):
    f3=np.array([vals[i] for i in fold3_val])
    fo=np.array([vals[i] for i in fold_others])
    print(f'  {name}: Fold3={f3.mean():.2f}±{f3.std():.2f}, Others={fo.mean():.2f}±{fo.std():.2f}')

print(); compare_fold('cycle_life', y_life)
compare_fold('C1', np.array([meta[i]['C1'] for i in range(N_BAT)]))
compare_fold('Q1', np.array([meta[i]['Q1'] for i in range(N_BAT)]))
compare_fold('C2', np.array([meta[i]['C2'] for i in range(N_BAT)]))
compare_fold('SOH(std, first 100)', np.std(X_seq[:,:,0],axis=1))
compare_fold('IR(mean, first 100)', np.mean(X_seq[:,:,1],axis=1))
compare_fold('dQ/dV var(mean)', np.mean(X_seq[:,:,13],axis=1))

# ============================================================
# 5. SHAP 分析
# ============================================================
print('\n'+'='*60)
print('SHAP 可解释分析')
print('='*60)

# 聚合特征 (每电池 → 标量向量) 以便用 TreeExplainer
def aggregate(X):
    N,T,D=X.shape
    feats=[]
    dyn_names=['SOH','IR','Tavg','QCharge','QDischarge','CT']
    for d in range(6):
        v=X[:,:,d]
        feats.extend([np.mean(v,1),np.std(v,1),np.min(v,1),np.max(v,1),
                       np.array([np.polyfit(np.arange(T),v[i],1)[0] for i in range(N)])])
    # C1,Q1,C2 (取均值=原值)
    feats.extend([X[:,0,6],X[:,0,7],X[:,0,8]])
    # SOC bins (均值)
    for d in range(9,13): feats.append(X[:,0,d])
    # dQ/dV stats (mean across cycles)
    for d in range(13,18): feats.append(np.mean(X[:,:,d],1))
    return np.column_stack(feats)

X_agg = aggregate(X_seq)
agg_names = []
dyn=['SOH','IR','温度','充电容量','放电容量','充电时间']
for n in dyn:
    for s in ['均值','标准差','最小值','最大值','斜率']: agg_names.append(f'{n}_{s}')
agg_names+=['C1','Q1','C2']; agg_names+=[f'Crate_{l}-{h}' for l,h in SOC_BINS]
agg_names+=['dQdV_方差','dQdV_峰数','dQdV_峰高','dQdV_峰位','dQdV_偏度']

print(f'聚合特征: {X_agg.shape[1]} 维')

# Random Forest (样本少，RF 比 XGBoost 更稳)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_agg[train_idx], y_life[train_idx])
yp_rf = rf_model.predict(X_agg[test_idx])
print(f'RF Test R2: {r2_score(y_life[test_idx],yp_rf):.3f}')

# SHAP TreeExplainer
import shap
print('计算 SHAP 值 (可能需要1-2分钟)...')
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_agg[train_idx])

# SHAP summary bar
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
FIG_DIR=r'E:\VSCode\BatteryLife-AI\figures'; os.makedirs(FIG_DIR,exist_ok=True)
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

shap.summary_plot(shap_values, X_agg[train_idx], feature_names=agg_names,
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'shap_summary.png'),dpi=150,bbox_inches='tight')
plt.close()
print(f'SHAP summary: {FIG_DIR}/shap_summary.png')

# SHAP 瀑布图 (单个电池示例)
shap.waterfall_plot(shap_values[0], show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'shap_waterfall.png'),dpi=150,bbox_inches='tight')
plt.close()
print(f'SHAP waterfall: {FIG_DIR}/shap_waterfall.png')

# 打印 Top 15 特征重要性
print('\nSHAP 特征重要性 Top 15:')
shap_imp = np.abs(shap_values.values).mean(0)
top_idx = np.argsort(shap_imp)[::-1][:15]
for rank,i in enumerate(top_idx,1):
    print(f'  {rank:2d}. {agg_names[i]:25s}  |SHAP|={shap_imp[i]:.4f}')

print('\n全部完成!')
