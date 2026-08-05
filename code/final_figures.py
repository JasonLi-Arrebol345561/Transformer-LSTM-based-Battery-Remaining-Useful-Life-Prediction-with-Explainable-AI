# -*- coding: utf-8 -*-
"""
论文收尾: Prediction scatter + Attention map + 汇总表
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np, h5py, os
from scipy.signal import find_peaks; from scipy.stats import skew
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

MAT_ORIG = r'E:\VSCode\BatteryLife-AI\MIT-Stanford Battery Dataset_cleaned.mat'
MAT_PROC = r'E:\VSCode\BatteryLife-AI\battery_analysis_results.mat'
FIG_DIR = r'E:\VSCode\BatteryLife-AI\figures'; os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False

# ============================================================
# 1. 读取 Feature D 全量数据 (复用 holdout_shap_analysis 逻辑)
# ============================================================
print('读取数据...')
f_orig=h5py.File(MAT_ORIG,'r'); f_proc=h5py.File(MAT_PROC,'r')
bc,bi=f_orig['batch_combined'],f_proc['battery_info']; bats=sorted(bi.keys())
N_BAT,SEQ_LEN=124,100; SOC_BINS=[(0,20),(20,40),(40,60),(60,80)]

X_seq=np.zeros((N_BAT,SEQ_LEN,18),dtype=np.float32); y_life=np.zeros(N_BAT)

for idx,g in enumerate(bats):
    bg=bi[g]; cl=int(bg['cycle_life'][()]); y_life[idx]=cl
    soh=bg['SOH'][()].flatten(); ir=bg['IR'][()].flatten()
    qc=bg['QCharge'][()].flatten(); qd=bg['QDischarge'][()].flatten(); ct=bg['chargetime'][()].flatten()
    ref_s=bc['summary'][idx,0]; summ=f_orig[ref_s]; tavg=summ['Tavg'][0,:]
    c1=float(bg['C1'][()]); c1=c1 if c1>0 else 0
    q1=float(bg['Q1'][()]); q1=q1 if q1>0 else 0
    c2=float(bg['C2'][()]); c2=c2 if c2>0 else 0
    n_take=min(SEQ_LEN,len(soh))
    for t in range(n_take):
        X_seq[idx,t,0]=soh[t]; X_seq[idx,t,1]=ir[t]; X_seq[idx,t,2]=tavg[t]
        X_seq[idx,t,3]=qc[t]; X_seq[idx,t,4]=qd[t]; X_seq[idx,t,5]=ct[t]
    for t in range(n_take):
        X_seq[idx,t,6]=c1; X_seq[idx,t,7]=q1; X_seq[idx,t,8]=c2
    for j,(lo,hi) in enumerate(SOC_BINS):
        cr=[]; s1l,s1h=max(lo,0),min(hi,q1)
        if s1h>s1l: cr.append(c1)
        s2l,s2h=max(lo,q1),min(hi,80)
        if s2h>s2l: cr.append(c2)
        if max(lo,80)<min(hi,100): cr.append(1.0)
        crate=np.mean(cr) if cr else 1.0
        for t in range(n_take): X_seq[idx,t,9+j]=crate
    ref_c=bc['cycles'][idx,0]; cg=f_orig[ref_c]; dqdv_r=cg['discharge_dQdV']
    for t in range(min(SEQ_LEN,dqdv_r.shape[0])):
        dqdv=f_orig[dqdv_r[t,0]][()].flatten()
        if len(dqdv)<10: continue
        X_seq[idx,t,13]=np.var(dqdv)
        peaks,_=find_peaks(-dqdv,prominence=0.01); X_seq[idx,t,14]=len(peaks)
        if len(peaks)>0:
            ph=dqdv[peaks]; mi=np.argmax(np.abs(ph))
            X_seq[idx,t,15]=ph[mi]; X_seq[idx,t,16]=peaks[mi]/1000.0
        X_seq[idx,t,17]=skew(dqdv)
f_orig.close(); f_proc.close()
print(f'  完成: {X_seq.shape}')

# ============================================================
# 2. 带 Attention 输出的 Transformer
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=200):
        super().__init__(); pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*-(np.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1),:]

class TransformerWithAttn(nn.Module):
    def __init__(self,input_dim,d_model=64,nhead=4,num_layers=2,dropout=0.2):
        super().__init__()
        self.ip=nn.Linear(input_dim,d_model); self.pe=PositionalEncoding(d_model)
        el=nn.TransformerEncoderLayer(d_model,nhead,dropout=dropout,batch_first=True)
        self.tr=nn.TransformerEncoder(el,num_layers)
        self.fc=nn.Sequential(nn.Linear(d_model,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,1))
        self.d_model=d_model

    def forward(self,x,return_attn=False):
        x=self.ip(x); x=self.pe(x)
        if return_attn:
            attn_weights=[]
            for layer in self.tr.layers:
                q=layer.self_attn.query(x) if hasattr(layer.self_attn,'query') else x
                # 用自定义forward捕获attention
                out,attn=layer.self_attn(x,x,x,need_weights=True,average_attn_weights=False)
                x=layer.norm1(x+layer.dropout1(out))
                x=layer.norm2(x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
                attn_weights.append(attn.cpu().detach())
            out=self.fc(x[:,-1,:]).squeeze(-1)
            return out,attn_weights
        else:
            x=self.tr(x)
            return self.fc(x[:,-1,:]).squeeze(-1)

device='cuda' if torch.cuda.is_available() else 'cpu'

def train_attn_model(X_tr,y_tr,X_val,y_val,epochs=200,lr=1e-3):
    xm=X_tr.mean(axis=(0,1),keepdims=True); xs=X_tr.std(axis=(0,1),keepdims=True); xs[xs==0]=1.0
    Xt=(X_tr-xm)/xs; Xv=(X_val-xm)/xs
    ym,ys=y_tr.mean(),y_tr.std(); yt=(y_tr-ym)/ys
    tr_ds=TensorDataset(torch.FloatTensor(Xt),torch.FloatTensor(yt))
    vl_ds=TensorDataset(torch.FloatTensor(Xv),torch.FloatTensor((y_val-ym)/ys))
    tr_dl=DataLoader(tr_ds,batch_size=16,shuffle=True); vl_dl=DataLoader(vl_ds,batch_size=64)
    model=TransformerWithAttn(input_dim=X_tr.shape[2]).to(device)
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
    return yp,model,(xm,xs,ym,ys)

# ============================================================
# 3. 训练最终模型 + 收集所有预测
# ============================================================
print('\n训练 Transformer (5折 CV + 收集预测)...')
kf=KFold(n_splits=5,shuffle=True,random_state=42)
all_y_true=[]; all_y_pred=[]; attn_maps=[]

for fold,(tr_idx,val_idx) in enumerate(kf.split(X_seq)):
    yp_val,model,norm=train_attn_model(X_seq[tr_idx],y_life[tr_idx],X_seq[val_idx],y_life[val_idx])
    all_y_true.extend(y_life[val_idx]); all_y_pred.extend(yp_val)
    # 取一个样本的 attention (最后一个 fold)
    if fold==4:
        X_val_norm=(X_seq[val_idx]-norm[0])/norm[1]
        with torch.no_grad():
            x_in=torch.FloatTensor(X_val_norm[:1]).to(device)  # 第一个val样本
            _,attn=model(x_in,return_attn=True)
            attn_maps=[a.numpy() for a in attn]  # list of (1, nhead, 100, 100)

all_y_true=np.array(all_y_true); all_y_pred=np.array(all_y_pred)
r2_all=r2_score(all_y_true,all_y_pred)
mae_all=mean_absolute_error(all_y_true,all_y_pred)
print(f'5折CV汇总: MAE={mae_all:.0f}, R2={r2_all:.3f}')

# ============================================================
# Fig 4: Prediction vs True scatter
# ============================================================
print('\n绘制 Prediction vs True scatter...')
fig,ax=plt.subplots(figsize=(6.5,6))
ax.scatter(all_y_true,all_y_pred,c='#2E86AB',edgecolors='#1A5276',alpha=0.7,s=60,zorder=3)
lims=[min(all_y_true.min(),all_y_pred.min()),max(all_y_true.max(),all_y_pred.max())]
ax.plot(lims,lims,'--',color='#FF6B6B',linewidth=2,label='理想预测',zorder=2)
ax.fill_between(lims,[l-141 for l in lims],[l+141 for l in lims],alpha=0.1,color='#FF6B6B',label='±1 RMSE (测试集)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('真实循环寿命',fontsize=13); ax.set_ylabel('预测循环寿命',fontsize=13)
ax.set_title(f'预测 vs 真实 RUL\n(5折交叉验证, R²={r2_all:.3f}, MAE={mae_all:.0f})',fontsize=13,fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'pred_vs_true.png'),dpi=150,bbox_inches='tight'); plt.close()
print(f'  {FIG_DIR}/pred_vs_true.png')

# ============================================================
# Fig 6: Attention map
# ============================================================
if attn_maps:
    print('绘制 Attention map...')
    # attn_maps: list of 2 layers, each (1, nhead, 100, 100)
    # 取 Layer 2 的 attention, 平均所有 heads
    attn_layer2 = attn_maps[1][0]  # (nhead, 100, 100)
    nhead = attn_layer2.shape[0]

    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for h in range(min(8,nhead)):
        ax=axes[h//4,h%4]
        im=ax.imshow(attn_layer2[h],cmap='YlOrRd',aspect='auto',vmin=0,vmax=attn_layer2[h].max()*0.5)
        ax.set_title(f'注意力头 {h+1}',fontsize=10)
        ax.set_xlabel('Key 位置'); ax.set_ylabel('Query 位置')
    if nhead<8:
        for h in range(nhead,8): axes[h//4,h%4].set_visible(False)
    fig.suptitle('Transformer 注意力图 — 第2层\n'
                 '(示例电池，不同注意力头学习不同的时间依赖关系)',
                 fontsize=14,fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'attention_map.png'),dpi=150,bbox_inches='tight'); plt.close()
    print(f'  {FIG_DIR}/attention_map.png')

    # 平均 attention 按 query position 的分布
    avg_attn = attn_layer2.mean(axis=0)  # (100, 100)
    fig,ax=plt.subplots(figsize=(8,5.5))
    # 看最后10个query position 关注哪些 key position
    for offset in [1,5,10,20,50]:
        q_pos = 100-offset
        ax.plot(avg_attn[q_pos],label=f'Query 位置={q_pos} (倒数第{offset})',linewidth=1.5)
    ax.set_xlabel('Key 位置 (前100循环中的周期编号)'); ax.set_ylabel('注意力权重')
    ax.set_title('注意力分布: 模型关注哪些周期?\n'
                 '(所有注意力头平均, 第2层)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'attention_profile.png'),dpi=150,bbox_inches='tight'); plt.close()
    print(f'  {FIG_DIR}/attention_profile.png')

# ============================================================
# 汇总表格 (控制台输出)
# ============================================================
print('\n'+'='*60)
print('表2: 模型对比 (5折交叉验证)')
print('='*60)
# 从 model_comparison 复用数值
print(f'{"模型":<15} {"MAE":>8} {"RMSE":>8} {"R²":>8}')
print('-'*40)
print(f'{"Ridge":<15} {"269":>8} {"379":>8} {"-0.09":>8}')
print(f'{"随机森林":<15} {"162":>8} {"250":>8} {"0.49":>8}')
print(f'{"LSTM":<15} {"139":>8} {"211":>8} {"0.50":>8}')
print(f'{"Transformer":<15} {"125":>8} {"181":>8} {"0.74":>8}')

print('\n'+'='*60)
print('表3: 特征消融实验 (Transformer, 5折交叉验证)')
print('='*60)
print(f'{"阶段":<20} {"维度":<5} {"R²":<12} {"ΔR²":<8}')
print('-'*45)
print(f'{"A (基础退化)":<20} {"6":<5} {"0.522":<12} {"—":<8}')
print(f'{"B (+策略)":<20} {"9":<5} {"0.612":<12} {"+0.090":<8}')
print(f'{"C (+SOC暴露)":<20} {"13":<5} {"0.667":<12} {"+0.055":<8}')
print(f'{"D (+dQ/dV)":<20} {"18":<5} {"0.763":<12} {"+0.097":<8}')

print('\n'+'='*60)
print('表4: 最终测试集性能 (10个留出电池)')
print('='*60)
print(f'  MAE  = 100 次循环')
print(f'  RMSE = 141 次循环')
print(f'  R²   = 0.878')

print('\n全部完成!')
