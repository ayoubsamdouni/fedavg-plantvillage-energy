"""Figures 2-4 of the paper, drawn from the per-round logs in results/logs (written to figures/)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

from runs import FIGURES, RUNS, log_file
runs = {sc: [str(log_file(sc, s)) for (k, s) in RUNS if k == sc] for sc in ("S1", "S2", "S3", "S4", "S5")}
FIGURES.mkdir(parents=True, exist_ok=True)
OUT = str(FIGURES) + "/"

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#6b6a64','axes.labelcolor':'#2b2a27','xtick.color':'#6b6a64','ytick.color':'#6b6a64'})
C={'S1':'#2a78d6','S2':'#eb6834','S3':'#1baf7a','S4':'#eda100','S5':'#e87ba4'}
E={'S1':3,'S2':3,'S3':3,'S4':1,'S5':1}
L={'S1':'S1  IID, E=3','S2':'S2  α=0.5, E=3','S3':'S3  α=0.1, E=3','S4':'S4  α=0.1, E=1','S5':'S5  α=0.5, E=1'}
acc=lambda f: pd.read_csv(f).set_index('round').accuracy*100

# ---- Figure 2: replicate curves, one panel per scenario
fig,axs=plt.subplots(1,4,figsize=(7.4,2.6),dpi=300,sharey=True)
s1=acc(runs['S1'][0])
for ax,k in zip(axs,['S2','S3','S4','S5']):
    ax.plot(s1.index,s1.values,color='#9a998f',lw=1.0,ls=(0,(3,2)),label='S1 (IID)')
    for i,f in enumerate(runs[k]):
        a=acc(f); ax.plot(a.index,a.values,color=C[k],lw=1.3,alpha=[1,0.7,0.45][i],label=f'seed {42+i}')
    ax.set_title(L[k].replace('  ',': '),fontsize=8,color='#2b2a27'); ax.set_xlim(0,30); ax.set_ylim(0,100)
    ax.set_xlabel('Round'); ax.grid(axis='y',color='#e3e2dc',lw=0.6); ax.set_axisbelow(True)
    fin=[acc(f)[30] for f in runs[k]]
    ax.text(29.5,4,f'round 30:\n{np.mean(fin):.1f} ± {np.std(fin,ddof=1):.1f}%',ha='right',va='bottom',fontsize=6.8,color='#2b2a27')
axs[0].set_ylabel('Global test accuracy (%)')
from matplotlib.lines import Line2D
h=[Line2D([],[],color='#9a998f',lw=1.0,ls=(0,(3,2)))]+[Line2D([],[],color='#55544f',lw=1.3,alpha=a) for a in (1,0.7,0.45)]
fig.legend(h,['S1 (IID, reference)','seed 42','seed 43','seed 44'],loc='upper center',ncol=4,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,0.99))
fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig(OUT+'fig2_replicates.png'); plt.close(fig)
# ---- Figure 3: accuracy vs cumulative local epochs (13 runs)
fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=300)
ax.axvspan(0,6,color='#e9e8e3',lw=0,zorder=0)
ax.annotate('BN warm-up: accuracy stays near chance for the\nfirst 6 local epochs (7 in one run), in all 13 runs',xy=(5.5,12),xytext=(34,30),fontsize=7.5,color='#55544f',arrowprops=dict(arrowstyle='-',color='#9a998f',lw=0.7))
ax.axvline(30,color='#9a998f',lw=0.8,ls=(0,(2,2)),zorder=1)
ax.text(31,3,'equal local computation\n(E=3 round 10 ≡ E=1 round 30)',fontsize=7.5,color='#55544f',va='bottom')
for k,fs in runs.items():
    for i,f in enumerate(fs):
        a=acc(f)
        ax.plot(a.index*E[k],a.values,color=C[k],lw=1.6 if i==0 else 1.0,alpha=1 if i==0 else 0.5,
                ls='-' if E[k]==3 else (0,(4,2)),label=(L[k]+(f' ({len(fs)} seeds)' if len(fs)>1 else '')) if i==0 else None,zorder=3)
ax.set_xlim(0,92); ax.set_ylim(0,100); ax.set_xlabel('Cumulative local epochs per client (round × E)')
ax.set_ylabel('Global test accuracy (%)'); ax.grid(axis='y',color='#e3e2dc',lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False,loc='lower right',fontsize=7.5,handlelength=2.6)
fig.tight_layout(); fig.savefig(OUT+'fig3_accuracy_vs_local_epochs.png'); plt.close(fig)
# ---- Figure 4: E=1 cost under two budgets, 3 matched seeds
g=lambda f,r: acc(f)[r]
pts={}
for b,(r3,r1) in {'comm':(30,30),'comp':(10,30)}.items():
    pts[('α=0.1',b)]=[g(runs['S3'][i],r3)-g(runs['S4'][i],r1) for i in range(3)]
    pts[('α=0.5',b)]=[g(runs['S2'][i],r3)-g(runs['S5'][i],r1) for i in range(3)]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),dpi=300,sharey=True)
for ax,(b,title) in zip(axs,[('comm','Equal communication\n(both after 30 rounds)'),('comp','Equal local computation\n(30 local epochs each)')]):
    for j,(a,col) in enumerate([('α=0.1',C['S4']),('α=0.5',C['S5'])]):
        v=pts[(a,b)]; ax.bar(j,np.mean(v),width=0.5,color=col,lw=0,zorder=2)
        xs=[j-0.12,j,j+0.12]; ax.scatter(xs,v,s=20,color='#2b2a27',zorder=5)
        ax.text(j+0.3,np.mean(v),f'mean {np.mean(v):+.1f}',fontsize=7,va='center',color='#2b2a27')
    ax.axhline(0,color='#6b6a64',lw=0.8,zorder=4)
    ax.set_xticks([0,1],['α=0.1 (S3 vs S4)','α=0.5 (S2 vs S5)']); ax.set_title(title,fontsize=8.5,color='#2b2a27')
    ax.grid(axis='y',color='#e3e2dc',lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-0.5,1.8)
axs[0].set_ylabel('Accuracy cost of E=1 vs E=3 (pp)\n(positive = E=1 worse)')
fig.tight_layout(); fig.savefig(OUT+'fig4_epoch_cost_two_budgets.png'); plt.close(fig)
print({k: [round(float(x), 2) for x in v] for k, v in pts.items()})
