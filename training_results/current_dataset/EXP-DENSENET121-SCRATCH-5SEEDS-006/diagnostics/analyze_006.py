from pathlib import Path
import csv, json, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'diagnostics'
SEEDS = [42,123,2026,3407,7777]
rows=[]
fig, axes=plt.subplots(3,5,figsize=(19,10),sharex='col')
for col, seed in enumerate(SEEDS):
    with (ROOT/f'seed_{seed}/logs/history.csv').open() as f:
        h=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    with (ROOT/f'seed_{seed}/validation/predictions.csv').open() as f:
        preds=list(csv.DictReader(f))
    with (ROOT/f'seed_{seed}/validation/confusion_matrix.csv').open() as f:
        cr=list(csv.reader(f)); cm=[[int(v) for v in r[1:]] for r in cr[1:]]
    classes=cr[0][1:]
    calc=[[0]*5 for _ in range(5)]
    for p in preds:
        calc[classes.index(p['true_label'])][classes.index(p['predicted_label'])]+=1
    assert cm==calc, seed
    assert len(preds)==1402 and len({p['filename'] for p in preds})==1402
    assert all((p['true_label']==p['predicted_label'])==(p['correct']=='True') for p in preds)
    best=min(h,key=lambda r:r['val_loss']); last=h[-1]
    acc=sum(cm[i][i] for i in range(5))/1402
    assert abs(acc-best['val_accuracy'])<1e-6
    spikes=[r for prev,r in zip(h,h[1:]) if r['epoch']>=5 and prev['val_accuracy']-r['val_accuracy']>=.10]
    rows.append(dict(seed=seed,epochs=len(h),best_epoch=int(best['epoch']),train_online_accuracy=best['accuracy'],val_accuracy=acc,gap_best_pp=100*(best['accuracy']-acc),gap_last_pp=100*(last['accuracy']-last['val_accuracy']),spikes_ge10pp_after_epoch4=len(spikes),spike_epochs=';'.join(str(int(r['epoch'])) for r in spikes)))
    epochs=[r['epoch'] for r in h]
    axes[0,col].plot(epochs,[r['accuracy']*100 for r in h],label='Train online',color='#2878a5')
    axes[0,col].plot(epochs,[r['val_accuracy']*100 for r in h],label='Validation',color='#d95f02')
    axes[0,col].scatter([r['epoch'] for r in spikes],[r['val_accuracy']*100 for r in spikes],color='#b2182b',s=24,zorder=4,label='Drop >= 10 pp')
    axes[0,col].set_title(f'Seed {seed} | best epoch {int(best["epoch"])}')
    axes[0,col].set_ylim(0,103)
    axes[1,col].plot(epochs,[r['loss'] for r in h],color='#2878a5')
    axes[1,col].plot(epochs,[r['val_loss'] for r in h],color='#d95f02')
    axes[1,col].set_yscale('log'); axes[1,col].set_ylim(.015,10)
    axes[2,col].step(epochs,[r['learning_rate'] for r in h],where='post',color='#5c3a8e')
    axes[2,col].set_yscale('log'); axes[2,col].set_ylim(7e-7,5e-4)
    axes[2,col].set_xlabel('Epoch')
    for ax in axes[:,col]:
        ax.axvline(best['epoch'],ls='--',lw=1,color='#333333',alpha=.65)
        ax.grid(alpha=.2)
axes[0,0].set_ylabel('Accuracy (%)'); axes[1,0].set_ylabel('Cross-entropy (log scale)');axes[2,0].set_ylabel('Learning rate (log scale)')
axes[0,0].legend(fontsize=8,loc='lower right')
fig.suptitle('DenseNet121 scratch 006: raw curves, instability and learning rate',fontsize=17)
fig.text(.5,.016,'Train metrics are measured online with augmentation / dropout / batch statistics; validation uses inference mode. Gap is not clean-eval gap.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.04,1,.955))
fig.savefig(OUT/'diagnostic_curves.png',dpi=160)
fig.savefig(OUT/'diagnostic_curves.pdf')
plt.close(fig)
with (OUT/'seed_diagnostics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'validation_predictions_verified': 7010, 'mean_best_online_gap_pp': statistics.mean(r['gap_best_pp'] for r in rows), 'total_spikes': sum(r['spikes_ge10pp_after_epoch4'] for r in rows)}, indent=2))
