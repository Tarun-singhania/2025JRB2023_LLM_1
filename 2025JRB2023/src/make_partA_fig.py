#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--csv',default='results/partA_sweep.csv')
ap.add_argument('--png',default='results/partA_fig.png')
ap.add_argument('--pdf',default='results/partA_fig.pdf')
a=ap.parse_args()
df=pd.read_csv(a.csv)
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
axes[0].scatter(df['trainable_pct'],df['val_entity_f1'])
axes[0].set_xscale('log')
axes[0].set_xlabel('Trainable parameters (%)')
axes[0].set_ylabel('Validation entity F1')
for _,r in df.iterrows(): axes[0].annotate(r['run_id'],(r['trainable_pct'],r['val_entity_f1']),xytext=(4,4),textcoords='offset points')
axes[1].scatter(df['peak_gpu_mem_gb'],df['val_entity_f1'])
axes[1].set_xlabel('Peak GPU memory (GB)')
axes[1].set_ylabel('Validation entity F1')
for _,r in df.iterrows(): axes[1].annotate(r['run_id'],(r['peak_gpu_mem_gb'],r['val_entity_f1']),xytext=(4,4),textcoords='offset points')
fig.tight_layout(); fig.savefig(a.png,dpi=200); fig.savefig(a.pdf); print('Saved',a.png,a.pdf)
