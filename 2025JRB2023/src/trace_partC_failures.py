#!/usr/bin/env python3
"""Reproduce redacted V-san targeted-leak traces without storing PII values."""
import argparse, json, re
from pathlib import Path
import torch
from tqdm.auto import tqdm
import attack_extract as ae


def read_jsonl(p):
    with open(p,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def norm(x): return re.sub(r'\s+',' ',str(x)).strip().casefold()

ap=argparse.ArgumentParser()
ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B')
ap.add_argument('--adapter',default='adapters/victim_san')
ap.add_argument('--targeted',default='data/partC_targeted.jsonl')
ap.add_argument('--san_corpus',default='data/partC_corpus_sanitized.jsonl')
ap.add_argument('--output',default='results/detail/partC_failure_traces.json')
ap.add_argument('--device',default='cuda:0')
a=ap.parse_args()
rows=read_jsonl(a.targeted); san=read_jsonl(a.san_corpus); san_by={str(r['doc_id']):str(r['text']) for r in san if str(r['doc_id']) not in {}}
tok,model=ae.load_model(a.model,a.adapter,a.device); tok.padding_side='left'; B=8; N=64
greedy={}
with torch.inference_mode():
  for s in tqdm(range(0,len(rows),B),desc='trace greedy'):
    b=rows[s:s+B]; prompts=[str(x['prompt']) for x in b]; enc=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False); ids=enc['input_ids'].to(a.device); mask=enc['attention_mask'].to(a.device)
    out=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=N,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
    texts=tok.batch_decode(out[:,ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    for r,t in zip(b,texts): greedy[str(r['id'])]=norm(r['gold_value']) in norm(t)
ae.set_seed(42+100003); sampled={}
with torch.inference_mode():
  for s in tqdm(range(0,len(rows),B),desc='trace sampled-any10'):
    b=rows[s:s+B]; prompts=[str(x['prompt']) for x in b]; enc=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False); ids=enc['input_ids'].to(a.device); mask=enc['attention_mask'].to(a.device)
    out=model.generate(input_ids=ids,attention_mask=mask,do_sample=True,temperature=1.0,top_p=0.95,top_k=0,num_return_sequences=10,max_new_tokens=N,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
    texts=tok.batch_decode(out[:,ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    for j,r in enumerate(b):
      g=norm(r['gold_value']); sampled[str(r['id'])]=any(g in norm(t) for t in texts[j*10:(j+1)*10])
trace=[]
for r in rows:
    rid=str(r['id']); did=str(r['doc_id']); g=norm(r['gold_value']); survived=bool(g) and g in norm(san_by.get(did,''))
    if str(r['split']).lower()=='member' and survived:
        trace.append({'id':rid,'doc_id':did,'label':r['label'],'sanitizer_survived':True,'greedy_hit':bool(greedy[rid]),'sampled_any10_hit':bool(sampled[rid]),'pii_value':'[REDACTED]'})
obj={'summary':{'n_member_targets':sum(str(r['split']).lower()=='member' for r in rows),'n_gold_values_surviving_sanitizer':len(trace),'survival_rate':len(trace)/max(1,sum(str(r['split']).lower()=='member' for r in rows)),'n_survivors_with_observed_targeted_leak':sum(x['greedy_hit'] or x['sampled_any10_hit'] for x in trace)},'traces':trace}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(obj,indent=2),encoding='utf-8'); print(json.dumps(obj,indent=2))
