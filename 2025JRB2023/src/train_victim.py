#!/usr/bin/env python3
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def seed_all(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, rows, tok, max_length):
        self.rows=rows; self.tok=tok; self.max_length=max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ids=self.tok(str(self.rows[i]['text']), add_special_tokens=False,
                     truncation=True, max_length=max(1,self.max_length-1))['input_ids']
        if self.tok.eos_token_id is not None:
            ids=ids[:self.max_length-1]+[self.tok.eos_token_id]
        else:
            ids=ids[:self.max_length]
        return ids


class Collator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,batch):
        m=max(map(len,batch)); pad=self.tok.pad_token_id
        input_ids=[]; attn=[]; labels=[]
        for ids in batch:
            n=m-len(ids)
            input_ids.append(ids+[pad]*n)
            attn.append([1]*len(ids)+[0]*n)
            labels.append(ids+[-100]*n)
        return {'input_ids':torch.tensor(input_ids,dtype=torch.long),
                'attention_mask':torch.tensor(attn,dtype=torch.long),
                'labels':torch.tensor(labels,dtype=torch.long)}


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base=base
        for p in base.parameters(): p.requires_grad=False
        self.r=int(r); self.alpha=float(alpha); self.scaling=self.alpha/self.r
        self.dropout=nn.Dropout(float(dropout))
        self.lora_A=nn.Parameter(torch.empty(self.r,base.in_features,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,self.r,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5)); nn.init.zeros_(self.lora_B)
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(self.dropout(x),self.lora_A.to(x.dtype)),self.lora_B.to(x.dtype))*self.scaling


def inject_lora(model, targets, r, alpha, dropout):
    for p in model.parameters(): p.requires_grad=False
    targets=set(targets); matches=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and name.split('.')[-1] in targets:
            matches.append((name,module))
    if not matches: raise RuntimeError(f'No target modules found for {sorted(targets)}')
    names=[]
    for name,module in matches:
        if '.' in name:
            parent_name,child=name.rsplit('.',1); parent=model.get_submodule(parent_name)
        else:
            parent=model; child=name
        setattr(parent,child,LoRALinear(module,r,alpha,dropout)); names.append(name)
    return names


def count_params(model):
    total=sum(p.numel() for p in model.parameters())
    train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train,total


def save_adapter(model,out_dir,cfg):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    state={}; names=[]
    for name,m in model.named_modules():
        if isinstance(m,LoRALinear):
            names.append(name)
            state[f'{name}.lora_A']=m.lora_A.detach().cpu().contiguous()
            state[f'{name}.lora_B']=m.lora_B.detach().cpu().contiguous()
    if not state: raise RuntimeError('No LoRA state to save')
    save_file(state,str(out_dir/'adapter_model.safetensors'))
    cfg=dict(cfg); cfg['format']='manual_lora_v1'; cfg['injected_module_names']=names
    (out_dir/'adapter_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')


@torch.no_grad()
def token_weighted_ppl(model,loader,acc):
    model.eval(); nll=torch.tensor(0.0,device=acc.device,dtype=torch.float64); ntok=torch.tensor(0,device=acc.device,dtype=torch.long)
    for b in tqdm(loader,desc='heldout PPL',disable=not acc.is_local_main_process):
        o=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
        logits=o.logits[:,:-1,:].float(); y=b['labels'][:,1:]
        nll += F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100,reduction='sum').double()
        ntok += (y!=-100).sum()
    nll=acc.reduce(nll,reduction='sum'); ntok=acc.reduce(ntok,reduction='sum')
    mean=(nll/ntok).item(); ppl=math.exp(mean); model.train()
    return {'token_nll_sum':float(nll.item()),'token_count':int(ntok.item()),'mean_token_nll':float(mean),'ppl':float(ppl)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--corpus',required=True); p.add_argument('--heldout',required=True)
    p.add_argument('--target_modules',required=True); p.add_argument('--lora_r',type=int,required=True); p.add_argument('--lora_alpha',type=float,required=True)
    p.add_argument('--epochs',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--lora_dropout',type=float,default=0.05); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight_decay',type=float,default=0.0); p.add_argument('--warmup_ratio',type=float,default=0.03)
    p.add_argument('--max_length',type=int,default=512); p.add_argument('--per_device_batch_size',type=int,default=4)
    p.add_argument('--gradient_accumulation_steps',type=int,default=2); p.add_argument('--num_workers',type=int,default=2)
    a=p.parse_args(); seed_all(a.seed)
    if 'Instruct' in a.model: raise ValueError('Part C victim must be the base Qwen2.5-0.5B, not Instruct.')
    acc=Accelerator(gradient_accumulation_steps=a.gradient_accumulation_steps)
    train_rows=read_jsonl(a.corpus); held_rows=read_jsonl(a.heldout)
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype); model.config.use_cache=False
    targets=[x.strip() for x in a.target_modules.split(',') if x.strip()]
    injected=inject_lora(model,targets,a.lora_r,a.lora_alpha,a.lora_dropout)
    trainable,total=count_params(model)
    coll=Collator(tok)
    train_dl=DataLoader(TextDataset(train_rows,tok,a.max_length),batch_size=a.per_device_batch_size,shuffle=True,collate_fn=coll,num_workers=a.num_workers,pin_memory=True)
    held_dl=DataLoader(TextDataset(held_rows,tok,a.max_length),batch_size=a.per_device_batch_size,shuffle=False,collate_fn=coll,num_workers=a.num_workers,pin_memory=True)
    # IMPORTANT: FP32 trainable LoRA parameters
    # required for Accelerate FP16 GradScaler.
    for _p in model.parameters():
        if _p.requires_grad:
            _p.data = _p.data.float()
    
    _trainable_dtypes = {
        str(_p.dtype)
        for _p in model.parameters()
        if _p.requires_grad
    }
    
    if _trainable_dtypes != {'torch.float32'}:
        raise RuntimeError(
            f'Trainable parameters must be FP32, got {_trainable_dtypes}'
        )
    
    if acc.is_main_process:
        print(
            'Trainable LoRA dtype check: PASS',
            _trainable_dtypes
        )
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay)
    model,opt,train_dl,held_dl=acc.prepare(model,opt,train_dl,held_dl)
    upd_per_epoch=math.ceil(len(train_dl)/a.gradient_accumulation_steps); total_upd=upd_per_epoch*a.epochs; warm=int(total_upd*a.warmup_ratio)
    def lrf(step):
        if warm>0 and step<warm: return (step+1)/warm
        prog=min(1.0,max(0.0,(step-warm)/max(1,total_upd-warm)))
        return 0.5*(1+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lrf)
    if acc.is_main_process:
        print('='*80); print('PART C VICTIM TRAINING'); print('='*80)
        print('model              :',a.model); print('corpus             :',a.corpus); print('train/heldout rows :',len(train_rows),'/',len(held_rows))
        print('LoRA modules       :',len(injected)); print('r/alpha/dropout    :',a.lora_r,a.lora_alpha,a.lora_dropout)
        print('trainable params   :',f'{trainable:,}'); print('trainable pct      :',f'{100*trainable/total:.6f}%')
        print('epochs/lr          :',a.epochs,a.lr); print('batch/GPU          :',a.per_device_batch_size); print('grad accumulation  :',a.gradient_accumulation_steps)
        print('world size         :',acc.num_processes); print('effective batch    :',a.per_device_batch_size*a.gradient_accumulation_steps*acc.num_processes)
        print('max_length/seed    :',a.max_length,a.seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); loss_sum=0.0; loss_steps=0; seen_tokens=0
    model.train()
    for epoch in range(a.epochs):
        ep=0.0; n=0
        bar=tqdm(train_dl,desc=f'epoch {epoch+1}/{a.epochs}',disable=not acc.is_local_main_process)
        for b in bar:
            with acc.accumulate(model):
                o=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],labels=b['labels'],use_cache=False); loss=o.loss
                acc.backward(loss)
                if acc.sync_gradients: acc.clip_grad_norm_(params,1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if acc.sync_gradients: sched.step()
            gl=acc.gather(loss.detach().float().reshape(1)); ml=float(gl.mean().item()); ep+=ml; n+=1; loss_sum+=ml; loss_steps+=1
            gtok=acc.reduce(b['attention_mask'].sum(),reduction='sum'); seen_tokens+=int(gtok.item())
            if acc.is_local_main_process: bar.set_postfix(loss=f'{ml:.4f}',lr=f'{sched.get_last_lr()[0]:.2e}')
        if acc.is_main_process: print(f'epoch {epoch+1}: mean_train_loss={ep/max(1,n):.6f}')
    acc.wait_for_everyone(); wall=time.perf_counter()-start
    ppl=token_weighted_ppl(model,held_dl,acc); acc.wait_for_everyone()
    unwrapped=acc.unwrap_model(model); out=Path(a.output_dir)
    cfg={'base_model_name_or_path':a.model,'target_modules':targets,'r':a.lora_r,'lora_alpha':a.lora_alpha,'lora_dropout':a.lora_dropout,
         'epochs':a.epochs,'seed':a.seed,'lr':a.lr,'weight_decay':a.weight_decay,'warmup_ratio':a.warmup_ratio,'max_length':a.max_length,
         'per_device_batch_size':a.per_device_batch_size,'gradient_accumulation_steps':a.gradient_accumulation_steps,'corpus':a.corpus,'heldout':a.heldout}
    if acc.is_main_process: save_adapter(unwrapped,out,cfg)
    if torch.cuda.is_available():
        peak=acc.reduce(torch.tensor(torch.cuda.max_memory_allocated()/(1024**3),device=acc.device,dtype=torch.float64),reduction='max').item()
    else: peak=0.0
    metrics={'model':a.model,'corpus':a.corpus,'heldout':a.heldout,'n_train_rows':len(train_rows),'n_heldout_rows':len(held_rows),
             'target_modules':targets,'matched_lora_modules':len(injected),'lora_r':a.lora_r,'lora_alpha':a.lora_alpha,'lora_dropout':a.lora_dropout,
             'trainable_params':trainable,'total_params_including_lora':total,'trainable_pct':100*trainable/total,'epochs':a.epochs,'seed':a.seed,
             'lr':a.lr,'max_length':a.max_length,'per_device_batch_size':a.per_device_batch_size,'gradient_accumulation_steps':a.gradient_accumulation_steps,
             'world_size':acc.num_processes,'effective_batch_size':a.per_device_batch_size*a.gradient_accumulation_steps*acc.num_processes,
             'train_wallclock_s':wall,'approx_tokens_per_s':seen_tokens/max(wall,1e-9),'mean_train_loss_over_microsteps':loss_sum/max(1,loss_steps),
             'peak_gpu_mem_gb':float(peak),'heldout_token_weighted':ppl}
    if acc.is_main_process:
        out.mkdir(parents=True,exist_ok=True); (out/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('\n'+'='*80); print('TRAINING COMPLETE'); print('='*80)
        print('output_dir        :',out); print('train_wallclock_s :',f'{wall:.2f}'); print('mean_train_loss   :',f"{metrics['mean_train_loss_over_microsteps']:.6f}")
        print('peak_gpu_mem_gb   :',f'{peak:.4f}'); print('heldout mean NLL  :',f"{ppl['mean_token_nll']:.6f}"); print('heldout PPL       :',f"{ppl['ppl']:.6f}")
        print('adapter weights   :',out/'adapter_model.safetensors'); print('metrics           :',out/'training_metrics.json')

if __name__=='__main__': main()
