"""CPU-only v2 experiment, pinned Qwen3.5-0.8B and unchanged reference Clef."""
import os,sys,json,time,math,random,hashlib,resource,shutil
from pathlib import Path
from collections import defaultdict
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from transformers.utils import logging
from peft import LoraConfig,get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file,save_file
R=Path(__file__).parent;V1=R.parent/'tdcc_compass_cpu';sys.path.insert(0,str(R/'upstream'))
from joint_schema_model import JointSchemaHead,encode_record,collate_records
logging.disable_progress_bar()
SEED=872341;MODEL_REV='2fc06364715b967f1860aea9cf38778875588b17';DATA_REV='6a7927fa3c8ac114e29f5300eacb86f6f37a6447'
HEAD_CONFIG=dict(hidden_size=1024,width=512,routing_layers=2,layers=4,heads=8,feedforward=2048)
LORA_CONFIG=dict(r=8,lora_alpha=16,lora_dropout=0.,target_modules='all-linear',bias='none')
def key(v):return 'true' if v is True else 'false' if v is False else str(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(p)
def load_items(path,tok):
 items=[]
 for row in map(json.loads,Path(path).read_text().splitlines()):
  enc=encode_record(tok,row,max_length=50000)
  if len(enc.input_ids)>1536:raise ValueError(f"Truncation forbidden: {row['id']} has {len(enc.input_ids)} tokens")
  if 'tokens' in row and row['tokens']!=len(enc.input_ids):raise ValueError('Input token count mismatch')
  targets=[];labels=[]
  for q in enc.questions:
   gold=row['gold'][q.question_id];label=key(gold.get('label') if isinstance(gold,dict) else gold)
   if isinstance(gold,dict) and 'probabilities' in gold:
    d=gold['probabilities'];p=[d[k] for k in q.option_ids]
    if set(d)!=set(q.option_ids) or any(not math.isfinite(x) or x<0 for x in p) or abs(sum(p)-1)>1e-4:raise ValueError('Bad probability target')
    y=torch.tensor(p,dtype=torch.float32);y/=y.sum()
   else:y=torch.tensor([float(o==label) for o in q.option_ids])
   if label not in q.option_ids or not torch.isclose(y.sum(),torch.tensor(1.)):raise ValueError('Bad label')
   targets.append(y);labels.append(q.option_ids.index(label))
  items.append(dict(row=row,record=enc,targets=targets,labels=labels,id=row.get('id',row['metadata']['id'])))
 return items
class Engine:
 def __init__(self):
  torch.manual_seed(SEED);random.seed(SEED);torch.set_num_threads(6);torch.set_num_interop_threads(1)
  self.tok=AutoTokenizer.from_pretrained(R/'base_model',local_files_only=True)
  self.base=Qwen3_5ForConditionalGeneration.from_pretrained(R/'base_model',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).requires_grad_(False)
  self.text=get_peft_model(self.base.model.language_model,LoraConfig(**LORA_CONFIG))
  self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.adapter=[p for n,p in self.text.named_parameters() if 'lora_' in n]
  for p in self.adapter:p.requires_grad_(False)
  self.head=JointSchemaHead(**HEAD_CONFIG).float();self.embedding=self.base.get_output_embeddings().weight.detach();self.adapters_zero=True
  self.old_inputs={}
  for part in ['train','dev','calibration']:
   for row in map(json.loads,(V1/f'data/pilot/{part}.jsonl').read_text().splitlines()):
    old_enc=encode_record(self.tok,row,max_length=50000)
    self.old_inputs[row.get('id',row['metadata']['id'])]=hashlib.sha256(bytes(str(old_enc.input_ids),'utf8')).hexdigest()[:20]
 def load(self,p):
  p=Path(p);self.head.load_state_dict(load_file(str(p/'joint_head.safetensors')))
  state=load_file(str(p/'adapter/adapter_model.safetensors'));set_peft_model_state_dict(self.text,state)
  self.adapters_zero=all(bool((v==0).all()) for k,v in state.items() if 'lora_B' in k)
 def batch(self,item):
  b=collate_records([item['record']],self.tok.pad_token_id,torch.device('cpu'));b.pop('media');return b
 def feature(self,item,b):
  if not self.adapters_zero:raise RuntimeError('Frozen feature cache used with trained adapters')
  sig=hashlib.sha256(bytes(str(item['record'].input_ids),'utf8')).hexdigest()[:20]
  path=R/'feature_cache'/f"{item['id']}.{sig}.safetensors"
  if path.exists():return load_file(str(path))['hidden']
  old=V1/'feature_cache'/f"{item['id']}.safetensors"
  # Old cache is proven same pinned model, encoding and zero-adapter stage; require matching full encoded token fingerprint and token count.
  if old.exists() and self.old_inputs.get(item['id'])==sig:
   h=load_file(str(old))['hidden']
   if h.shape[1]==len(item['record'].input_ids):save_file({'hidden':h},str(path));return h
  self.text.eval()
  with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):h=self.text(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).last_hidden_state
  save_file({'hidden':h.contiguous()},str(path));return h
 def logits(self,item,cached=False):
  b=self.batch(item)
  h=self.feature(item,b) if cached else self.text(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).last_hidden_state
  return self.head(h.float(),b['input_ids'],b['attention_mask'],[item['record']],self.embedding)[0]
 def loss(self,zs,item,hard_weight=0.):
  return torch.stack([-( (1-hard_weight)*y+hard_weight*torch.nn.functional.one_hot(torch.tensor(k),len(y)) ) .mul(z.float().log_softmax(-1)).sum() for z,y,k in zip(zs,item['targets'],item['labels'])]).mean()
 def evaluate(self,items,cached=False):
  self.head.eval();self.text.eval();out=[]
  with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
   for item in items:out.append([z.float().detach().cpu() for z in self.logits(item,cached)])
  return out
 def save(self,p,extra,opt=None,state=None):
  p=Path(p);tmp=p.with_name(p.name+'.writing');old=p.with_name(p.name+'.previous')
  if tmp.exists():shutil.rmtree(tmp)
  tmp.mkdir(parents=True)
  self.text.save_pretrained(tmp/'adapter',safe_serialization=True)
  save_file({k:v.detach().cpu().contiguous() for k,v in self.head.state_dict().items()},str(tmp/'joint_head.safetensors'))
  atomic_json(tmp/'joint_head_config.json',HEAD_CONFIG)
  cfg=dict(format='tdcc-compass-cpu-clef-reference-v1',base_model='Qwen/Qwen3.5-0.8B',base_revision=MODEL_REV,dataset_revision=DATA_REV,max_length=2048,training_max_length=1536,backbone_dtype='bfloat16',head_parameter_dtype='float32',autocast='cpu_bfloat16',adapter_target='model.language_model',lora=LORA_CONFIG,temperature=1.0,seed=SEED,source_commit='0b1291bfd7c147e2edf82c085b8412c2630be4fe',reference_source='upstream/joint_schema_model.py',**extra)
  atomic_json(tmp/'compass_cpu_config.json',cfg);self.tok.save_pretrained(tmp/'tokenizer')
  if opt is not None:torch.save(dict(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),**state),tmp/'training_state.pt')
  if old.exists():shutil.rmtree(old)
  if p.exists():p.rename(old)
  tmp.rename(p)
  if old.exists():shutil.rmtree(old)
def metrics(items,zss,temp=1.):
 rows=[]
 for item,zs in zip(items,zss):
  m=item['row']['metadata'];group=m.get('dataset_group',m['source'])+':'+m['language']
  for q,y,k,z in zip(item['record'].questions,item['targets'],item['labels'],zs):
   lp=(z/temp).log_softmax(-1);p=lp.exp();rows.append(dict(correct=float(int(p.argmax())==k),hard_nll=float(-lp[k]),soft_cross_entropy=float(-(y*lp).sum()),group=group,domain=m['domain'],language=m['language'],type=['noul','choice','score'][q.question_type]))
 def summary(ss):return dict(n=len(ss),**{('accuracy' if k=='correct' else k):sum(x[k] for x in ss)/len(ss) for k in ['correct','hard_nll','soft_cross_entropy']})
 slices=defaultdict(list)
 for r in rows:
  for k in ['group','domain','language','type']:slices[k+':'+r[k]].append(r)
 out={'overall':summary(rows),'slices':{k:summary(v) for k,v in slices.items()}}
 orig=[r for r in rows if r['group']=='typed_decisions:en'];other=[v for k,v in slices.items() if k.startswith('group:') and k!='group:typed_decisions:en']
 orig_acc=sum(summary(v)['accuracy'] for k,v in slices.items() if k.startswith('domain:') and v[0]['group']=='typed_decisions:en')/4 if orig else 0
 if orig and other:score=.5*orig_acc+.5*sum(summary(v)['accuracy'] for v in other)/len(other)
 else:score=sum(summary(v)['accuracy'] for k,v in slices.items() if k.startswith('group:'))/sum(k.startswith('group:') for k in slices)
 out['selection_score']=score
 return out
