"""Load a Compass CPU reference checkpoint and score arbitrary typed schemas."""
import os,sys,json,argparse,math,hashlib
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
R=Path(__file__).resolve().parent

def _load_runtime():
 global torch, AutoTokenizer, Qwen3_5ForConditionalGeneration, PeftModel, load_file
 global JointSchemaHead, encode_record, collate_records
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from transformers.utils import logging
 from peft import PeftModel
 from safetensors.torch import load_file
 sys.path.insert(0,str(R/'upstream'))
 from joint_schema_model import JointSchemaHead,encode_record,collate_records
 logging.disable_progress_bar()

def default_checkpoint():
 return R if (R/'compass_cpu_config.json').is_file() else R/'model'

class CompassCPU:
 def __init__(self,checkpoint,base_model=None,threads=6):
  _load_runtime()
  self.path=Path(checkpoint);self.config=json.loads((self.path/'compass_cpu_config.json').read_text())
  if self.config['format']!='tdcc-compass-cpu-clef-reference-v1':raise ValueError('Unsupported checkpoint format')
  if base_model is None:base_model=R/'base_model'
  for filename,wanted in self.config.get('base_model_sha256',{}).items():
   path=Path(base_model)/filename
   with path.open('rb') as f:
    got=hashlib.file_digest(f,'sha256').hexdigest()
   if got!=wanted:raise ValueError(f'Base model checksum mismatch for {filename}')
  torch.set_num_threads(threads)
  self.tokenizer=AutoTokenizer.from_pretrained(self.path/'tokenizer',local_files_only=True)
  self.base=Qwen3_5ForConditionalGeneration.from_pretrained(base_model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).requires_grad_(False)
  self.text=PeftModel.from_pretrained(self.base.model.language_model,self.path/'adapter',local_files_only=True,is_trainable=False).eval()
  hc=json.loads((self.path/'joint_head_config.json').read_text());self.head=JointSchemaHead(**hc).float()
  self.head.load_state_dict(load_file(str(self.path/'joint_head.safetensors')),strict=True);self.head.eval()
  self.temperature=float(self.config.get('temperature',1.0))
  if not math.isfinite(self.temperature) or self.temperature<=0:raise ValueError('Invalid calibration temperature')
 def predict(self,row):
  if not isinstance(row,dict) or 'state' not in row or not isinstance(row.get('questions'),dict) or not row['questions']:raise ValueError('Expected state and nonempty questions mapping')
  for qid,q in row['questions'].items():
   if not isinstance(qid,str) or not qid or not isinstance(q,dict) or q.get('type') not in {'noul','choice','score'}:raise ValueError('Unsupported question')
   if q['type']=='choice' and (not isinstance(q.get('criteria'),dict) or len(q['criteria'])<2):raise ValueError('Choice requires at least two options')
   if q['type']=='score' and (not isinstance(q.get('criteria'),list) or len(q['criteria'])<2):raise ValueError('Score requires at least two ordered levels')
  # Encode completely first; never silently truncate the state.
  record=encode_record(self.tokenizer,{'state':row['state'],'questions':row['questions']},max_length=1000000)
  if len(record.input_ids)>self.config['max_length']:raise ValueError(f"Input has {len(record.input_ids)} tokens; checkpoint limit is {self.config['max_length']}. No decision was made.")
  batch=collate_records([record],self.tokenizer.pad_token_id,torch.device('cpu'))
  with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
   hidden=self.text(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
   values=self.head(hidden.float(),batch['input_ids'],batch['attention_mask'],[record],self.base.get_output_embeddings().weight.detach())[0]
  answers={}
  for question,z in zip(record.questions,values):
   p=(z.float()/self.temperature).softmax(-1)
   if not bool(torch.isfinite(p).all()):raise ValueError('Nonfinite prediction; no decision made')
   answer=question.option_ids[int(p.argmax())]
   qt=['noul','choice','score'][question.question_type]
   value={'type':qt,'logits':{k:float(v) for k,v in zip(question.option_ids,z.float())},'uncalibrated_probabilities':{k:float(v) for k,v in zip(question.option_ids,z.float().softmax(-1))},'probabilities':{k:float(v) for k,v in zip(question.option_ids,p)},'answer':answer=='true' if qt=='noul' else int(answer) if qt=='score' else answer}
   if qt=='score':value['score']=sum(int(k)*float(v) for k,v in zip(question.option_ids,p))
   answers[question.question_id]=value
  return {'answers':answers,'input_tokens':len(record.input_ids),'pilot_warning':'Experimental typed-decision model. Benchmark performance does not establish real-world reliability, and probabilities are not safety guarantees.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default=str(default_checkpoint()));ap.add_argument('--base-model');ap.add_argument('--threads',type=int,default=6);ap.add_argument('--input',required=True);ap.add_argument('--output');args=ap.parse_args()
 model=CompassCPU(args.checkpoint,args.base_model,args.threads)
 rows=[json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
 result=[model.predict(row) for row in rows]
 text='\n'.join(json.dumps(x,ensure_ascii=False) for x in result)+'\n'
 if args.output:Path(args.output).write_text(text,encoding='utf8')
 else:print(text,end='')
if __name__=='__main__':main()
