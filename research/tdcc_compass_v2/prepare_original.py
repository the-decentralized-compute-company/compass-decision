import sys,json,random,hashlib,collections
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(__file__).parent;V1=R.parent/'tdcc_compass_cpu'
sys.path.insert(0,str(R/'upstream'))
from joint_schema_model import encode_record
SEED=872341
out=R/'data'
tok=AutoTokenizer.from_pretrained(R/'base_model',local_files_only=True)
v1train={r['metadata']['id'] for r in map(json.loads,(V1/'data/pilot/train.jsonl').read_text().splitlines())}
raw=pq.read_table(V1/'data/typed-decisions/all/train-00000-of-00001.parquet').to_pylist()
by=collections.defaultdict(list)
for r in raw:
 row={k:json.loads(r[k]) for k in ['state','questions','gold']};row['id']=r['id'];row['metadata']={'id':r['id'],'group_id':r['id'],'domain':r['workflow'],'language':'en','source':'LocalLLaMA/typed-decisions','dataset_group':'typed_decisions','demo_only':False,'official_split':'train'}
 row['tokens']=len(encode_record(tok,row,max_length=50000).input_ids)
 assert row['tokens']<=1536
 by[r['workflow']].append(row)
splits={'train':[],'dev':[],'calibration':[]}
for domain,rows in sorted(by.items()):
 eligible=sorted([r for r in rows if r['id'] not in v1train],key=lambda r:r['id']);random.Random(SEED+sum(domain.encode())).shuffle(eligible)
 dev=eligible[:30];cal=eligible[30:60];held={r['id'] for r in dev+cal}
 splits['dev']+=dev;splits['calibration']+=cal;splits['train']+=[r for r in rows if r['id'] not in held and r['id']!='tr_security_incidents_000063']
manifest={'seed':SEED,'source':'LocalLLaMA/typed-decisions','dataset_revision':'6a7927fa3c8ac114e29f5300eacb86f6f37a6447','notes':['All rows originate from official train; official test is owned by independent evaluator.','Dev and calibration exclude every v1 training record.','Previous v1 original-test rows are development-contaminated and never enter v2 training.','Public labels are teacher-generated, not independently verified correctness.','Excluded original-train ID tr_security_incidents_000063 after evaluator found an exact factors overlap with a sealed test record; no test labels were consulted.'],'splits':{}}
for part,rows in splits.items():
 random.Random(SEED).shuffle(rows);p=out/f'original_{part}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 manifest['splits'][part]={'records':len(rows),'decisions':sum(len(r['questions']) for r in rows),'ids':[r['id'] for r in rows],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'max_tokens':max(r['tokens'] for r in rows),'mean_tokens':sum(r['tokens'] for r in rows)/len(rows)}
(out/'original_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({p:{k:v for k,v in x.items() if k!='ids'} for p,x in manifest['splits'].items()},indent=2))
