import json,hashlib,collections,random
from pathlib import Path
R=Path(__file__).parent;X=R.parent/'tdcc_compass_bilingual/data'
manifest={'seed':872341,'source_manifests':['data/original_manifest.json','../tdcc_compass_bilingual/MANIFEST.json'],'splits':{},'excluded_typed_train_ids':['tr_security_incidents_000063'],'fresh_test_policy':'Owned by independent evaluator; never read by trainer.'}
splits={}
for part in ['train','dev','calibration']:
 paths=[R/f'data/original_{part}.jsonl',X/f'extra_{part}.jsonl'];rows=[]
 for p in paths:rows+=list(map(json.loads,p.read_text().splitlines()))
 random.Random(872341).shuffle(rows);splits[part]=rows
for a,rows in splits.items():
 ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))
 for b,rr in splits.items():
  if a>=b:continue
  assert not ({r['metadata']['group_id'] for r in rows}&{r['metadata']['group_id'] for r in rr})
  assert not ({json.dumps(r['state'],sort_keys=True,ensure_ascii=False).lower() for r in rows}&{json.dumps(r['state'],sort_keys=True,ensure_ascii=False).lower() for r in rr})
 p=R/f'data/{a}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 manifest['splits'][a]={'records':len(rows),'decisions':sum(len(r['questions']) for r in rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ids':ids,'language_records':dict(collections.Counter(r['metadata']['language'] for r in rows)),'dataset_records':dict(collections.Counter(r['metadata']['dataset_group'] for r in rows))}
(R/'data/manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(json.dumps({k:{f:v for f,v in x.items() if f!='ids'} for k,x in manifest['splits'].items()},indent=2))
