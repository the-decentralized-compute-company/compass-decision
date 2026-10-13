#!/usr/bin/env python3
"""Data integrity and token audit. Never loads final test examples."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--compass-code',type=Path,default=R.parent/'tdcc_compass')
parser.add_argument('--cpu-project',type=Path,default=R.parent/'tdcc_compass_cpu')
args=parser.parse_args()
sys.path[:0]=[str(args.compass_code),str(args.cpu_project/'upstream')]
from compass.data import validate_row
from joint_schema_model import encode_record
from transformers import AutoTokenizer
from prepare_extra import HELDOUT_DOMAINS,load_massive,massive_families,slot_template,norm

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 tok=AutoTokenizer.from_pretrained(args.cpu_project/'base_model',local_files_only=True)
 manifest=json.loads((R/'data/manifest.json').read_text()); summary={};groups={};texts={};all_ids=set()
 for part in ['train','dev','calibration']:
  p=R/'data'/f'extra_{part}.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];assert sha(p)==manifest['partitions'][part]['sha256']
  groups[part]=set();texts[part]=set();lengths=collections.defaultdict(list);group_rows=collections.defaultdict(list)
  for row in rows:
   validate_row(row);id=row['id'];assert id not in all_ids;all_ids.add(id);m=row['metadata'];groups[part].add(m['group_id']);group_rows[m['group_id']].append(row)
   texts[part].add(hashlib.sha256(json.dumps(row['state'],sort_keys=True,ensure_ascii=False).encode()).hexdigest())
   assert m['domain'] not in HELDOUT_DOMAINS
   assert m['official_partition']==('train' if part=='train' else 'dev')
   n=len(encode_record(tok,row,max_length=100000).input_ids);lengths[m['dataset_group']+'/'+m['language']].append(n)
  for family,rs in group_rows.items():
   assert len(rs)==2 and {r['metadata']['language'] for r in rs}=={'en','tr'}
   assert len({r['metadata']['source_id'] for r in rs})==1
  summary[part]={k:{'records':len(v),'min':min(v),'mean':sum(v)/len(v),'max':max(v),'above1536':sum(x>1536 for x in v)} for k,v in lengths.items()}
 for a in groups:
  for b in groups:
   if a>=b:continue
   assert not groups[a]&groups[b],(a,b,'groups');assert not texts[a]&texts[b],(a,b,'exact_states')
 held=json.loads((R/'data/snli_official_heldout_families.json').read_text())
 for p in groups:assert not groups[p]&set(held['test'])
 assert not groups['train']&set(held['dev'])
 fam=json.loads((R/'data/massive_families.json').read_text())['families']
 for p in groups:
  for f in groups[p]:
   if f.startswith('massive:'):
    assert 'test' not in fam[f]['partitions'];assert not set(fam[f]['domains'])&set(HELDOUT_DOMAINS)
    if p=='train':assert 'dev' not in fam[f]['partitions']
 out={'passed':True,'tests':['row schema validity','unique row IDs','translation pairing','disjoint original source/template/image families','disjoint exact states','official partition preservation','no reserved domain in train/dev/calibration','all official-test families excluded','no final-test labels/examples selected','no truncation needed at1536 on train/dev/calibration'],'token_lengths':summary}
 assert all(s['above1536']==0 for d in summary.values() for s in d.values())
 (R/'reports/data_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
