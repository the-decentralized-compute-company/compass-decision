#!/usr/bin/env python3
"""Prepare deterministic bilingual training data; does not select final-test examples.
No paid API, dataset execution, new translation, or label generation is used.
"""
import argparse, collections, hashlib, heapq, itertools, json, re, sys, unicodedata, zipfile
from pathlib import Path
from schemas import INTENT_IDS, MASSIVE_DESCRIPTIONS, massive_questions, nli_questions
R=Path(__file__).resolve().parent
SEED='compass-v2-bilingual-3407'
HELDOUT_DOMAINS=('general','news','datetime')
LANGUAGES=('en','tr')

def sha(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def rank(value): return sha(SEED+'|'+str(value))
def norm(x): return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
def slot_template(x): return norm(re.sub(r'\[\s*([^:\[\]]+)\s*:[^\[\]]*\]',lambda m:'['+m.group(1).strip()+']',x))
def dump(path,x): Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write_rows(path,rows):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows));temporary.replace(path)
def image_family(row): return 'snli:image:'+row['captionID'].split('#')[0]
def iter_snli(language,partition):
    filename,member=(('snli_1.0.zip',f'snli_1.0/snli_1.0_{partition}.jsonl') if language=='en' else ('snli_tr_1.1.zip',f'snli_tr_1.1_{partition}.jsonl'))
    with zipfile.ZipFile(R/'raw'/filename) as z:
        with z.open(member) as f:
            for line in f: yield json.loads(line)

def load_massive():
    locales={l:{r['id']:r for r in map(json.loads,(R/'raw'/f'{loc}.jsonl').read_text().splitlines())} for l,loc in [('en','en-US'),('tr','tr-TR')]}
    assert locales['en'].keys()==locales['tr'].keys()
    for id in locales['en']:
        a,b=locales['en'][id],locales['tr'][id]
        assert all(a[k]==b[k] for k in ('partition','scenario','intent'))
    return locales

def massive_families(locales):
    ids=list(locales['en']); parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for lang,rows in locales.items():
        for id,row in rows.items():
            for typ,value in [('text',norm(row['utt'])),('template',slot_template(row['annot_utt']))]:
                key=(lang,typ,value)
                if key in seen:union(id,seen[key])
                else:seen[key]=id
    members=collections.defaultdict(list)
    for id in ids:members[find(id)].append(id)
    ids_to_family={}; families={}
    for members_list in members.values():
        members_list.sort(); family='massive:family:'+sha('|'.join(members_list))[:20]
        ids_to_family.update({id:family for id in members_list})
        families[family]={'ids':members_list,'partitions':sorted({locales['en'][i]['partition'] for i in members_list}),'domains':sorted({locales['en'][i]['scenario'] for i in members_list})}
    return ids_to_family,families

def massive_quality(row):
    judgments=row.get('judgments',[])
    return len(judgments)>=3 and sum(j['intent_score']==1 for j in judgments)>=2 and sum(j['grammar_score'] for j in judgments)/len(judgments)>=3

def massive_record(row,language,family):
    id='massive:'+row['id']+':'+language
    return {'id':id,'state':row['utt'],'questions':massive_questions(language),'gold':{'intent':{'label':INTENT_IDS[row['intent']]}},'metadata':{'id':id,'group_id':family,'source_id':row['id'],'language':language,'domain':row['scenario'],'source':'Amazon MASSIVE 1.1','dataset_group':'massive','official_partition':row['partition'],'source_label':row['intent'],'annotation_provenance':'Original corpus intent; TR professionally localized in source corpus; schema descriptions authored for this conversion.','license':'CC-BY-4.0','demo_only':False,'synthetic':False}}

def nli_record(row,language):
    id='snli:'+row['pairID']+':'+language
    return {'id':id,'state':{'premise':row['sentence1'],'hypothesis':row['sentence2']},'questions':nli_questions(language),'gold':{'relation':{'label':row['gold_label']}},'metadata':{'id':id,'group_id':image_family(row),'source_id':row['pairID'],'language':language,'domain':'caption_nli','source':'SNLI 1.0' if language=='en' else 'SNLI-TR 1.1','dataset_group':'snli','annotation_provenance':'Original SNLI crowd label; Turkish text machine-translated by original NLI-TR authors, not independently reviewed in this work.','license':'CC-BY-SA-4.0','demo_only':False,'synthetic':False,'translation':'none' if language=='en' else 'source-corpus machine translation'}}

def pick_massive(locales,id_to_family,families,partition,count_per_domain,used):
    result=[]; exclusions=collections.Counter()
    domains=sorted(set(r['scenario'] for r in locales['en'].values())-set(HELDOUT_DOMAINS))
    for domain in domains:
        buckets=collections.defaultdict(list)
        for id,row in locales['en'].items():
            if row['partition']!=partition or row['scenario']!=domain:continue
            family=id_to_family[id]; info=families[family]
            if family in used:continue
            # No record is moved between official partitions. Earlier overlapping groups are dropped.
            if 'test' in info['partitions'] or (partition=='train' and 'dev' in info['partitions']):exclusions['later_partition_family']+=1;continue
            if set(info['domains'])&set(HELDOUT_DOMAINS):exclusions['heldout_domain_family']+=1;continue
            if not massive_quality(locales['tr'][id]):exclusions['turkish_source_quality_filter']+=1;continue
            buckets[row['intent']].append(id)
        for b in buckets.values():b.sort(key=rank)
        selected=[]
        # Predeclared equal-domain quota and round-robin intent coverage, only for training/development.
        while len(selected)<count_per_domain:
            advanced=False
            for label in sorted(buckets,key=rank):
                while buckets[label] and id_to_family[buckets[label][0]] in used:buckets[label].pop(0)
                if buckets[label] and len(selected)<count_per_domain:
                    id=buckets[label].pop(0);selected.append(id);used.add(id_to_family[id]);advanced=True
            if not advanced:break
        if len(selected)!=count_per_domain:raise ValueError(f'Insufficient independent MASSIVE families: {partition}/{domain}: {len(selected)} < {count_per_domain}')
        for id in selected:
            result.extend(massive_record(locales[l][id],l,id_to_family[id]) for l in LANGUAGES)
    return result,dict(exclusions)

def pick_snli(partition,per_label,forbidden,used):
    labels=('entailment','neutral','contradiction'); heaps={k:[] for k in labels}; stats=collections.Counter()
    # Keep a bounded deterministic ranked candidate pool; no final-test predictions or labels used.
    for en,tr in itertools.zip_longest(iter_snli('en',partition),iter_snli('tr',partition)):
        assert en is not None and tr is not None and en['pairID']==tr['pairID']
        label=en['gold_label'];family=image_family(en)
        if label not in labels or tr['gold_label']!=label:stats['invalid_gold']+=1;continue
        if family in forbidden or family in used:stats['family_excluded']+=1;continue
        if partition=='train' and en.get('annotator_labels',[]).count(label)<3:stats['not_three_vote_gold_agreement']+=1;continue
        if not en['sentence1'].strip() or not tr['sentence1'].strip() or not en['sentence2'].strip() or not tr['sentence2'].strip():continue
        # top rank uses unique pairID as the stable deterministic sampling unit; families dedup below.
        order=int(rank(en['pairID']),16); entry=(-order,en['pairID'],en,tr)
        heap=heaps[label]
        if len(heap)<2000:heapq.heappush(heap,entry)
        elif entry>heap[0]:heapq.heapreplace(heap,entry)
        stats['eligible']+=1
    result=[]
    for label in labels:
        selected=0
        for _,_,en,tr in sorted(heaps[label],reverse=True):
            family=image_family(en)
            if family in used:continue
            used.add(family)
            for l,row in [('en',en),('tr',tr)]:
                out=nli_record(row,l);out['metadata']['official_partition']=partition;result.append(out)
            selected+=1
            if selected==per_label:break
        if selected!=per_label:raise ValueError(f'Insufficient SNLI families: {partition}/{label}')
    return result,dict(stats)

def describe(rows):
    return {'records':len(rows),'decisions':sum(len(r['questions']) for r in rows),'source_groups':len({r['metadata']['group_id'] for r in rows}),'language':dict(collections.Counter(r['metadata']['language'] for r in rows)),'dataset_group':dict(collections.Counter(r['metadata']['dataset_group'] for r in rows)),'domain':dict(collections.Counter(r['metadata']['domain'] for r in rows))}

def main():
    (R/'data').mkdir(exist_ok=True);(R/'reports').mkdir(exist_ok=True)
    locales=load_massive();id_to_family,families=massive_families(locales)
    dump(R/'data/massive_families.json',{'ids_to_family':id_to_family,'families':families})
    stats={};splits={};used=set()
    for part,original,count in [('train','train',20),('dev','dev',2),('calibration','dev',2)]:
        splits[part],stats['massive_'+part]=pick_massive(locales,id_to_family,families,original,count,used)
    # Family-only scan of official dev/test prevents image-story leakage; no test examples selected or shown.
    snli_families={p:{image_family(r) for r in iter_snli('en',p)} for p in ['dev','test']}
    dump(R/'data/snli_official_heldout_families.json',{p:sorted(v) for p,v in snli_families.items()})
    used=set()
    for part,original,count in [('train','train',50),('dev','dev',3),('calibration','dev',3)]:
        forbidden=snli_families['test']|(snli_families['dev'] if part=='train' else set())
        rows,stats['snli_'+part]=pick_snli(original,count,forbidden,used);splits[part]+=rows
    initial_ids={part:{r['id'] for r in rows} for part,rows in splits.items()}
    # Pre-optimization amendment: deterministic supersets of the initial 78-row dev/cal sets.
    # Reserve BOTH initial sets before adding rows so no prior assignment ever changes.
    massive_used={r['metadata']['group_id'] for rows in splits.values() for r in rows if r['metadata']['dataset_group']=='massive'}
    snli_used={r['metadata']['group_id'] for rows in splits.values() for r in rows if r['metadata']['dataset_group']=='snli'}
    for part in ['dev','calibration']:
        rows,stats['massive_'+part+'_expansion']=pick_massive(locales,id_to_family,families,'dev',2,massive_used);splits[part]+=rows
        rows,stats['snli_'+part+'_expansion']=pick_snli('dev',17,snli_families['test'],snli_used);splits[part]+=rows
    manifest={'schema_version':1,'seed':SEED,'heldout_massive_domains':list(HELDOUT_DOMAINS),'intent_ids':INTENT_IDS,'partitions':{},'filters':stats,'notes':['Original test records are not selected by this script. Evaluation is independently owned.','MASSIVE exact normalized text and slot-annotated templates in either language are unioned with bilingual source IDs. Earlier official partitions sharing a family with later ones are excluded, never reassigned.','SNLI source image families are indivisible, including all captions and their translations. All official-test image families excluded from training/development/calibration; all original-dev image families excluded from training.','SNLI-TR contains source-corpus machine translations, not newly commissioned or human-reviewed translations.','Schema descriptions are authored conversion glue; labels are inherited from source corpora. One-hot targets encode source labels, not calibrated certainty.','Dataset is limited to intent classification and caption-based natural language inference. It does not establish broad real-world decision competence.']}
    for part,rows in splits.items():
        rows.sort(key=lambda r:rank(r['id']));path=R/'data'/f'extra_{part}.jsonl';write_rows(path,rows)
        manifest['partitions'][part]={**describe(rows),'path':str(path),'sha256':sha(path.read_bytes()),'ids':[r['id'] for r in rows],'group_ids':sorted({r['metadata']['group_id'] for r in rows})}
    for part,rows in splits.items():
        assert initial_ids[part] <= {r['id'] for r in rows}
    dump(R/'reports/superset_expansion.json',{'before_optimization':True,'initial_records':{p:len(ids) for p,ids in initial_ids.items()},'final_records':{p:len(rows) for p,rows in splits.items()},'all_initial_ids_retained_in_same_partition':True,'train_ids_unchanged':initial_ids['train']=={r['id'] for r in splits['train']},'initial_ids':{p:sorted(ids) for p,ids in initial_ids.items()}})
    dump(R/'data/manifest.json',manifest);dump(R/'reports/preparation_stats.json',stats)
    print(json.dumps({p:describe(r) for p,r in splits.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
