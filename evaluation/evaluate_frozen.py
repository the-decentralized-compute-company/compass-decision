"""Independent full-precision scorer. No model loading or training dependencies.

Usage:
  python evaluate_frozen.py --labels public_fresh_labels.jsonl \
    --predictions v2=... v1=... v1_soft_prior=... v2_soft_prior=... v2_hard_prior=... \
    --output results.json --primary-gate
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
BASELINES=('v1','v1_soft_prior','v2_soft_prior','v2_hard_prior')
BOOTSTRAPS=20000
SEED=48021

def read(path):
 p=Path(path);s=p.read_text();return json.loads(s) if s.lstrip().startswith('[') else [json.loads(x) for x in s.splitlines() if x.strip()]
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def identifier(r):return r.get('id',r.get('metadata',{}).get('id'))
def labelkey(x):return 'true' if x is True else 'false' if x is False else str(x)
def options(q):
 if q['type']=='noul':return ['true','false']
 if q['type']=='choice':return sorted(str(k) for k in q['criteria'])
 if q['type']=='score':return [str(i) for i in range(len(q['criteria']))]
 raise ValueError('Unknown type')
def gold_values(g,q):
 opts=options(q);label=labelkey(g.get('label') if isinstance(g,dict) else g)
 if label not in opts:raise ValueError('Label missing from options')
 if isinstance(g,dict) and 'probabilities' in g:
  if set(g['probabilities'])!=set(opts):raise ValueError('Gold option mismatch')
  p=np.array([g['probabilities'][k] for k in opts],dtype=np.float64)
  if not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-5:raise ValueError('Invalid target probability')
  p/=p.sum() # Public target probabilities are rounded to six places.
 else:p=np.array([float(k==label) for k in opts])
 return opts,label,p

def schema(qid,q):return canonical([qid,q])
def priors(train,inputs,mode):
 totals={};counts=collections.Counter()
 for r in train:
  for qid,q in r['questions'].items():
   opts,label,y=gold_values(r['gold'][qid],q);sk=schema(qid,q)
   if sk not in totals:totals[sk]=np.ones(len(opts)) if mode=='hard' else np.zeros(len(opts))
   totals[sk]+=np.array([float(k==label) for k in opts]) if mode=='hard' else y
   counts[sk]+=1
 out=[];matched=0;decisions=0
 for r in inputs:
  aa={}
  for qid,q in r['questions'].items():
   opts=options(q);sk=schema(qid,q);p=totals.get(sk,np.ones(len(opts))).copy();p/=p.sum()
   matched+=int(sk in totals);decisions+=1
   aa[qid]={'answer':opts[int(p.argmax())],'probabilities':dict(zip(opts,p.tolist()))}
  out.append({'id':identifier(r),'answers':aa})
 return out,{'training_records':len(train),'schemas':len(totals),'matched_test_decisions':matched,'test_decisions':decisions,'coverage':matched/decisions,'mode':mode,'hard_prior_pseudocount_per_option':1 if mode=='hard' else 0}

def score(rows,predictions):
 pids=[identifier(r) for r in predictions];rids=[identifier(r) for r in rows]
 if len(set(pids))!=len(pids) or set(pids)!=set(rids):raise ValueError('Prediction IDs must match every frozen ID exactly once')
 pred=dict(zip(pids,predictions));out=[]
 for r in rows:
  rid=identifier(r);pr=pred[rid]
  if set(pr['answers'])!=set(r['questions']):raise ValueError(f'Question mismatch: {rid}')
  m=r['metadata']
  for qid,q in r['questions'].items():
   opts,label,y=gold_values(r['gold'][qid],q);a=pr['answers'][qid];pd=a['probabilities']
   if set(pd)!=set(opts):raise ValueError(f'Probability options mismatch: {rid}/{qid}')
   p=np.array([pd[k] for k in opts],dtype=np.float64)
   if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or abs(p.sum()-1)>1e-5:raise ValueError(f'Malformed probabilities: {rid}/{qid}')
   p/=p.sum();selected=opts[int(p.argmax())]
   if labelkey(a['answer'])!=selected:raise ValueError(f'Answer/argmax mismatch: {rid}/{qid}')
   logp=np.log(np.maximum(p,1e-12));positive=y>0
   out.append(dict(id=rid,group=m.get('group_id',rid),domain=m.get('domain','unknown'),language=m.get('language','unknown'),source=m.get('source','unknown'),evaluation_slice=m.get('evaluation_slice','unknown'),question=qid,type=q['type'],correct=float(selected==label),hard_nll=float(-logp[opts.index(label)]),soft_cross_entropy=float(-np.sum(y*logp)),kl_from_target=float(np.sum(y[positive]*(np.log(y[positive])-logp[positive]))),brier_sum=float(np.sum((p-y)**2)),confidence=float(p.max()),gold=label,prediction=selected,selected_option_index=opts.index(selected),n_options=len(opts)))
 return out

def summarize(ss):
 n=len(ss);rec=collections.defaultdict(list)
 for s in ss:rec[s['id']].append(s['correct'])
 v={'decisions':n,'records':len(rec),'source_groups':len({s['group'] for s in ss})}
 for k in ['correct','hard_nll','soft_cross_entropy','kl_from_target','brier_sum']:
  v['accuracy' if k=='correct' else k]=sum(s[k] for s in ss)/n
 v['record_accuracy']=sum(all(x) for x in rec.values())/len(rec)
 v['ece_10_equal_width']=0.
 for b in range(10):
  sb=[s for s in ss if min(int(s['confidence']*10),9)==b]
  if sb:v['ece_10_equal_width']+=len(sb)/n*abs(sum(x['confidence'] for x in sb)/len(sb)-sum(x['correct'] for x in sb)/len(sb))
 v['gold_counts']=dict(collections.Counter(s['gold'] for s in ss));v['prediction_counts']=dict(collections.Counter(s['prediction'] for s in ss))
 v['option_position_counts']=dict(collections.Counter(str(s['selected_option_index']) for s in ss))
 return v

def bootstrap(scores):
 """Paired stratified source-group bootstrap, including unequal group sizes."""
 names=list(scores);reference=scores[names[0]]
 keys=[(s['id'],s['question']) for s in reference]
 for name in names:
  if [(s['id'],s['question']) for s in scores[name]]!=keys:raise ValueError('Decision ordering mismatch')
 groups=collections.defaultdict(list)
 for i,s in enumerate(reference):groups[s['group']].append(i)
 strata=collections.defaultdict(list);gids=sorted(groups)
 for g in gids:
  ds={reference[i]['domain'] for i in groups[g]}
  if len(ds)!=1:raise ValueError('A source group spans domain strata')
  strata[next(iter(ds))].append(g)
 rng=np.random.default_rng(SEED);num=np.zeros((BOOTSTRAPS,len(names)));den=np.zeros(BOOTSTRAPS)
 for domain,gg in sorted(strata.items()):
  ns=np.array([len(groups[g]) for g in gg],dtype=np.int64)
  hits=np.array([[sum(scores[name][i]['correct'] for i in groups[g]) for name in names] for g in gg])
  # Chunking limits memory; the same sampled groups are used for every model.
  for start in range(0,BOOTSTRAPS,500):
   stop=min(start+500,BOOTSTRAPS);idx=rng.integers(0,len(gg),size=(stop-start,len(gg)))
   den[start:stop]+=ns[idx].sum(1);num[start:stop]+=hits[idx].sum(1)
 acc=num/den[:,None];out={'method':'paired domain-stratified source-group percentile bootstrap','resamples':BOOTSTRAPS,'seed':SEED,'independent_groups':len(groups),'accuracy_ci95':{},'paired_lifts':{}}
 for j,name in enumerate(names):out['accuracy_ci95'][name]=np.quantile(acc[:,j],[.025,.975]).tolist()
 if 'v2' in names:
  j=names.index('v2');v2=summarize(scores['v2'])['accuracy'];registered=sum(x in names for x in BASELINES)
  for k,name in enumerate(names):
   if name=='v2':continue
   diff=acc[:,j]-acc[:,k]
   out['paired_lifts'][name]={'v2_minus_baseline':v2-summarize(scores[name])['accuracy'],'ci95':np.quantile(diff,[.025,.975]).tolist()}
   if name in BASELINES:out['paired_lifts'][name]['familywise95_lower']=float(np.quantile(diff,.05/max(registered,1)))
 return out

def report(rows,predictions,primary_gate=False):
 scored={k:score(rows,v) for k,v in predictions.items()};coverage={l:{t:sum(r['metadata'].get('language')==l and q['type']==t for r in rows for q in r['questions'].values()) for t in ['choice','noul','score']} for l in ['en','tr']};result={'models':{},'uncertainty':bootstrap(scored),'language_type_decision_coverage':coverage}
 for name,ss in scored.items():
  slices=collections.defaultdict(list)
  for s in ss:
   for key in ['language','domain','source','type','question','evaluation_slice']:
    slices[key+':'+s[key]].append(s)
   slices['language_domain:'+s['language']+':'+s['domain']].append(s)
   slices['language_type:'+s['language']+':'+s['type']].append(s)
   slices['language_task:'+s['language']+':'+s['evaluation_slice']].append(s)
  result['models'][name]={'overall':summarize(ss),'slices':{k:summarize(v) for k,v in sorted(slices.items())}}
 if not primary_gate and all(x in predictions for x in ('v2',)+BASELINES):
  language_results={};passes=[]
  for language in sorted({r['metadata'].get('language','unknown') for r in rows}):
   ls={name:[x for x in ss if x['language']==language] for name,ss in scored.items()}
   u=bootstrap(ls);support=summarize(ls['v2']);task_gaps={}
   for task in sorted({s['evaluation_slice'] for s in ls['v2']}):
    task_scores={name:summarize([s for s in ss if s['evaluation_slice']==task])['accuracy'] for name,ss in ls.items()}
    task_gaps[task]=task_scores['v2']-max(task_scores[b] for b in BASELINES)
   gates={'at_least_100_source_groups':support['source_groups']>=100,'lift_at_least_5pp_each_baseline':all(u['paired_lifts'][b]['v2_minus_baseline']>=.05-1e-12 for b in BASELINES),'paired_ci95_lower_above_zero_each_baseline':all(u['paired_lifts'][b]['ci95'][0]>0 for b in BASELINES),'no_task_loss_over_5pp':all(d>=-.05-1e-12 for d in task_gaps.values())}
   language_results[language]={'support':support,'uncertainty':u,'conditions':gates,'passed':all(gates.values()),'task_gap_vs_strongest_baseline':task_gaps};passes.append(all(gates.values()))
  result['secondary_bilingual_gate']={'by_language':language_results,'passed':all(passes) and set(language_results)=={'en','tr'},'interpretation':'Bounded intent/NLI improvement only; does not establish general Turkish decisions or independently validated translation quality.'}
  # Report task-local uncertainty rather than hiding a difficult unseen-domain slice.
  result['secondary_task_uncertainty']={task:bootstrap({name:[s for s in ss if s['evaluation_slice']==task] for name,ss in scored.items()}) for task in sorted({s['evaluation_slice'] for s in next(iter(scored.values()))})}
 if primary_gate:
  if not all(x in predictions for x in ('v2',)+BASELINES):raise ValueError('All preregistered baselines are mandatory')
  if len(rows)!=368 or sum(len(r['questions']) for r in rows)!=1840:raise ValueError('Primary sample size changed')
  model=result['models']['v2'];lifts=result['uncertainty']['paired_lifts'];domain_gaps={}
  for k,v in model['slices'].items():
   if k.startswith('domain:'):domain_gaps[k]=v['accuracy']-max(result['models'][b]['slices'][k]['accuracy'] for b in BASELINES)
  gates={'accuracy_at_least_65pct':model['overall']['accuracy']>=.65,'lift_at_least_5pp_each_baseline':all(lifts[b]['v2_minus_baseline']>=.05-1e-12 for b in BASELINES),'familywise_lower_above_zero_each_baseline':all(lifts[b]['familywise95_lower']>0 for b in BASELINES),'no_domain_loss_over_5pp':all(x>=-.05-1e-12 for x in domain_gaps.values())}
  result['primary_performance_gate']={'conditions':gates,'passed':all(gates.values()),'domain_gap_vs_strongest_baseline':domain_gaps,'integrity_gate':'Separate audit required; numerical performance pass is not release approval.'}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--predictions',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--primary-gate',action='store_true');a=p.parse_args()
 paths=dict(x.split('=',1) for x in a.predictions);rows=read(a.labels);r=report(rows,{k:read(v) for k,v in paths.items()},a.primary_gate)
 r['inputs']={'labels':{'path':a.labels,'sha256':hashlib.sha256(Path(a.labels).read_bytes()).hexdigest()},'predictions':{k:{'path':v,'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k,v in paths.items()}}
 Path(a.output).write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps({k:v['overall']['accuracy'] for k,v in r['models'].items()}));print(r.get('primary_performance_gate','No primary gate requested'))
if __name__=='__main__':main()
