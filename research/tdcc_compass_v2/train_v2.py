"""Resumable, bounded development-selected CPU experiments; never opens final-test."""
import argparse
from engine import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['head','lora'],required=True);ap.add_argument('--name',required=True);ap.add_argument('--hard-weight',type=float,default=0);ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--from-checkpoint');ap.add_argument('--accumulation',type=int,default=4);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 out=R/'artifacts'/a.name;out.mkdir(exist_ok=True);start=time.time()
 if (R/'STOP_REQUESTED').exists():raise SystemExit(75)
 def log(event,**kw):
  row=dict(event=event,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),wall_seconds=round(time.time()-start,2),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**kw)
  print(json.dumps(row),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if (out/'result.json').exists():raise RuntimeError('Completed experiment exists, refusing overwrite')
 e=Engine();train=load_items(R/'data/train.jsonl',e.tok);dev=load_items(R/'data/dev.jsonl',e.tok)
 source_hashes={s:sha(R/f'data/{s}.jsonl') for s in ['train','dev']}
 source_root=Path(__file__).parent
 source_hashes_code={f:sha(source_root/f) for f in ['train_v2.py','engine.py','upstream/joint_schema_model.py']}
 cfg=dict(code_sha256=source_hashes_code,stage=a.stage,hard_weight=a.hard_weight,epochs=a.epochs,accumulation=a.accumulation,data_sha256=source_hashes,name=a.name,from_checkpoint=a.from_checkpoint)
 if (out/'run_config.json').exists():
  if json.loads((out/'run_config.json').read_text())!=cfg:raise RuntimeError('Configuration changed during resume')
 else:atomic_json(out/'run_config.json',cfg)
 if a.stage=='lora' and not a.from_checkpoint:raise ValueError('LoRA stage requires selected head checkpoint')
 if a.from_checkpoint:e.load(a.from_checkpoint)
 for p in e.adapter:p.requires_grad_(a.stage=='lora')
 opt=torch.optim.AdamW(([{'params':e.adapter,'lr':2e-5},{'params':e.head.parameters(),'lr':1e-4}] if a.stage=='lora' else [{'params':e.head.parameters(),'lr':2e-4}]),weight_decay=.01)
 base_lrs=[g['lr'] for g in opt.param_groups];updates=0;examples=0;epoch_start=1;position_start=0;best_score=-1.;best_nll=float('inf');best_source=None;evals=[];bad_checks=0;epoch_loss_sum=0.;epoch_loss_count=0;losses=[]
 latest=out/'latest'
 if not latest.exists() and (out/'latest.previous').exists():(out/'latest.previous').rename(latest)
 if a.resume and (latest/'training_state.pt').exists():
  e.load(latest);s=torch.load(latest/'training_state.pt',weights_only=False);opt.load_state_dict(s['optimizer']);torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng']);updates=s['updates'];examples=s['examples'];epoch_start=s['epoch'];position_start=s['position'];best_score=s['best_score'];best_nll=s['best_nll'];best_source=s['best_source'];evals=s['evals'];bad_checks=s['bad_checks'];epoch_loss_sum=s['epoch_loss_sum'];epoch_loss_count=s['epoch_loss_count'];losses=s['completed_epoch_losses'];log('resumed',updates=updates,examples=examples,epoch=epoch_start,position=position_start)
 elif latest.exists():raise RuntimeError('Partial run exists; pass --resume')
 def state(epoch,pos):return dict(updates=updates,examples=examples,epoch=epoch,position=pos,best_score=best_score,best_nll=best_nll,best_source=best_source,evals=evals,bad_checks=bad_checks,epoch_loss_sum=epoch_loss_sum,epoch_loss_count=epoch_loss_count,completed_epoch_losses=losses,pending_scheduled_assessment=(pos>0 and (pos==len(train) or a.stage=='lora' and pos==math.ceil(len(train)/2/a.accumulation)*a.accumulation) and not any(x['checkpoint']==f'{a.stage}_epoch_{epoch}_position_{pos}' for x in evals)))
 def save_latest(epoch,pos):e.save(latest,dict(stage=a.stage,epoch=epoch,position=pos,updates_completed=updates,examples_seen=examples,run_config=cfg),opt,state(epoch,pos))
 def assess(epoch,pos):
  nonlocal best_score,best_nll,best_source,bad_checks
  zs=e.evaluate(dev,cached=a.stage=='head');m=metrics(dev,zs);ident=f'{a.stage}_epoch_{epoch}_position_{pos}'
  (out/'dev_predictions').mkdir(exist_ok=True)
  atomic_json(out/'dev_predictions'/f'{ident}.json',[dict(id=item['id'],answers={q.question_id:{'option_ids':list(q.option_ids),'logits':[float(x) for x in z]} for q,z in zip(item['record'].questions,ll)}) for item,ll in zip(dev,zs)])
  improved=(m['selection_score']>best_score+1e-9 or abs(m['selection_score']-best_score)<1e-9 and m['overall']['hard_nll']<best_nll)
  if improved:
   best_score=m['selection_score'];best_nll=m['overall']['hard_nll'];best_source=ident;bad_checks=0
   e.save(out/'best',dict(stage=a.stage,epoch=epoch,position=pos,updates_completed=updates,examples_seen=examples,run_config=cfg,dev_metrics=m))
  else:bad_checks+=1
  evals.append(dict(checkpoint=ident,metrics=m,improved=improved));atomic_json(out/'dev_evaluations.json',evals)
  log('dev_evaluation',checkpoint=ident,metrics=m,improved=improved,bad_checks=bad_checks)
  return m
 if updates==0 and position_start==0 and not evals:
  m=assess(0,0);log('loaded',records=len(train),dev_records=len(dev),base_parameters=sum(p.numel() for p in e.base.parameters()),head_parameters=sum(p.numel() for p in e.head.parameters()),adapter_parameters=sum(p.numel() for p in e.adapter),baseline_dev=m['overall'])
  save_latest(1,0)
 total_steps=math.ceil(len(train)/a.accumulation)*a.epochs;warmup=max(10,round(total_steps*.05));stopped=False
 for epoch in range(epoch_start,a.epochs+1):
  order=list(range(len(train)));random.Random(SEED+epoch).shuffle(order);startpos=position_start if epoch==epoch_start else 0
  if startpos==0:epoch_loss_sum=0.;epoch_loss_count=0
  halfway=math.ceil(len(train)/2/a.accumulation)*a.accumulation
  # A stop/checkpoint may have occurred immediately before a scheduled check.
  # Complete that pending assessment on resume, never while a stop remains requested.
  if a.stage=='lora' and startpos==halfway and not any(x['checkpoint']==f'{a.stage}_epoch_{epoch}_position_{startpos}' for x in evals):
   assess(epoch,startpos);save_latest(epoch,startpos)
  if a.stage=='lora' and bad_checks>=2:stopped=True;break
  e.head.train();e.text.train(a.stage=='lora');opt.zero_grad(set_to_none=True)
  for pos in range(startpos,len(order)):
   item=train[order[pos]];block_start=(pos//a.accumulation)*a.accumulation;block_size=min(a.accumulation,len(order)-block_start)
   with torch.autocast('cpu',dtype=torch.bfloat16):zs=e.logits(item,cached=a.stage=='head');loss=e.loss(zs,item,a.hard_weight)
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
   (loss/block_size).backward();epoch_loss_sum+=float(loss.detach());epoch_loss_count+=1;examples+=1
   if (pos+1)%a.accumulation==0 or pos+1==len(order):
    frac=min(1.,(updates+1)/warmup) if updates<warmup else .1+.9*.5*(1+math.cos(math.pi*min(1.,(updates-warmup)/max(1,total_steps-warmup))))
    for g,lr in zip(opt.param_groups,base_lrs):g['lr']=lr*frac
    grad=float(torch.nn.utils.clip_grad_norm_(list(e.head.parameters())+(e.adapter if a.stage=='lora' else []),1.))
    if not math.isfinite(grad):raise ValueError('Nonfinite gradient')
    opt.step();opt.zero_grad(set_to_none=True);updates+=1
   if (pos+1)%40==0 or pos+1==len(order):log('train_progress',stage=a.stage,epoch=epoch,position=pos+1,total=len(order),updates=updates,examples=examples,mean_loss=epoch_loss_sum/epoch_loss_count)
   if (pos+1)%100==0:save_latest(epoch,pos+1)
   if (R/'STOP_REQUESTED').exists() and ((pos+1)%a.accumulation==0 or pos+1==len(order)):
    save_latest(epoch,pos+1);log('paused_on_request',epoch=epoch,position=pos+1);raise SystemExit(75)
   if a.stage=='lora' and pos+1==halfway:
    assess(epoch,pos+1);save_latest(epoch,pos+1);e.head.train();e.text.train()
    if bad_checks>=2:stopped=True;break
  if stopped:break
  losses.append(dict(epoch=epoch,mean_loss=epoch_loss_sum/epoch_loss_count,examples=epoch_loss_count))
  if not any(x['checkpoint']==f'{a.stage}_epoch_{epoch}_position_{len(order)}' for x in evals):assess(epoch,len(order))
  epoch_loss_sum=0.;epoch_loss_count=0
  save_latest(epoch+1,0);log('epoch_complete',epoch=epoch,updates=updates,examples=examples,mean_loss=losses[-1]['mean_loss'])
  if a.stage=='lora' and bad_checks>=2:stopped=True;log('early_stop',reason='Two consecutive half/whole-epoch development checks failed to improve');break
 e.load(out/'best');atomic_json(out/'result.json',dict(status='stage_complete',name=a.name,stage=a.stage,updates=updates,examples_seen=examples,unique_train_records=len(train),best_source=best_source,best_selection_score=best_score,best_hard_nll=best_nll,epoch_losses=losses,partial_epoch_loss=({'epoch':epoch,'examples':epoch_loss_count,'mean_loss':epoch_loss_sum/epoch_loss_count} if epoch_loss_count else None),dev_evaluations=evals,early_stopped=stopped,wall_seconds_this_process=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
 log('complete',best_source=best_source,best_selection_score=best_score,updates=updates)
if __name__=='__main__':main()
