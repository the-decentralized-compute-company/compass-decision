"""One backward dry-run on predeclared training examples; no parameters saved."""
from engine import *
if (R/'STOP_REQUESTED').exists():raise SystemExit(75)
e=Engine();items=load_items(R/'data/train.jsonl',e.tok)
longest=max(items,key=lambda x:len(x['record'].input_ids));assert longest['id']=='massive:13488:tr'
records=[longest,max([x for x in items if x['row']['metadata']['dataset_group']=='typed_decisions'],key=lambda x:len(x['record'].input_ids)),max([x for x in items if x['row']['metadata']['dataset_group']=='snli'],key=lambda x:len(x['record'].input_ids))]
results=[]
for p in e.adapter:p.requires_grad_(True)
e.text.train();e.head.train()
for item in records:
 t=time.perf_counter()
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=e.loss(e.logits(item),item,.35)
 loss.backward();seconds=time.perf_counter()-t
 result=dict(stage='lora_backward',id=item['id'],tokens=len(item['record'].input_ids),seconds=seconds,loss=float(loss.detach()),rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gradient_checkpointing=e.text.is_gradient_checkpointing,finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in list(e.head.parameters())+e.adapter));result['nonzero_adapter_gradient_parameters']=sum(p.grad is not None and bool((p.grad!=0).any()) for p in e.adapter);assert result['nonzero_adapter_gradient_parameters']>0;results.append(result);print(json.dumps(result),flush=True)
 e.head.zero_grad(set_to_none=True);e.text.zero_grad(set_to_none=True)
for p in e.adapter:p.requires_grad_(False)
e.text.eval();e.head.train()
for item in records:
 e.feature(item,e.batch(item));t=time.perf_counter()
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=e.loss(e.logits(item,cached=True),item,.35)
 loss.backward();seconds=time.perf_counter()-t
 result=dict(stage='cached_head_backward',id=item['id'],tokens=len(item['record'].input_ids),seconds=seconds,loss=float(loss.detach()),rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);results.append(result);print(json.dumps(result),flush=True);e.head.zero_grad(set_to_none=True)
atomic_json(R/'artifacts/benchmark.json',dict(status='dry_run_no_saved_parameter_updates',results=results))
