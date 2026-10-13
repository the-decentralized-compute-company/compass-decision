"""Fast control-flow integration tests using a tiny substitute engine, not model claims."""
import unittest,tempfile,contextlib,io,sys,json,random
from pathlib import Path
from types import SimpleNamespace
import torch
import train_v2 as t
ROWS=[dict(id=f'r{i}',record=SimpleNamespace(questions=[SimpleNamespace(question_id='q',option_ids=('a','b'))]),targets=[torch.tensor([1.,0.])],labels=[0]) for i in range(8)]
class ToyEngine:
 stop_after=0
 def __init__(self):
  torch.manual_seed(t.SEED);self.tok=None;self.head=torch.nn.Linear(1,1,bias=False);self.base=self.head;self.text=torch.nn.Identity();self.adapter=[];self.calls=0
 def load(self,p):
  p=Path(p)/'toy.pt'
  if p.exists():self.head.load_state_dict(torch.load(p,weights_only=True))
 def logits(self,item,cached=False):
  w=self.head.weight.flatten()[0];return [torch.stack([w,-w])]
 def loss(self,zs,item,hard_weight):
  self.calls+=1
  if self.stop_after and self.calls==self.stop_after:(t.R/'STOP_REQUESTED').touch()
  return -zs[0].log_softmax(-1)[0]
 def evaluate(self,items,cached=False):
  return [[z.detach().clone() for z in self.logits(x)] for x in items]
 def save(self,p,extra,opt=None,state=None):
  p=Path(p);p.mkdir(parents=True,exist_ok=True);torch.save(self.head.state_dict(),p/'toy.pt')
  if opt is not None:torch.save(dict(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),**state),p/'training_state.pt')
def toy_metrics(items,zs):
 nll=float(-zs[0][0].log_softmax(-1)[0]);return {'overall':{'hard_nll':nll},'selection_score':-nll}
class ResumeChecks(unittest.TestCase):
 def execute(self,root,stage,stop_after=0):
  t.R=root;(root/'data').mkdir(exist_ok=True);(root/'artifacts').mkdir(exist_ok=True)
  for x in ['train','dev']:(root/f'data/{x}.jsonl').write_text('[]')
  ToyEngine.stop_after=stop_after
  sys.argv=['train_v2.py','--stage',stage,'--name','run','--hard-weight','.35','--epochs','2','--accumulation','4','--resume']
  if stage=='lora':sys.argv+=['--from-checkpoint',str(root/'initial')]
  with contextlib.redirect_stdout(io.StringIO()):t.main()
  return json.loads((root/'artifacts/run/result.json').read_text())
 def compare(self,stage,pause):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   (root/'ref').mkdir();(root/'resumed').mkdir();ref=self.execute(root/'ref',stage)
   with self.assertRaises(SystemExit) as err:self.execute(root/'resumed',stage,pause)
   self.assertEqual(err.exception.code,75)
   state=torch.load(root/'resumed/artifacts/run/latest/training_state.pt',weights_only=False)
   self.assertEqual(state['position'],pause)
   self.assertEqual(state['epoch_loss_count'],pause)
   self.assertEqual(state['pending_scheduled_assessment'],pause==8 or stage=='lora' and pause==4)
   (root/'resumed/STOP_REQUESTED').unlink();resumed=self.execute(root/'resumed',stage)
   for key in ['updates','examples_seen','epoch_losses','best_source','dev_evaluations','early_stopped']:
    self.assertEqual(ref[key],resumed[key],key)
   aa=torch.load(root/'ref/artifacts/run/best/toy.pt',weights_only=True);bb=torch.load(root/'resumed/artifacts/run/best/toy.pt',weights_only=True)
   for k in aa:self.assertTrue(torch.equal(aa[k],bb[k]))
 def test_lora_halfway(self):self.compare('lora',4)
 def test_lora_epoch_end(self):self.compare('lora',8)
 def test_head_epoch_end(self):self.compare('head',8)
if __name__=='__main__':
 t.Engine=ToyEngine;t.load_items=lambda p,tok:ROWS;t.metrics=toy_metrics
 unittest.main()
