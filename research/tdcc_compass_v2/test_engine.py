import unittest
from engine import *
class EngineChecks(unittest.TestCase):
 def test_soft_hard_mix(self):
  e=Engine.__new__(Engine);z=torch.tensor([.2,-.3,1.7],requires_grad=True);y=torch.tensor([.2,.1,.7]);item={'targets':[y],'labels':[2]}
  for a in [0,.35,1]:
   actual=e.loss([z],item,a);wanted=(1-a)*(-(y*z.log_softmax(-1)).sum())+a*(-z.log_softmax(-1)[2]);self.assertTrue(torch.allclose(actual,wanted))
  e.loss([z],item,.35).backward();self.assertTrue(torch.isfinite(z.grad).all())
 def test_splits(self):
  parts={p:list(map(json.loads,(R/f'data/original_{p}.jsonl').read_text().splitlines())) for p in ['train','dev','calibration']}
  for p,x in parts.items():
   self.assertEqual(len(x),959 if p=='train' else 120)
   for q,y in parts.items():
    if p==q:continue
    self.assertFalse({r['id'] for r in x}&{r['id'] for r in y})
    self.assertFalse({json.dumps(r['state'],sort_keys=True) for r in x}&{json.dumps(r['state'],sort_keys=True) for r in y})
 def test_expected_types(self):
  self.assertEqual(key(True),'true');self.assertEqual(key(False),'false');self.assertEqual(key(0),'0')
if __name__=='__main__':unittest.main()
