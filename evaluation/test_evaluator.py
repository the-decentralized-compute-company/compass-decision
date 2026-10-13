import copy,math,unittest
from evaluate_frozen import *
class EvaluatorTests(unittest.TestCase):
 def rows(self):
  return [dict(id='a',state={'x':'test'},questions={'bool':{'type':'noul','instructions':'True?'},'choice':{'type':'choice','criteria':{'z':'Z','a':'A'}}},gold={'bool':{'label':False,'probabilities':{'false':.7,'true':.300001}},'choice':'z'},metadata={'group_id':'a','domain':'d','language':'en'})]
 def preds(self):return [{'id':'a','answers':{'bool':{'answer':False,'probabilities':{'false':.8,'true':.2}},'choice':{'answer':'a','probabilities':{'a':.6,'z':.4}}}}]
 def test_types(self):
  self.assertEqual(options({'type':'noul'}),['true','false']);self.assertEqual(options({'type':'choice','criteria':{'z':0,'a':0}}),['a','z']);self.assertEqual(options({'type':'score','criteria':[0]*12})[-2:],['10','11'])
 def test_metrics(self):
  r=summarize(score(self.rows(),self.preds()));self.assertEqual(r['accuracy'],.5);self.assertEqual(r['record_accuracy'],0);self.assertAlmostEqual(r['hard_nll'],(-math.log(.8)-math.log(.4))/2);self.assertAlmostEqual(r['ece_10_equal_width'],.4)
 def test_rounding_normalized(self):
  opts,y,p=gold_values(self.rows()[0]['gold']['bool'],self.rows()[0]['questions']['bool']);self.assertAlmostEqual(sum(p),1.)
 def test_bad_probs(self):
  p=self.preds();p[0]['answers']['bool']['probabilities']['true']=.3
  with self.assertRaises(ValueError):score(self.rows(),p)
 def test_missing_id(self):
  with self.assertRaises(ValueError):score(self.rows(),[])
 def test_mismatched_answer(self):
  p=self.preds();p[0]['answers']['bool']['answer']=True
  with self.assertRaises(ValueError):score(self.rows(),p)
 def test_prior_uniform_encoder_tie(self):
  p,coverage=priors([],self.rows(),'soft');self.assertEqual(p[0]['answers']['bool']['answer'],'true');self.assertEqual(p[0]['answers']['choice']['answer'],'a');self.assertEqual(coverage['coverage'],0)
 def test_hard_soft_priors(self):
  p,c=priors(self.rows(),self.rows(),'hard');self.assertAlmostEqual(p[0]['answers']['choice']['probabilities']['z'],2/3)
  p,c=priors(self.rows(),self.rows(),'soft');self.assertEqual(p[0]['answers']['choice']['probabilities']['z'],1)
 def test_paired_bootstrap_identical(self):
  s=score(self.rows(),self.preds());r=bootstrap({'v2':s,'v1':s});self.assertEqual(r['paired_lifts']['v1']['ci95'],[0.,0.]);self.assertEqual(r['accuracy_ci95']['v2'],[.5,.5])
 def test_translation_groups(self):
  r=self.rows();tr=copy.deepcopy(r[0]);tr['id']='a-tr';tr['metadata']['language']='tr';r.append(tr);p=self.preds();trp=copy.deepcopy(p[0]);trp['id']='a-tr';p.append(trp);s=score(r,p);result=bootstrap({'v2':s,'v1':s});self.assertEqual(result['independent_groups'],1)
if __name__=='__main__':unittest.main()
