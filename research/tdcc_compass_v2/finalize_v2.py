"""Freeze one development-selected model; fit calibration only, never final-test."""
from engine import *
import argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidates',nargs='+',required=True);a=ap.parse_args()
 if (R/'artifacts/selected/compass_cpu_config.json').exists():raise RuntimeError('Selected checkpoint exists; refuse to refit after freeze')
 candidates=[]
 for name in a.candidates:
  p=R/'artifacts'/name
  r=json.loads((p/'result.json').read_text());assert r['status']=='stage_complete'
  candidates.append(dict(name=name,checkpoint=str(p/'best'),selection_score=r['best_selection_score'],hard_nll=r['best_hard_nll'],result_sha256=sha(p/'result.json')))
 winner=max(candidates,key=lambda c:(c['selection_score'],-c['hard_nll']))
 atomic_json(R/'artifacts/selection.json',dict(selected=winner,candidates=candidates,selection_completed_before_calibration=True,decision_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),data_manifest_sha256=sha(R/'data/manifest.json')))
 e=Engine();e.load(winner['checkpoint']);cal=load_items(R/'data/calibration.jsonl',e.tok);zs=e.evaluate(cal)
 temps=[math.exp(math.log(.25)+(math.log(4)-math.log(.25))*i/160) for i in range(161)]
 temp=min(temps,key=lambda t:metrics(cal,zs,t)['overall']['hard_nll'])
 before=metrics(cal,zs);after=metrics(cal,zs,temp)
 out=R/'artifacts/selected';e.save(out,dict(stage='selected',selected_checkpoint=winner['checkpoint'],dev_selection_score=winner['selection_score'],data_manifest_sha256=sha(R/'data/manifest.json'),limitations=['Experimental public benchmark adaptation, not validated general reliability.','Typed Decisions targets are model-teacher labels.','Turkish SNLI inputs are machine-translated and not independently reviewed.','No consequential autonomous decision use is established.']))
 cfg=json.loads((out/'compass_cpu_config.json').read_text());cfg['temperature']=temp;cfg['temperature_objective']='calibration hard-label NLL';cfg['calibration_data_sha256']=sha(R/'data/calibration.jsonl');atomic_json(out/'compass_cpu_config.json',cfg)
 rows=[]
 for item,ll in zip(cal,zs):
  rows.append(dict(id=item['id'],answers={q.question_id:dict(option_ids=list(q.option_ids),logits=[float(x) for x in z],probabilities={k:float(v) for k,v in zip(q.option_ids,(z/temp).softmax(-1))}) for q,z in zip(item['record'].questions,ll)}))
 atomic_json(R/'artifacts/calibration_predictions.json',rows)
 atomic_json(R/'artifacts/calibration_result.json',dict(temperature=temp,grid=[.25,4,161],objective='hard_nll',before=before,after=after))
 tracked=[out/'joint_head.safetensors',out/'adapter/adapter_model.safetensors',out/'adapter/adapter_config.json',out/'joint_head_config.json',out/'compass_cpu_config.json',R/'engine.py',R/'train_v2.py',R/'predict_cpu.py',R/'finalize_v2.py',R/'data/manifest.json',R/'data/train.jsonl',R/'data/dev.jsonl',R/'data/calibration.jsonl',R/'EXPERIMENT_PLAN.md']
 atomic_json(R/'artifacts/FROZEN_MANIFEST.json',dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selected=winner,temperature=temp,base_model='Qwen/Qwen3.5-0.8B',base_revision=MODEL_REV,source_commit='0b1291bfd7c147e2edf82c085b8412c2630be4fe',files={str(p.relative_to(R)):sha(p) for p in tracked},training_max_tokens=1536,inference_max_tokens=2048,test_status='not opened by training worker'))
 print(json.dumps(dict(selected=winner,temperature=temp,calibration=after['overall']),indent=2),flush=True)
if __name__=='__main__':main()
