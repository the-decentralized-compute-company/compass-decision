import time,json,sys
from engine import *
e=Engine();t=time.time();n=0
for path in sys.argv[1:]:
 items=load_items(path,e.tok)
 for i,item in enumerate(items,1):
  if (R/'STOP_REQUESTED').exists():raise SystemExit(75)
  e.feature(item,e.batch(item));n+=1
  if i%20==0 or i==len(items):print(json.dumps(dict(file=path,position=i,total=len(items),overall=n,seconds=time.time()-t,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)
