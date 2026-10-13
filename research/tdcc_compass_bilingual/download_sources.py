#!/usr/bin/env python3
"""Download official public corpus archives and verify frozen bytes. No API token needed."""
import hashlib, json, tarfile, urllib.request, zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
ASSETS=[
 ('massive-1.1.tar.gz','https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz','4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577'),
 ('snli_1.0.zip','https://nlp.stanford.edu/projects/snli/snli_1.0.zip','afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e'),
 ('snli_tr_1.1.zip','https://nli-tr.s3.eu-central-1.amazonaws.com/snli_tr_1.1.zip','155aac79fe5085065a59fd3ceea441bbbce729c85f898918a6ffba0d28fecb5c'),
]
def main():
 (R/'raw').mkdir(exist_ok=True);(R/'licenses').mkdir(exist_ok=True)
 for name,url,expected in ASSETS:
  p=R/'raw'/name
  if not p.exists():
   temporary=p.with_suffix(p.suffix+'.partial')
   with urllib.request.urlopen(url,timeout=90) as remote,temporary.open('wb') as output:
    while chunk:=remote.read(1024*1024):output.write(chunk)
   temporary.rename(p)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(f'Checksum mismatch: {name}; do not silently accept changed source bytes')
 with tarfile.open(R/'raw/massive-1.1.tar.gz') as archive:
  for member,target in [('1.1/data/en-US.jsonl','raw/en-US.jsonl'),('1.1/data/tr-TR.jsonl','raw/tr-TR.jsonl'),('1.1/LICENSE','licenses/MASSIVE_LICENSE'),('1.1/NOTICE.md','licenses/MASSIVE_NOTICE.md'),('1.1/CITATION.md','licenses/MASSIVE_CITATION.md')]:
   (R/target).write_bytes(archive.extractfile(member).read())
 for name,member,target in [('snli_1.0.zip','snli_1.0/README.txt','SNLI_README.txt'),('snli_tr_1.1.zip','README.txt','SNLI_TR_README.txt')]:
  with zipfile.ZipFile(R/'raw'/name) as archive:(R/'licenses'/target).write_bytes(archive.read(member))
 print('Official corpus archives verified; exact designated members extracted.')
if __name__=='__main__':main()
