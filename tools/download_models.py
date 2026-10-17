import argparse, concurrent.futures, hashlib, json, os, urllib.request, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    meta=json.loads((ROOT/'MODEL_SOURCE.json').read_text(encoding='utf8'));jobs=[]
    base=ROOT/'pretrained_models/CosyVoice-300M-25Hz';encoder=ROOT/'pretrained_models/distiluse-base-multilingual-cased-v1'
    for entry in meta['files']:
        name=entry['Path'];url=f"https://modelscope.cn/api/v1/models/{meta['cosyvoice_model']}/repo?Revision={entry['Revision']}&FilePath={name}"
        jobs.append((base,name,url,entry['Sha256']))
    for entry in meta['sentence_encoder_files']:
        name=entry['path'].replace('\\','/')
        url=f"https://huggingface.co/{meta['sentence_encoder']}/resolve/{meta['sentence_encoder_revision']}/{name}"
        jobs.append((encoder,name,url,entry['sha256']))
    def run(job):
        directory,name,url,expected=job;path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('Unsafe artifact path.')
        if path.exists() and digest(path)==expected:print('VERIFIED',name,flush=True);return
        if args.verify:raise ValueError(f'Missing or mismatched artifact: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        req=urllib.request.Request(url,headers={'User-Agent':'ChainTalker-reproduction/1.0'})
        temp=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(req,timeout=180) as response, open(temp,'wb') as target:shutil.copyfileobj(response,target,4<<20)
        if digest(temp)!=expected:raise ValueError(f'Artifact SHA256 mismatch: {name}')
        os.replace(temp,path);print('DOWNLOADED/VERIFIED',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
    print('All pinned model artifacts verified.')

if __name__=='__main__':main()
