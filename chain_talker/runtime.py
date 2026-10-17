import hashlib, json, os, random, sys, time
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
for p in (ROOT/'third_party/Matcha-TTS',):
    sys.path.insert(0, str(p))
for key, suffix in {'HF_HOME':'huggingface','TRANSFORMERS_CACHE':'huggingface/transformers',
                    'NUMBA_CACHE_DIR':'numba','TORCH_HOME':'torch'}.items():
    os.environ.setdefault(key, str(ROOT/'.cache'/suffix))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def project_path(value):
    if value is None:return None
    if os.name!='nt' and PureWindowsPath(str(value)).drive:
        raise ValueError('Windows absolute path cannot be used here; migrate the dataset/job to relative paths.')
    path=Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()

def relative_path(value, base=None):
    return Path(os.path.relpath(Path(value).resolve(), Path(base or ROOT).resolve())).as_posix()

def data_path(value, data):
    if os.name!='nt' and PureWindowsPath(str(value)).drive:
        raise ValueError('Dataset paths must be relative. Prepare the WAV/LAB directories again on this system.')
    path=Path(str(value).replace('\\','/'))
    return (path if path.is_absolute() else Path(data)/path).resolve()

def default_log_dir():
    override=os.environ.get('CHAIN_TALKER_TENSORBOARD_DIR')
    return project_path(override) if override else (Path('/root/tf-logs') if sys.platform.startswith('linux') else ROOT/'tf-logs')

def validate_prepared_data(data, verify_sources=False):
    data=Path(data).resolve()
    report=json.loads((data/'prepare_report.json').read_text(encoding='utf-8'))
    if report.get('schema')!=2 or report.get('source_layout')!='wav_lab_caption':
        raise ValueError('Expected WAV/LAB data. Run prepare --input <WAV/LAB directory> with a new --output.')
    hashes={'utterances.jsonl':report['manifest_sha256'],
            'dialogue_splits.json':report['dialogue_splits_sha256'],
            'source_manifest.json':report['source_manifest_sha256']}
    hashes.update({split+'.jsonl':report['windows_sha256'][split] for split in ('train','valid','test')})
    for name,expected in hashes.items():
        if sha256(data/name)!=expected:raise ValueError('Prepared data changed: '+name)
    if verify_sources:
        index=json.loads((data/'source_manifest.json').read_text(encoding='utf-8'))
        expected_paths=set()
        for item in index['files']:
            path=data_path(item['path'],data).resolve()
            expected_paths.add(path)
            if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
                raise ValueError(f'Source data changed: {path}. Prepare into a new output directory.')
        source=data_path(index['source_directory'],data)
        actual_paths={path.resolve() for split in ('train','valid','test') for suffix in ('*.wav','*.lab')
                      for path in (source/split).rglob(suffix)}
        if expected_paths!=actual_paths:raise ValueError('Source WAV/LAB inventory changed. Prepare into a new output directory.')
    return report

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def read_jsonl(path):
    with open(path,encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

def atomic_replace(temp,path):
    for attempt in range(8):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.1*(2**attempt),1.0))

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    atomic_replace(temp,path)

def write_jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'w',encoding='utf8',newline='\n') as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    atomic_replace(temp,path)

def seed_all(seed):
    import numpy as np, torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def tokenizer():
    from cosyvoice.tokenizer.tokenizer import get_tokenizer
    return get_tokenizer(multilingual=True,num_languages=100,language='en',task='transcribe')

def controls(tok):
    a=tok.encode('<|SPECIAL_TOKEN_1|>',allowed_special='all')
    b=tok.encode('<|SPECIAL_TOKEN_2|>',allowed_special='all')
    if len(a)!=1 or len(b)!=1: raise ValueError('Reserved control tokens not found.')
    return a[0],b[0]

def audio(path, sample_rate=16000):
    import numpy as np, soundfile as sf, torch, torchaudio
    data,sr=sf.read(path,dtype='float32',always_2d=True)
    if not len(data) or not np.isfinite(data).all(): raise ValueError(f'Invalid audio: {path}')
    wav=torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr!=sample_rate: wav=torchaudio.functional.resample(wav,sr,sample_rate)
    return wav

def default_model_dir(): return ROOT/'pretrained_models/CosyVoice-300M-25Hz'
def default_encoder_dir(): return ROOT/'pretrained_models/distiluse-base-multilingual-cased-v1'
