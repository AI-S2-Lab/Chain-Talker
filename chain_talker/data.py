import json
from .runtime import read_jsonl, project_path, validate_prepared_data

class FeatureStore:
    def __init__(self,data):
        self.data=project_path(data)
        self.prepare_report=validate_prepared_data(self.data)
        self.records={r['id']:r for r in read_jsonl(self.data/'utterances.jsonl')}
        self.index=json.loads((self.data/'features_index.json').read_text(encoding='utf8'))
        from .runtime import sha256
        if self.index['manifest_sha256']!=sha256(self.data/'utterances.jsonl'):raise ValueError('Features/manifest mismatch; rerun extraction.')
        self.cache=self.data/self.index['cache_relative'].replace('\\','/')
        self.available={path.stem for path in self.cache.glob('*.npz')}

    def get(self,uid):
        import numpy as np
        with np.load(self.cache/(uid+'.npz'),allow_pickle=False) as loaded:
            if str(loaded['fingerprint'])!=self.index['fingerprint']:raise ValueError('Stale feature cache.')
            return {k:loaded[k].copy() for k in ['text_token','caption_token','speech_token','embedding','caption_embedding','mel']}

class Dataset:
    def __init__(self,data,manifest,component='llm',max_tokens=4096):
        self.store=FeatureStore(data);self.samples=read_jsonl(manifest)
        self.component=component;self.max_tokens=max_tokens
        missing=set()
        for w in self.samples:
            if len(w['history'])!=2:raise ValueError('Every sample must have exactly two history turns.')
            ids=([*w['history'],w['target']] if component=='llm' else [w['target']])
            missing.update(uid for uid in ids if uid not in self.store.available)
        if missing:raise FileNotFoundError(f'Missing {len(missing)} feature files (first: {sorted(missing)[:3]}). Run features for the complete dataset.')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        sample=self.samples[index];history=sample['history']
        target=self.store.get(sample['target'])
        past=[self.store.get(uid) for uid in history] if self.component=='llm' else []
        length=1+sum(1+len(t['speech_token'])+len(t['text_token'])+2+len(t['caption_token']) for t in past)+1+len(target['text_token'])+1+len(target['caption_token'])+1+len(target['speech_token'])
        if self.component=='llm' and length>self.max_tokens:
            raise ValueError(f"{sample['target']} sequence {length} exceeds max_tokens={self.max_tokens}; increase explicitly.")
        return {'history':past,'target':target,'id':sample['target']}

def collate(samples,component):
    if component=='llm':return {'samples':samples}
    import torch
    from torch.nn.utils.rnn import pad_sequence
    targets=[s['target'] for s in samples]
    tokens=[torch.as_tensor(t['speech_token'],dtype=torch.long) for t in targets]
    mel=[torch.as_tensor(t['mel'],dtype=torch.float32) for t in targets]
    return {'speech_token':pad_sequence(tokens,batch_first=True,padding_value=0),
            'speech_token_len':torch.tensor([len(x) for x in tokens],dtype=torch.int32),
            'speech_feat':pad_sequence(mel,batch_first=True),
            'speech_feat_len':torch.tensor([len(x) for x in mel],dtype=torch.int32),
            'embedding':torch.tensor(__import__('numpy').stack([t['embedding'] for t in targets])),
            'caption_embedding':torch.tensor(__import__('numpy').stack([t['caption_embedding'] for t in targets]))}
