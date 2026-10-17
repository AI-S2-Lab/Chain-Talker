import json
from pathlib import Path
from .runtime import default_encoder_dir

class CaptionEncoder:
    def __init__(self, path=None, device='cpu'):
        import torch
        from transformers import AutoTokenizer,AutoModel
        from safetensors.torch import load_file
        self.path=Path(path or default_encoder_dir());self.device=device
        modules=json.loads((self.path/'modules.json').read_text())
        if [x['type'].split('.')[-1] for x in modules]!=['Transformer','Pooling','Dense']:
            raise ValueError('Unexpected sentence encoder modules.')
        pooling=json.loads((self.path/'1_Pooling/config.json').read_text())
        if not pooling.get('pooling_mode_mean_tokens') or pooling.get('pooling_mode_cls_token') or pooling.get('pooling_mode_max_tokens'):
            raise ValueError('Expected mean-token pooling only.')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        self.model=AutoModel.from_pretrained(self.path,local_files_only=True).eval().to(device)
        cfg=json.loads((self.path/'2_Dense/config.json').read_text())
        self.dense=torch.nn.Linear(cfg['in_features'],cfg['out_features'],bias=cfg['bias'])
        if 'Tanh' not in cfg['activation_function']: raise ValueError('Unexpected activation.')
        safe=self.path/'2_Dense/model.safetensors'
        weights=load_file(str(safe)) if safe.exists() else torch.load(self.path/'2_Dense/pytorch_model.bin',map_location='cpu',weights_only=True)
        self.dense.load_state_dict({k.removeprefix('linear.'):v for k,v in weights.items()},strict=True)
        self.dense.eval().to(device)
        self.max_length=json.loads((self.path/'sentence_bert_config.json').read_text()).get('max_seq_length',128)
        self.dim=cfg['out_features']
        if self.dim!=512: raise ValueError('Expected DistilUSE 512-dimensional output.')
        for module in [self.model,self.dense]:
            for param in module.parameters():param.requires_grad_(False)

    def encode(self,texts,batch_size=32):
        import torch
        outputs=[]
        with torch.inference_mode():
            for i in range(0,len(texts),batch_size):
                batch=self.tokenizer(texts[i:i+batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
                hidden=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                mean=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
                outputs.append(torch.tanh(self.dense(mean)).float().cpu())
        return torch.cat(outputs).numpy() if outputs else __import__('numpy').zeros((0,self.dim),dtype='float32')
