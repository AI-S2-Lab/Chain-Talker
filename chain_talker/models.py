import re
from .runtime import ROOT, project_path, relative_path, default_model_dir, sha256, tokenizer, controls


def _compatible_config_hash(checkpoint_hash, current_hash):
    equivalent = {
        '33732b5b9cb3dd3c5ee8a92cac9684facc80aac8a1e18466c385d6b08726b26b',
        '598563ea1251042533265f03e81710c86540a204e47d70edcb69589713ce0d1f',
    }
    return (checkpoint_hash is None or checkpoint_hash == current_hash
            or checkpoint_hash in equivalent and current_hash in equivalent)


def load_component(component, model_dir=None, checkpoint=None, device='cpu'):
    import torch
    from hyperpyyaml import load_hyperpyyaml
    if component not in {'llm','flow','hift'}: raise ValueError(component)
    model_dir=project_path(model_dir or default_model_dir())
    cfg_path=ROOT/'configs/chain_talker.yaml'
    raw=cfg_path.read_text(encoding='utf8')
    header=raw.split('\nllm:',1)[0]+'\n'
    match=re.search(r'(?ms)^'+component+r':.*?(?=^[a-zA-Z_][\w]*:|\Z)',raw)
    if not match: raise ValueError(f'Missing {component} config.')
    selected=match.group(0)
    model=load_hyperpyyaml(header+selected)[component]
    path=project_path(checkpoint) if checkpoint else model_dir/(component+'.pt')
    state=torch.load(path,map_location='cpu',weights_only=True)
    meta={}
    if 'state_dict' in state:
        meta=state.get('metadata',{});state=state['state_dict']
        if meta.get('component') not in (None,component): raise ValueError('Checkpoint component mismatch.')
        if not _compatible_config_hash(meta.get('config_sha256'),sha256(cfg_path)): raise ValueError('Checkpoint/config mismatch.')
    state={k:v for k,v in state.items() if k not in {'epoch','step'}}
    if component=='hift': state={k.removeprefix('generator.'):v for k,v in state.items()}
    expected=model.state_dict()
    mismatched={k:[list(v.shape),list(expected[k].shape)] for k,v in state.items() if k in expected and v.shape!=expected[k].shape}
    if mismatched: raise RuntimeError(f'Checkpoint shape mismatch: {mismatched}')
    report=model.load_state_dict(state,strict=False)
    prefixes={'llm':('caption_input.','caption_decoder.'),'flow':('caption_condition.',),'hift':()}[component]
    allowed=[] if checkpoint else [k for k in expected if k.startswith(prefixes)]
    if set(report.missing_keys)!=set(allowed) or report.unexpected_keys:
        raise RuntimeError(f'Checkpoint key mismatch: {report}')
    if component=='llm': model.set_control_tokens(*controls(tokenizer()))
    model.load_report={'checkpoint':relative_path(path),'missing_initialized':list(report.missing_keys),
                       'unexpected':list(report.unexpected_keys),'component':component,'config_sha256':sha256(cfg_path)}
    return model.to(device),meta
