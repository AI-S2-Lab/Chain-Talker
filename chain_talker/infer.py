"""Leakage-safe offline Chain-Talker inference and test-job export."""
import argparse, contextlib, gc, json, shutil
from pathlib import Path
from .runtime import ROOT, project_path, relative_path, data_path, default_encoder_dir, default_model_dir, read_jsonl, seed_all, sha256, tokenizer, write_json

def artifact_paths(output):
    return {'audio':output,'caption':output.with_suffix('.caption.txt'),
            'report':output.with_suffix('.json')}

def input_snapshot(args, job, history, reference_text, text, tok):
    base=args.output.parent
    target={'id':job.get('target_id'),'text':text,'speaker':job.get('target_speaker'),
            'turn_id':job.get('target_turn_id'),'dialogue_id':job.get('dialogue_id')}
    result={'target':{k:v for k,v in target.items() if v is not None},
            'history':[],'reference':{'text':reference_text}}
    for i,feature in enumerate(history):
        source=job.get('history',[{}]*len(history))[i]
        turn={'id':source.get('id'),'speaker':source.get('speaker'),'turn_id':source.get('turn_id'),
              'text':tok.decode(feature['text_token'].tolist()),
              'caption':tok.decode(feature['caption_token'].tolist())}
        for key in ('feature','audio'):
            if source.get(key):turn[key]=relative_path(data_path(source[key],args.job.parent),base)
        result['history'].append({k:v for k,v in turn.items() if v is not None})
    result['reference'].update(id=Path(job['reference_feature']).stem,
        feature=relative_path(data_path(job['reference_feature'],args.job.parent),base))
    return result

def groundtruth_source(args, job):
    target_id=job.get('target_id')
    if not target_id:return None
    explicit=args.data is not None or bool(job.get('data_directory'))
    dataset=args.data or (data_path(job['data_directory'],args.job.parent) if job.get('data_directory') else ROOT/'data/dailytalk')
    manifest=dataset/'utterances.jsonl'
    if not manifest.is_file():
        if explicit:raise FileNotFoundError(f'Ground-truth dataset manifest not found: {manifest}')
        return None
    record=None
    with manifest.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():continue
            candidate=json.loads(line)
            if candidate['id']==target_id:record=candidate;break
    if record is None:
        if explicit:raise ValueError(f'Ground-truth target {target_id} is absent from {manifest}.')
        return None
    index_path=dataset/'features_index.json'
    if job.get('feature_fingerprint') and index_path.is_file():
        index=json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('fingerprint')!=job['feature_fingerprint']:
            raise ValueError('Ground-truth dataset and inference job use different feature versions.')
        if index.get('manifest_sha256')!=sha256(manifest):
            raise ValueError('Ground-truth dataset manifest does not match its feature index.')
    source=data_path(record['audio'],dataset)
    if not source.is_file():raise FileNotFoundError(f'Ground-truth target audio not found: {source}')
    return record,manifest,source

def export_groundtruth(source, paths):
    if source is None:
        return None
    record,_,audio_path=source
    paths['groundtruth_caption'].write_text(record['caption'],encoding='utf-8',newline='\n')
    shutil.copy2(audio_path,paths['groundtruth_audio'])
    digest=sha256(audio_path)
    if sha256(paths['groundtruth_audio'])!=digest:raise RuntimeError('Ground-truth audio copy checksum mismatch.')
    info={'id':record['id'],'text':record['text'],'caption':record['caption'],
          'speaker':record.get('speaker'),'turn_id':record.get('turn_id'),'dialogue_id':record.get('dialogue_id'),
          'audio':paths['groundtruth_audio'].name}
    return {k:v for k,v in info.items() if v is not None}

def make_job(argv=None):
    p=argparse.ArgumentParser(description='Export a job containing visible context, reference and target text only.')
    p.add_argument('--data',type=project_path,default=ROOT/'data/dailytalk')
    p.add_argument('--manifest',type=project_path)
    p.add_argument('--target')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',type=project_path,required=True)
    args=p.parse_args(argv)
    from .data import FeatureStore
    store=FeatureStore(args.data)
    windows=read_jsonl(args.manifest or args.data/'test.jsonl')
    eligible=[w for w in windows if w['reference'] and all(uid in store.available for uid in w['history']+[w['reference']])]
    if args.target:eligible=[w for w in eligible if w['target']==args.target]
    eligible.sort(key=lambda w:store.records[w['target']]['duration'])
    if not eligible:raise ValueError('No eligible cached job. Extract features first; target requires an earlier same-speaker reference.')
    window=eligible[args.index];target=store.records[window['target']]
    job={'schema':2,'paths_relative_to':'job_directory','target_id':target['id'],'text':target['text'],
         'data_directory':relative_path(args.data,args.output.parent),
         'dialogue_id':target['dialogue_id'],'target_turn_id':target['turn_id'],'target_speaker':target['speaker'],
         'history':[{'feature':relative_path(store.cache/(uid+'.npz'),args.output.parent),'id':uid,
                     'speaker':store.records[uid]['speaker'],'turn_id':store.records[uid]['turn_id'],
                     'audio':relative_path(data_path(store.records[uid]['audio'],args.data),args.output.parent)} for uid in window['history']],
         'reference_feature':relative_path(store.cache/(window['reference']+'.npz'),args.output.parent),
         'reference_text':store.records[window['reference']]['text'],
         'split':target['split'],'feature_fingerprint':store.index['fingerprint'],
         'input_policy':'Target transcript only; reference and history are earlier observed turns. No target caption/audio/tokens.'}
    write_json(args.output,job);print(json.dumps(job,ensure_ascii=False,indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--job',type=project_path,required=True)
    p.add_argument('--data',type=project_path,help='Dataset used only for ground-truth export; inferred from dataset jobs.')
    p.add_argument('--llm-checkpoint',type=project_path,required=True)
    p.add_argument('--flow-checkpoint',type=project_path,required=True)
    p.add_argument('--model-dir',type=project_path,default=default_model_dir())
    p.add_argument('--encoder-dir',type=project_path,default=default_encoder_dir())
    p.add_argument('--output',type=project_path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--seed',type=int,default=1986)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-caption-tokens',type=int,default=128)
    p.add_argument('--max-speech-tokens',type=int,default=750)
    p.add_argument('--min-speech-tokens',type=int,default=10)
    args=p.parse_args(argv)
    if min(args.max_caption_tokens,args.max_speech_tokens,args.min_speech_tokens)<1:p.error('Token limits must be positive.')
    if args.min_speech_tokens>args.max_speech_tokens:p.error('min-speech-tokens exceeds max-speech-tokens.')
    paths=artifact_paths(args.output)
    if len(set(paths.values()))!=len(paths):p.error('--output must use an audio extension such as .wav.')
    import numpy as np, soundfile as sf, torch
    from .models import load_component
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    def load_feature(path):
        path=data_path(path,args.job.parent)
        with np.load(path,allow_pickle=False) as f:
            if job.get('feature_fingerprint') and str(f['fingerprint'])!=job['feature_fingerprint']:raise ValueError('Job/cache fingerprint mismatch.')
            return {k:f[k].copy() for k in ['speech_token','text_token','caption_token','embedding','mel']}
    tok=tokenizer();history=[]
    job=json.loads(args.job.read_text(encoding='utf-8-sig'))
    forbidden={'target_caption','target_audio','target_feature','target_speech_token'}
    if forbidden&job.keys():raise ValueError('Inference job contains target labels/audio.')
    target_id=job.get('target_id')
    if any(h.get('id')==target_id or Path(h['feature']).stem==target_id for h in job['history']):raise ValueError('Target appears in its own history.')
    if Path(job['reference_feature']).stem==target_id:raise ValueError('Target cannot be its own reference.')
    history=[load_feature(h['feature']) for h in job['history']]
    reference=load_feature(job['reference_feature'])
    text=job['text'];reference_text=job['reference_text']
    if not text.strip():raise ValueError('Empty target text.')
    groundtruth=groundtruth_source(args,job)
    if groundtruth:
        paths.update(groundtruth_caption=args.output.with_suffix('.groundtruth.caption.txt'),
                     groundtruth_audio=args.output.with_suffix('.groundtruth'+groundtruth[2].suffix))
    protected=[args.job,args.llm_checkpoint,args.flow_checkpoint]
    protected += [data_path(h['feature'],args.job.parent) for h in job['history']]+[data_path(job['reference_feature'],args.job.parent)]
    protected += [data_path(h['audio'],args.job.parent) for h in job['history'] if h.get('audio')]
    if groundtruth:protected += [groundtruth[1],groundtruth[2]]
    if any(path.resolve()==source.resolve() for path in paths.values() for source in protected if source):
        raise ValueError('Inference outputs cannot overwrite a job, reference, history feature or checkpoint.')
    snapshot=input_snapshot(args,job,history,reference_text,text,tok)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'input':snapshot}
    write_json(paths['report'],report)
    def amp():return torch.autocast(device.type,dtype=torch.bfloat16) if args.precision=='bf16' else contextlib.nullcontext()
    llm,_=load_component('llm',args.model_dir,args.llm_checkpoint,device=device)
    llm.eval();seed_all(args.seed)
    target_tokens=tok.encode(text,allowed_special=set())
    with torch.inference_mode(),amp():
        prediction=llm.inference_chain(history,target_tokens,reference['embedding'],tok,
            max_caption_tokens=args.max_caption_tokens,max_speech_tokens=args.max_speech_tokens,
            min_speech_tokens=args.min_speech_tokens)
    caption=prediction['caption']
    paths['caption'].write_text(caption if caption is not None else '',encoding='utf-8',newline='\n')
    report['prediction']={'caption':caption,'caption_token':prediction.get('caption_token',[]),
                          'speech_token':prediction['speech_token']}
    write_json(paths['report'],report)
    del llm;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    if not prediction['speech_token']:raise RuntimeError('No generated tokens.')
    from .sentence_encoder import CaptionEncoder
    encoder=CaptionEncoder(args.encoder_dir,device='cpu')
    caption_vector=encoder.encode([prediction['caption']])
    del encoder;gc.collect()
    flow,_=load_component('flow',args.model_dir,args.flow_checkpoint,device=device)
    flow.eval();seed_all(args.seed+1)
    def tensor(value,dtype=torch.float32):return torch.as_tensor(value,device=device,dtype=dtype)
    with torch.inference_mode(),amp():
        kwargs={'caption_embedding':tensor(caption_vector)}
        mel,_=flow.inference(tensor(prediction['speech_token'],torch.long).unsqueeze(0),tensor([len(prediction['speech_token'])],torch.int32),
                    tensor(reference['speech_token'],torch.long).unsqueeze(0),tensor([len(reference['speech_token'])],torch.int32),
                    tensor(reference['mel']).unsqueeze(0),tensor([len(reference['mel'])],torch.int32),
                    tensor(reference['embedding']).unsqueeze(0),torch.zeros(1,80,0,2,device=device),**kwargs)
    del flow;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    hift,_=load_component('hift',args.model_dir,device=device);hift.eval();seed_all(args.seed+2)
    with torch.inference_mode():wav,_=hift.inference(mel.float())
    data=wav.float().cpu().numpy().reshape(-1)
    if not len(data) or not np.isfinite(data).all():raise RuntimeError('Empty or non-finite generated waveform.')
    args.output.parent.mkdir(parents=True,exist_ok=True);sf.write(args.output,data,22050,subtype='PCM_16')
    report['prediction']['audio']=args.output.name
    report['groundtruth']=export_groundtruth(groundtruth,paths)
    write_json(paths['report'],report)
    print(json.dumps({key:relative_path(path,args.output.parent) for key,path in paths.items()},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
