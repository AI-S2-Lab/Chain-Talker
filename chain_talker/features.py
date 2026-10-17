"""Resumable official CAM++/25Hz-token extraction and frozen caption encoding."""
import argparse, concurrent.futures as cf, hashlib, json, os, time, threading
from .runtime import ROOT, project_path, data_path, audio, default_encoder_dir, default_model_dir, read_jsonl, sha256, tokenizer, write_json, validate_prepared_data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=project_path,default=ROOT/'data/dailytalk')
    p.add_argument('--model-dir',type=project_path,default=default_model_dir())
    p.add_argument('--encoder-dir',type=project_path,default=default_encoder_dir())
    p.add_argument('--caption-device',default='cpu')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--onnx-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gpu-concurrency',type=int,default=1,help='Serialize GPU Run calls while overlapping CPU audio processing.')
    p.add_argument('--verify-cache',action='store_true',help='Also open every existing selected NPZ to verify its fingerprint.')
    args=p.parse_args(argv)
    validate_prepared_data(args.data,verify_sources=True)
    import sys
    if args.onnx_device=='cuda' and (ROOT/'.ort-gpu').exists():sys.path.insert(0,str(ROOT/'.ort-gpu'))
    import numpy as np, torch, onnxruntime as ort, whisper
    import torchaudio.compliance.kaldi as kaldi
    from matcha.utils.audio import mel_spectrogram
    from .sentence_encoder import CaptionEncoder
    torch.set_num_threads(1)
    rows=read_jsonl(args.data/'utterances.jsonl');records={r['id']:r for r in rows}
    meta={'schema':2,'manifest_sha256':sha256(args.data/'utterances.jsonl'),
          'semantic_backend':args.onnx_device,'onnxruntime_version':str(ort.__version__),
          'speech_tokenizer_sha256':sha256(args.model_dir/'speech_tokenizer_v1.onnx'),
          'campplus_sha256':sha256(args.model_dir/'campplus.onnx'),
          'caption_model_sha256':sha256(args.encoder_dir/'model.safetensors'),
          'caption_dense_sha256':sha256(args.encoder_dir/'2_Dense/model.safetensors') if (args.encoder_dir/'2_Dense/model.safetensors').exists() else sha256(args.encoder_dir/'2_Dense/pytorch_model.bin'),
          'audio_sample_rates':{'semantic_and_speaker':16000,'mel':22050},'mel':{'n_fft':1024,'hop':256,'bins':80,'fmax':8000},
          'caption_pooling':'mean + Dense + tanh; no L2 normalization','speaker':'per utterance CAM++; no global local-ID averaging'}
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    cache=args.data/'features'/fingerprint[:16];cache.mkdir(parents=True,exist_ok=True)
    existing_ids={path.stem for path in cache.glob('*.npz')}
    pending=[]
    for row in rows:
        path=cache/(row['id']+'.npz')
        if row['id'] in existing_ids:
            if args.verify_cache:
                with np.load(path,allow_pickle=False) as f:
                    if str(f['fingerprint'])!=fingerprint:raise ValueError(f'Stale cache: {path}')
        else:pending.append(row)
    print(f'Feature cache: {cache}; selected={len(rows)}, pending={len(pending)}',flush=True)
    caption_vectors={}
    if pending:
        vector_cache=cache/'caption_vectors.npz'
        if vector_cache.exists():
            with np.load(vector_cache,allow_pickle=False) as existing:
                caption_vectors=dict(zip(existing['ids'].tolist(),existing['vectors']))
        missing_vectors=[r for r in pending if r['id'] not in caption_vectors]
        if missing_vectors:
            encoder=CaptionEncoder(args.encoder_dir,args.caption_device)
            vectors=encoder.encode([r['caption'] for r in missing_vectors],batch_size=32)
            caption_vectors.update({r['id']:v for r,v in zip(missing_vectors,vectors)})
            del encoder
            with open(vector_cache.with_suffix('.tmp'),'wb') as f:np.savez_compressed(f,ids=np.array(list(caption_vectors)),vectors=np.stack(list(caption_vectors.values())))
            os.replace(vector_cache.with_suffix('.tmp'),vector_cache)
        print(f'Caption embeddings ready: {len(caption_vectors)}',flush=True)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    tok=tokenizer()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=3
    providers=['CPUExecutionProvider']
    if args.onnx_device=='cuda':
        if hasattr(ort,'preload_dlls'):ort.preload_dlls()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA ONNX Runtime unavailable; install optional GPU runtime or use --onnx-device cpu.')
        providers=[('CUDAExecutionProvider',{'device_id':0,'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':2<<30}), 'CPUExecutionProvider']
    semantic=ort.InferenceSession(str(args.model_dir/'speech_tokenizer_v1.onnx'),sess_options=options,providers=providers) if pending else None
    if pending and args.onnx_device=='cuda' and semantic.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA provider creation failed; refusing silent CPU fallback.')
    speaker=ort.InferenceSession(str(args.model_dir/'campplus.onnx'),sess_options=options,providers=['CPUExecutionProvider']) if pending else None
    gate=threading.Semaphore(args.gpu_concurrency if args.onnx_device=='cuda' else args.workers)
    def extract(row):
        wav16=audio(data_path(row['audio'],args.data),16000)
        feat=whisper.log_mel_spectrogram(wav16,n_mels=128)
        with gate:
            speech=semantic.run(None,{semantic.get_inputs()[0].name:feat.numpy(),semantic.get_inputs()[1].name:np.array([feat.shape[2]],dtype=np.int32)})[0].reshape(-1).astype('int64')
        if not len(speech) or speech.min()<0 or speech.max()>=4096:raise ValueError('Invalid semantic tokens')
        fbank=kaldi.fbank(wav16,num_mel_bins=80,dither=0,sample_frequency=16000)
        fbank=fbank-fbank.mean(0,keepdim=True)
        embedding=speaker.run(None,{speaker.get_inputs()[0].name:fbank.unsqueeze(0).numpy()})[0].reshape(-1).astype('float32')
        if embedding.shape!=(192,):raise ValueError('Invalid CAM++ dimensions')
        wav22=audio(data_path(row['audio'],args.data),22050)
        mel=mel_spectrogram(wav22,n_fft=1024,num_mels=80,sampling_rate=22050,hop_size=256,win_size=1024,fmin=0,fmax=8000,center=False)[0].T.numpy().astype('float32')
        if not np.isfinite(mel).all() or not len(mel):raise ValueError('Invalid Mel')
        text_ids=np.array(tok.encode(row['text'],allowed_special=set()),dtype='int64')
        caption_ids=np.array(tok.encode(row['caption'],allowed_special=set()),dtype='int64')
        if not len(text_ids) or not len(caption_ids):raise ValueError('Empty BPE sequence')
        if max(text_ids.max(),caption_ids.max())>=60515:raise ValueError('Tokenizer/checkpoint vocabulary mismatch')
        path=cache/(row['id']+'.npz');temp=path.with_suffix('.tmp')
        with open(temp,'wb') as f:
            np.savez_compressed(f,speech_token=speech,text_token=text_ids,caption_token=caption_ids,
                               embedding=embedding,caption_embedding=caption_vectors[row['id']],mel=mel,
                               fingerprint=fingerprint,audio_sha256=sha256(data_path(row['audio'],args.data)))
        os.replace(temp,path)
        return row['id']
    failures=[];started=time.monotonic();done=0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(extract,row):row for row in pending}
        for future in cf.as_completed(futures):
            row=futures[future]
            try:future.result();done+=1
            except Exception as e:failures.append({'id':row['id'],'error':repr(e)})
            if (done+len(failures))%25==0 or done+len(failures)==len(pending):
                print(f'Features {done}/{len(pending)}; failed={len(failures)}; seconds={time.monotonic()-started:.1f}',flush=True)
    available=len(set(records)&{path.stem for path in cache.glob('*.npz')})
    report={**meta,'fingerprint':fingerprint,'cache_relative':cache.relative_to(args.data).as_posix(),
            'selected':len(rows),'processed_now':done,'available_total':available,'manifest_total':len(rows),
            'complete':available==len(rows),'failures':failures,'elapsed_seconds':time.monotonic()-started,
            'onnx_device':args.onnx_device,'onnxruntime_version':ort.__version__}
    write_json(args.data/'features_index.json',report)
    write_json(args.data/'features_report.json',report)
    print(json.dumps({k:report[k] for k in ['selected','processed_now','available_total','manifest_total','complete','failures']},indent=2),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
