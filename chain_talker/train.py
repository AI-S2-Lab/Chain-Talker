import argparse, contextlib, json, math, os, platform, random, time
from pathlib import Path
from functools import partial
from .runtime import ROOT, project_path, relative_path, default_log_dir, atomic_replace, default_model_dir, seed_all, sha256, write_json

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',choices=['llm','flow'],required=True)
    p.add_argument('--data',type=project_path,default=ROOT/'data/dailytalk')
    p.add_argument('--train-manifest',type=project_path)
    p.add_argument('--valid-manifest',type=project_path)
    p.add_argument('--model-dir',type=project_path,default=default_model_dir())
    p.add_argument('--output',type=project_path,required=True)
    p.add_argument('--resume',type=project_path)
    p.add_argument('--overwrite',action='store_true',help='Explicitly restart in an existing output directory; archive previous metrics.')
    p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='bf16')
    p.add_argument('--optimizer',choices=['adamw','sgd'],default='adamw')
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--new-lr',type=float,default=1e-4)
    p.add_argument('--trainable',choices=['all','new'],default='all')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accum-grad',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--valid-batches',type=int,default=0,help='0 evaluates all validation batches.')
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1986)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no-gradient-checkpointing',action='store_true')
    p.add_argument('--log-dir',type=project_path,default=default_log_dir(),help='TensorBoard root; Linux default /root/tf-logs.')
    p.add_argument('--run-name',help='TensorBoard run name; default is the output directory name.')
    args=p.parse_args(argv)
    if args.overwrite and args.resume:p.error('--overwrite and --resume are mutually exclusive.')
    if not args.resume and (args.output/'last.pt').exists() and not args.overwrite:
        p.error('Output already has last.pt; use --resume, a new --output, or explicit --overwrite.')
    if min(args.batch_size,args.accum_grad,args.epochs)<1:p.error('batch-size, accum-grad and epochs must be positive.')
    import torch, numpy as np
    from torch.utils.data import DataLoader
    from .data import Dataset, collate
    from .models import load_component
    torch.set_num_threads(args.threads);seed_all(args.seed)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    if world>1:
        if device.type=='cuda':device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
        torch.distributed.init_process_group('nccl' if device.type=='cuda' and platform.system()!='Windows' else 'gloo')
    train_path=args.train_manifest or args.data/'train.jsonl'
    valid_path=args.valid_manifest or args.data/'valid.jsonl'
    trainset=Dataset(args.data,train_path,args.component,args.max_tokens)
    validset=Dataset(args.data,valid_path,args.component,args.max_tokens)
    if not len(trainset) or not len(validset):raise ValueError('Empty dataset.')
    model,previous=load_component(args.component,args.model_dir,args.resume,device=device)
    seed_all(args.seed+rank)
    prefixes=('caption_input.','caption_decoder.') if args.component=='llm' else ('caption_condition.',)
    for name,param in model.named_parameters():param.requires_grad_(args.trainable=='all' or name.startswith(prefixes))
    if args.component=='llm' and not args.no_gradient_checkpointing:
        model.llm.gradient_checkpointing=True
        model.text_encoder.gradient_checkpointing=True
    base_params=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith(prefixes)]
    new_params=[v for k,v in model.named_parameters() if v.requires_grad and k.startswith(prefixes)]
    groups=([{'params':base_params,'lr':args.lr,'name':'backbone'}] if base_params else [])+[{'params':new_params,'lr':args.new_lr,'name':'caption'}]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01) if args.optimizer=='adamw' else torch.optim.SGD(groups,momentum=0)
    step=int(previous.get('step',0));start_epoch=int(previous.get('next_epoch',0));start_batch=int(previous.get('next_batch',0))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and args.precision=='fp16')
    if args.resume:
        opt_path=args.resume.with_name(args.resume.stem+'.optimizer.pt')
        saved=torch.load(opt_path,map_location='cpu',weights_only=True)
        if saved['optimizer_name']!=args.optimizer:raise ValueError('Resume optimizer mismatch.')
        if previous.get('train_manifest_sha256')!=sha256(train_path):raise ValueError('Resume training manifest mismatch.')
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        for i,group in enumerate(optimizer.param_groups):group['name']='backbone' if base_params and i==0 else 'caption'
        random.setstate(saved['python_rng']);np.random.set_state((saved['numpy_rng'][0],np.array(saved['numpy_rng'][1],dtype='uint32'),*saved['numpy_rng'][2:]))
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    raw_model=model
    if world>1:model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank] if device.type=='cuda' else None,find_unused_parameters=True)
    sampler=torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=world,rank=rank,seed=args.seed) if world>1 else None
    args.output.mkdir(parents=True,exist_ok=True)
    if args.overwrite and rank==0 and (args.output/'train_metrics.jsonl').exists():
        atomic_replace(args.output/'train_metrics.jsonl',args.output/f'previous_metrics_{time.time_ns()}.jsonl')
    metadata={'component':args.component,'config_sha256':sha256(ROOT/'configs/chain_talker.yaml'),
              'train_manifest_sha256':sha256(train_path),'valid_manifest_sha256':sha256(valid_path),
              'feature_fingerprint':trainset.store.index['fingerprint'],
              'source_manifest_sha256':trainset.store.prepare_report['source_manifest_sha256'],
              'arguments':{k:(v.as_posix() if k=='log_dir' and not v.is_relative_to(ROOT) else relative_path(v)) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'trainable_parameters':sum(x.numel() for x in raw_model.parameters() if x.requires_grad),
              'total_parameters':sum(x.numel() for x in raw_model.parameters()),'world_size':world,
              'torch':str(torch.__version__),'validation':'loss validation; no claim of paper metrics or convergence'}
    if args.resume and previous.get('feature_fingerprint')!=metadata['feature_fingerprint']:raise ValueError('Resume feature cache mismatch.')
    if args.max_steps and step>=args.max_steps:raise ValueError('--max-steps must exceed the resumed global step.')
    from .training_log import TrainingLog
    logger=TrainingLog(args.log_dir,args.component,args.run_name or args.output.name,
                       enabled=rank==0,resume_step=step if args.resume else None)
    metadata['tensorboard_run']='/'.join([args.component,args.run_name or args.output.name])
    logger.configuration(metadata,step)
    if rank==0:
        print(f'TensorBoard: {logger.log_dir}',flush=True)
        write_json(args.output/'run_config.json',metadata)
        write_json(args.output/'weight_load_report.json',raw_model.load_report)
    def amp():
        if args.precision=='fp32':return contextlib.nullcontext()
        return torch.autocast(device.type,dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16)
    def evaluate():
        raw_model.eval();totals={};count=0
        indices=list(range(rank,len(validset),world))
        loader=DataLoader(torch.utils.data.Subset(validset,indices),batch_size=args.batch_size,collate_fn=partial(collate,component=args.component))
        with torch.inference_mode():
            for i,batch in enumerate(loader):
                if args.valid_batches and i>=args.valid_batches:break
                with amp():losses=raw_model(batch,device)
                n=len(batch['samples']) if args.component=='llm' else len(batch['embedding'])
                count+=n
                for k,v in losses.items():totals[k]=totals.get(k,0.)+float(v)*n
        keys=sorted(totals)
        if world>1:
            aggregate=torch.tensor([totals[k] for k in keys]+[count],dtype=torch.float64,device=device)
            torch.distributed.all_reduce(aggregate);count=int(aggregate[-1]);totals=dict(zip(keys,aggregate[:-1].tolist()))
        raw_model.train()
        if not count:raise ValueError('No validation examples.')
        result={k:v/count for k,v in totals.items()}
        if not all(math.isfinite(v) for v in result.values()):raise FloatingPointError('Non-finite validation loss.')
        logger.validation(result,step)
        return result
    def save(next_epoch,next_batch,validation):
        if rank!=0:return
        meta={**metadata,'step':step,'next_epoch':next_epoch,'next_batch':next_batch,'validation_losses':validation}
        temp=args.output/'last.pt.tmp'
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in raw_model.state_dict().items()},'metadata':meta},temp)
        atomic_replace(temp,args.output/'last.pt')
        npstate=np.random.get_state()
        opt={'optimizer_name':args.optimizer,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
             'python_rng':random.getstate(),'numpy_rng':(npstate[0],npstate[1].tolist(),*npstate[2:]),'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        torch.save(opt,args.output/'last.optimizer.pt.tmp');atomic_replace(args.output/'last.optimizer.pt.tmp',args.output/'last.optimizer.pt')
        write_json(args.output/'training_report.json',meta)
    trainable=[x for x in raw_model.parameters() if x.requires_grad]
    started=time.monotonic();last_validation={}
    try:
        for epoch in range(start_epoch,args.epochs):
            if sampler is not None:sampler.set_epoch(epoch)
            generator=torch.Generator().manual_seed(args.seed+epoch)
            loader=DataLoader(trainset,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,
                              collate_fn=partial(collate,component=args.component),generator=generator,num_workers=0)
            raw_model.train();optimizer.zero_grad(set_to_none=True);running={};micro_count=0
            for batch_index,batch in enumerate(loader):
                if epoch==start_epoch and batch_index<start_batch:continue
                remaining=len(loader)-(batch_index//args.accum_grad)*args.accum_grad
                group_size=min(args.accum_grad,remaining)
                update=(batch_index+1)%args.accum_grad==0 or batch_index+1==len(loader)
                sync=contextlib.nullcontext() if world==1 or update else model.no_sync()
                with sync:
                    with amp():losses=model(batch,device);loss=losses['loss']/group_size
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss.')
                    scaler.scale(loss).backward()
                micro_count+=1
                for k,v in losses.items():running[k]=running.get(k,0.)+float(v.detach())
                if not update:continue
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(trainable,5.)
                if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradients.')
                new_norms=[torch.linalg.vector_norm(v.grad.detach().float()) for v in new_params if v.grad is not None]
                new_grad_norm=float(torch.linalg.vector_norm(torch.stack(new_norms))) if new_norms else 0.0
                scaler.step(optimizer);scaler.update();step+=1
                optimizer.zero_grad(set_to_none=True)
                record={'step':step,'epoch':epoch,'batch':batch_index,'elapsed_seconds':time.monotonic()-started,
                        **{k:v/micro_count for k,v in running.items()},'grad_norm':float(norm),'new_grad_norm_after_clip':new_grad_norm}
                if rank==0:
                    logger.train(record,optimizer,device)
                    print(json.dumps(record),flush=True)
                    with (args.output/'train_metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
                running={};micro_count=0
                finished=args.max_steps and step>=args.max_steps
                if finished or (args.save_every and step%args.save_every==0):
                    last_validation=evaluate()
                    save(epoch+1 if batch_index+1==len(loader) else epoch,
                         0 if batch_index+1==len(loader) else batch_index+1,last_validation)
                if finished:
                    return
            start_batch=0
            last_validation=evaluate();save(epoch+1,0,last_validation)
            if rank==0:print(json.dumps({'epoch':epoch,'validation':last_validation}),flush=True)
    finally:
        logger.close()
        if world>1 and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
