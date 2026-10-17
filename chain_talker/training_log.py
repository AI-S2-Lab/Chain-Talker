import json
import re
from pathlib import Path


class TrainingLog:
    def __init__(self, root, component, name, *, enabled=True, resume_step=None):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
            raise ValueError('TensorBoard run name must be a single name: letters, digits, dot, dash or underscore.')
        self.log_dir=Path(root)/component/name
        self.writer=None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError('TensorBoard is required: install requirements.txt in the active environment.') from exc
            self.writer=SummaryWriter(str(self.log_dir), purge_step=resume_step+1 if resume_step is not None else 0,
                                      max_queue=20, flush_secs=30)

    def configuration(self, metadata, step):
        if self.writer:
            self.writer.add_text('run/config', '```json\n'+json.dumps(metadata,ensure_ascii=False,indent=2)+'\n```', step)
            self.writer.flush()

    def train(self, record, optimizer, device):
        if not self.writer:return
        step=record['step']
        for name,value in record.items():
            if name not in {'step','epoch','batch'} and isinstance(value,(int,float)):
                self.writer.add_scalar('train/'+name,value,step)
        self.writer.add_scalar('train/epoch',record['epoch']+1,step)
        for i,group in enumerate(optimizer.param_groups):
            self.writer.add_scalar('lr/'+group.get('name',f'group_{i}'),group['lr'],step)
        if device.type=='cuda':
            import torch
            self.writer.add_scalar('system/cuda_allocated_mb',torch.cuda.memory_allocated(device)/2**20,step)
            self.writer.add_scalar('system/cuda_reserved_mb',torch.cuda.memory_reserved(device)/2**20,step)

    def validation(self, metrics, step):
        if self.writer:
            for name,value in metrics.items():self.writer.add_scalar('validation/'+name,value,step)
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
