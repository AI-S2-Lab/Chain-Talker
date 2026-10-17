import torch
from torch import nn
from cosyvoice.flow.flow import MaskedDiffWithXvec


class CaptionMaskedDiffWithXvec(MaskedDiffWithXvec):
    def __init__(self, *args, caption_dim=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_condition = nn.Linear(caption_dim, self.input_size)
        nn.init.zeros_(self.caption_condition.weight)
        nn.init.zeros_(self.caption_condition.bias)

    def condition_tokens(self, token, mask, caption_embedding=None, start=0):
        if caption_embedding is None:
            raise ValueError('Chain-Talker Flow requires a caption embedding.')
        caption_embedding = caption_embedding.to(device=token.device, dtype=token.dtype)
        if caption_embedding.ndim != 2 or caption_embedding.shape[0] != token.shape[0]:
            raise ValueError('Expected caption_embedding of shape [batch, 512].')
        condition = self.caption_condition(caption_embedding).unsqueeze(1)
        target_mask = (torch.arange(token.shape[1], device=token.device) >= start)[None, :, None]
        return token + condition * mask * target_mask
