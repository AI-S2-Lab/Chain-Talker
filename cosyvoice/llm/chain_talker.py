from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from cosyvoice.llm.llm import TransformerLM


class EmGPT(TransformerLM):
    def __init__(self, *args, caption_loss_weight=1.0, speech_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_input = nn.Linear(self.text_embedding.embedding_dim, self.llm_input_size)
        self.caption_decoder = nn.Linear(self.llm_decoder.in_features, self.text_embedding.num_embeddings)
        self.caption_loss_weight = caption_loss_weight
        self.speech_loss_weight = speech_loss_weight
        self.sps_id = self.spe_id = None

    def set_control_tokens(self, sps_id, spe_id):
        assert sps_id != spe_id and max(sps_id, spe_id) < self.text_embedding.num_embeddings
        self.sps_id, self.spe_id = int(sps_id), int(spe_id)

    def _caption(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.text_embedding.weight.device)
        return self.caption_input(self.text_embedding(ids))

    def _speaker(self, vector):
        vector = torch.as_tensor(vector, device=self.text_embedding.weight.device, dtype=self.text_embedding.weight.dtype)
        return self.spk_embed_affine_layer(F.normalize(vector.reshape(1, -1), dim=1))

    def _text(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.text_embedding.weight.device).reshape(1, -1)
        if ids.numel() == 0:
            raise ValueError('Empty transcript tokens.')
        encoded, length = self.encode(self.text_embedding(ids), torch.tensor([ids.shape[1]], device=ids.device))
        return encoded[0, :int(length[0])]

    def prefix(self, history, text_tokens, speaker_embedding):
        if self.sps_id is None:
            raise RuntimeError('Call set_control_tokens first.')
        pieces = [self.llm_embedding.weight[self.sos].reshape(1, -1)]
        for turn in history:
            pieces += [self._speaker(turn['embedding']),
                       self.speech_embedding(torch.as_tensor(turn['speech_token'], dtype=torch.long, device=pieces[0].device)),
                       self._text(turn['text_token']), self._caption([self.sps_id]),
                       self._caption(turn['caption_token']), self._caption([self.spe_id])]
        pieces += [self._speaker(speaker_embedding), self._text(text_tokens), self._caption([self.sps_id])]
        return torch.cat(pieces, dim=0)

    def pack(self, samples):
        sequences, caption_targets, speech_targets = [], [], []
        device = self.text_embedding.weight.device
        for sample in samples:
            target = sample['target']
            prefix = self.prefix(sample['history'], target['text_token'], target['embedding'])
            caption = list(map(int, target['caption_token'])) + [self.spe_id]
            speech = torch.as_tensor(target['speech_token'], dtype=torch.long, device=device)
            if not speech.numel(): raise ValueError('Empty target speech tokens.')
            sequence = torch.cat([prefix, self._caption(caption), self.speech_embedding(speech)], dim=0)
            ct = torch.full((len(sequence),), -1, dtype=torch.long, device=device)
            st = ct.clone()
            p = len(prefix)
            ct[p-1:p-1+len(caption)] = torch.tensor(caption, device=device)
            st[p+len(caption)-1:] = torch.cat([speech, torch.tensor([self.eos_token], device=device)])
            sequences.append(sequence); caption_targets.append(ct); speech_targets.append(st)
        lengths = torch.tensor([len(x) for x in sequences], device=device, dtype=torch.int32)
        return (pad_sequence(sequences, batch_first=True, padding_value=0.0), lengths,
                pad_sequence(caption_targets, batch_first=True, padding_value=-1),
                pad_sequence(speech_targets, batch_first=True, padding_value=-1))

    def forward(self, batch, device=None):
        packed, lengths, captions, speech = self.pack(batch['samples'])
        hidden, _ = self.llm(packed, lengths)
        cmask, smask = captions >= 0, speech >= 0
        clogits = self.caption_decoder(hidden[cmask]).float()
        slogits = self.llm_decoder(hidden[smask]).float()
        lc = F.cross_entropy(clogits, captions[cmask])
        ls = F.cross_entropy(slogits, speech[smask])
        return {'loss': self.caption_loss_weight*lc + self.speech_loss_weight*ls,
                'loss_caption': lc, 'loss_speech': ls,
                'acc_caption': (clogits.argmax(-1) == captions[cmask]).float().mean(),
                'acc_speech': (slogits.argmax(-1) == speech[smask]).float().mean()}

    @torch.inference_mode()
    def inference_chain(self, history, text_tokens, speaker_embedding, tokenizer,
                        max_caption_tokens=128, max_speech_tokens=750, min_speech_tokens=10,
                        top_k=25):
        self.eval()
        prefix = self.prefix(history, text_tokens, speaker_embedding)
        current = prefix.unsqueeze(0)
        offset = 0
        cache = torch.zeros((0, 0, 0, 0), device=current.device)
        att_cache, cnn_cache = cache, cache

        def step(value):
            nonlocal offset, att_cache, cnn_cache
            length = value.shape[1]
            attention_mask = (torch.arange(offset+length,device=value.device)[None,:]
                              <= torch.arange(offset,offset+length,device=value.device)[:,None]).unsqueeze(0)
            output, att_cache, cnn_cache = self.llm.forward_chunk(
                value, offset=offset, required_cache_size=-1, att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=attention_mask)
            offset += length
            return output[:, -1]

        hidden = step(current)
        caption_ids = []
        caption_truncated = False
        special_ids = list(tokenizer.encoding._special_tokens.values())
        for i in range(max_caption_tokens):
            scores = self.caption_decoder(hidden).float().squeeze(0)
            scores[tokenizer.encoding.n_vocab:] = -torch.inf
            scores[special_ids] = -torch.inf
            if i >= 1:
                scores[self.spe_id] = self.caption_decoder(hidden).float()[0,self.spe_id]
            values, ids = scores.topk(min(top_k, scores.numel()))
            chosen = int(ids[torch.multinomial(values.softmax(-1), 1)])
            if chosen == self.spe_id:
                hidden = step(self._caption([chosen]).unsqueeze(0)); break
            caption_ids.append(chosen)
            hidden = step(self._caption([chosen]).unsqueeze(0))
        else:
            caption_truncated = True
            hidden = step(self._caption([self.spe_id]).unsqueeze(0))
        speech_ids: List[int] = []
        speech_truncated = True
        for i in range(max_speech_tokens):
            logp = self.llm_decoder(hidden).float().squeeze(0).log_softmax(-1)
            chosen = int(self.sampling_ids(logp, speech_ids, top_k, ignore_eos=i < min_speech_tokens))
            if chosen == self.eos_token:
                speech_truncated = False; break
            speech_ids.append(chosen)
            hidden = step(self.speech_embedding.weight[chosen].reshape(1,1,-1))
        if not speech_ids: raise RuntimeError('No speech tokens generated.')
        return {'caption': tokenizer.decode(caption_ids), 'caption_token': caption_ids,
                'speech_token': speech_ids, 'caption_truncated': caption_truncated,
                'speech_truncated': speech_truncated, 'caption_source': 'predicted'}
