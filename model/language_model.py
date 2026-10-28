"""Llama/OPT interfaces with upstream causal-LM loss and cache handling."""
from copy import deepcopy
import torch
from transformers import LlamaForCausalLM, OPTForCausalLM


def generation_kwargs(*, max_new_tokens=80, do_sample=True, temperature=0.7,
                      top_p=None, top_k=None, pad_token_id=None, eos_token_id=None):
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  temperature=temperature if do_sample else 1.0)
    # The release specifies only do_sample, temperature, and max_new_tokens.
    # In particular, Vicuna's top_p=0.6 must not be replaced by a default 1.0.
    if do_sample:
        if top_p is not None:
            kwargs['top_p'] = top_p
        if top_k is not None:
            kwargs['top_k'] = top_k
    else:
        kwargs.update(top_p=1.0, top_k=50)
    if pad_token_id is not None:
        kwargs['pad_token_id'] = pad_token_id
    if eos_token_id is not None:
        kwargs['eos_token_id'] = eos_token_id
    return kwargs


def resolved_generation_settings(model, kwargs):
    config = deepcopy(model.generation_config)
    config.update(**kwargs)
    fields = ('do_sample', 'temperature', 'max_new_tokens', 'top_p', 'top_k',
              'num_beams', 'repetition_penalty', 'pad_token_id', 'eos_token_id')
    return {name: getattr(config, name) for name in fields}


class GraphGenerationMixin:
    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask, max_new_tokens=80, do_sample=True,
               temperature=0.7, top_p=None, top_k=None):
        self.eval()
        return self.generate(generation_config=deepcopy(self.generation_config),
                             inputs_embeds=in_embeds, attention_mask=attention_mask,
                             **generation_kwargs(max_new_tokens=max_new_tokens, do_sample=do_sample,
                                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                                 pad_token_id=self.config.pad_token_id,
                                                 eos_token_id=self.config.eos_token_id))


class InstructGLM(GraphGenerationMixin, LlamaForCausalLM):
    pass


class OptGLM(GraphGenerationMixin, OPTForCausalLM):
    pass
