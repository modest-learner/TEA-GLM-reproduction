import math
import torch
from transformers import LlamaConfig, get_scheduler
from model import GraphEncoder, InstructGLM
from .utils import get_optimizer, load_checkpoint, resolve_checkpoint
from .instruction_preprocess import InstructionDataset


class TrainerBase:
    def __init__(self, args, device):
        self.args, self.cur_device = args, torch.device(device)

    def create_dataloader(self, tokenizer):
        dataset = InstructionDataset(tokenizer, self.args,
                                     mode='test' if self.args.inference else 'train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
                                          drop_last=False, shuffle=not self.args.inference,
                                          collate_fn=dataset.collate_fn,
                                          pin_memory=self.cur_device.type == 'cuda')

    def create_config(self):
        if self.args.config_class != 'LlamaConfig' or self.args.model_class != 'InstructGLM':
            raise ValueError('This pipeline supports LlamaConfig / InstructGLM')
        config = LlamaConfig.from_pretrained(self.args.backbone)
        config.attention_dropout = self.args.dropout
        return config

    def create_model(self):
        config = self.create_config()
        if self.args.dtype == 'auto':
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.cur_device.type == 'cuda' else torch.float32
        else:
            dtype = getattr(torch, self.args.dtype)
        model = InstructGLM.from_pretrained(self.args.backbone, config=config,
                                            torch_dtype=dtype, low_cpu_mem_usage=True).to(self.cur_device)
        model.config.pad_token_id = self.args.pad_token_id
        model.generation_config.pad_token_id = self.args.pad_token_id
        if self.args.method == 'vicuna_text':
            model.requires_grad_(False)
            return None, model
        if not self.args.inference and self.args.gradient_checkpointing:
            # Transformers 4.28.0 exposes a parameterless checkpointing API.
            # Projected inputs_embeds require gradients, so PyTorch 1.13's
            # reentrant checkpointing propagates loss into the projector even
            # when the complete decoder is frozen.
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        llama_embeds = model.get_input_embeddings().weight.detach()
        if self.args.freeze_llama:
            model.requires_grad_(False)
        else:
            from peft import LoraConfig, get_peft_model, PeftModel
            if self.args.lora_weights:
                model = PeftModel.from_pretrained(model, self.args.lora_weights,
                                                  is_trainable=not self.args.inference)
            else:
                model = get_peft_model(model, LoraConfig(
                    r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'lm_head'],
                    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'))
        self.args.gnn_output = llama_embeds.shape[1]
        first_model = GraphEncoder(self.args, llama_embed=llama_embeds).to(self.cur_device, dtype=dtype)
        if self.args.inference or self.args.projector_checkpoint:
            checkpoint = resolve_checkpoint(self.args, 'first_model')
            self.args.loaded_projector_checkpoint = str(checkpoint.resolve())
            try:
                first_model.load_state_dict(load_checkpoint(checkpoint), strict=True)
            except RuntimeError as exc:
                raise ValueError(f'Incompatible projector/GNN checkpoint {checkpoint}. '
                                 'Published weights require Vicuna-7B-v1.5, 128 input features, '
                                 '2048 hidden units, 2 GNN layers and 5 graph tokens.') from exc
        else:
            first_model.GT.load_state_dict(load_checkpoint(resolve_checkpoint(self.args, 'gnn')))
        first_model.GT.requires_grad_(False)
        first_model.GT.eval()
        if self.args.inference and not self.args.freeze_llama and not self.args.lora_weights:
            model.load_state_dict(load_checkpoint(resolve_checkpoint(self.args, 'model')))
        return first_model, model

    def create_optimizer_and_scheduler(self, first_model, model, train_loader):
        groups = [dict(params=[p for p in module.parameters() if p.requires_grad],
                       weight_decay=self.args.weight_decay) for module in (first_model, model)]
        optimizer_class = get_optimizer(self.args.optim)
        kwargs = dict(lr=self.args.lr)
        if self.args.optim != 'sgd':
            kwargs['eps'] = self.args.adam_eps
        if self.args.optim in ('adam', 'adamw', 'adamax'):
            kwargs['betas'] = (self.args.adam_beta1, self.args.adam_beta2)
        optimizer = optimizer_class(groups, **kwargs)
        total_steps = math.ceil(len(train_loader) / self.args.grad_steps) * self.args.epoch
        scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=int(total_steps * self.args.warmup_ratio),
                                  num_training_steps=total_steps)
        return optimizer, scheduler
