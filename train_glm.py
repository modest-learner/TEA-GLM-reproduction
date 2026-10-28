"""TEA-GLM training and inference entry point (original repository layout)."""
import gc
import traceback
from copy import deepcopy
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoTokenizer

from config import parse_args
from model.language_model import generation_kwargs, resolved_generation_settings
from utils.trainer_base import TrainerBase
from utils.utils import seed_everything, print_trainable_params, InferenceRecorder


def main(args, seed, accelerator):
    if args.inference:
        output_parent = Path(args.results_dir) / args.test_dataset
        if ((output_parent / args.prefix).exists() or
                (output_parent / f'{args.prefix}_model_results.txt').exists()):
            raise FileExistsError('Inference outputs already exist; choose a new --prefix or --results_dir')
    seed_everything(seed)
    args.run_seed = seed
    if args.log_with:
        accelerator.init_trackers(args.project, config=vars(args))
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        if args.method == 'tea_glm':
            tokenizer.add_special_tokens({'additional_special_tokens':
                                          [f'<Node {i}>' for i in range(1, args.num_token + 1)]})
    args.pad_token_id = tokenizer.pad_token_id
    trainer = TrainerBase(args, accelerator.device)
    loader = trainer.create_dataloader(tokenizer)
    first_model, model = trainer.create_model()
    context_limit = model.config.max_position_embeddings
    trainable, total = print_trainable_params(first_model, model)
    accelerator.print(f'Trainable parameters: {trainable} / {total}')
    # PyG graphs have a different leading dimension from the token batch. The
    # entire collated graph batch must be moved, never split by Accelerator.
    if first_model is None:
        model, loader = accelerator.prepare(model, loader)
    else:
        first_model, model, loader = accelerator.prepare(first_model, model, loader)
    if args.inference:
        evaluate(args, accelerator, first_model, model, loader, tokenizer)
    else:
        optimizer, scheduler = trainer.create_optimizer_and_scheduler(first_model, model, loader)
        optimizer = accelerator.prepare(optimizer)
        optimizer.zero_grad()
        for epoch in range(args.epoch):
            first_model.train()
            # Checkpointing is active only in train mode; frozen parameters do
            # not stop autograd from carrying the loss back to graph tokens.
            model.train()
            loss_sum, examples = 0., 0
            for step, batch in enumerate(tqdm(loader, disable=not accelerator.is_main_process, desc=f'Epoch {epoch}')):
                # Accelerate 0.20.3 accepts exactly one module in accumulate().
                # It updates sync_gradients on entry; suppress LLM DDP sync as
                # well when optional LoRA training accumulates several batches.
                with accelerator.accumulate(first_model), (
                    nullcontext() if accelerator.sync_gradients else accelerator.no_sync(model)
                ):
                    if batch['input_ids'].size(1) > context_limit:
                        raise ValueError('Training sequence exceeds the base LLM context window')
                    embeds = first_model(batch['input_ids'], batch['is_node'], batch['graph'])
                    output = model(inputs_embeds=embeds, attention_mask=batch['attn_mask'],
                                   labels=batch['target_ids'], use_cache=False)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    # Accelerate divides by grad_steps. Correct the final partial
                    # window so it has the same weight as a full accumulation.
                    remainder = len(loader) % args.grad_steps
                    in_tail = remainder and step >= len(loader) - remainder
                    scaled_loss = loss * (args.grad_steps / remainder if in_tail else 1.)
                    accelerator.backward(scaled_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                                    args.clip_grad_norm)
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad()
                batch_size = batch['input_ids'].size(0)
                loss_sum += loss.detach().float().item() * batch_size
                examples += batch_size
            stats = torch.tensor([loss_sum, examples], device=accelerator.device)
            stats = accelerator.reduce(stats, reduction='sum')
            accelerator.print(f'Epoch {epoch}: loss={stats[0].item() / stats[1].item():.6f}')
            if args.log_with:
                accelerator.log({'epoch': epoch, 'loss': (stats[0] / stats[1]).item()})
            save_models(args, accelerator, first_model, model, epoch)
    # Accelerate 0.20.3 creates `trackers` only in init_trackers().
    if args.log_with:
        accelerator.end_training()


def save_models(args, accelerator, first_model, model, epoch):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for kind, marker, module in [('first_model', 'fm', first_model), ('model', 'm', model)]:
            if kind == 'model' and args.freeze_llama:
                continue
            folder = Path(args.save_dir) / kind
            folder.mkdir(parents=True, exist_ok=True)
            filename = folder / f'{args.prefix}_{marker}_{args.dataset}_epoch{epoch}_end.pth'
            accelerator.save(accelerator.unwrap_model(module).state_dict(), filename)
            accelerator.print(f'Saved {filename}')


def evaluate(args, accelerator, first_model, model, loader, tokenizer):
    # Loading/initializing the extra graph branch consumes RNG state. Start
    # decoding from the recorded seed for both methods in the paired comparison.
    seed_everything(args.run_seed)
    if first_model is not None:
        first_model.eval()
    model.eval()
    generator = accelerator.unwrap_model(model)
    decoding = generation_kwargs(max_new_tokens=args.gen_max_length, do_sample=args.do_sample,
                                  temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                  pad_token_id=tokenizer.pad_token_id)
    settings = resolved_generation_settings(generator, decoding)
    settings.update(model_dtype=str(next(generator.parameters()).dtype), cuda_runtime=torch.version.cuda,
                    device=str(accelerator.device), num_processes=accelerator.num_processes,
                    gpu=torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == 'cuda' else None)
    recorder = InferenceRecorder(args, loader.dataset, settings) if accelerator.is_main_process else None
    try:
        for batch in tqdm(loader, disable=not accelerator.is_main_process, desc=f'Evaluating {args.method}'):
            with torch.no_grad():
                if batch['input_ids'].size(1) + args.gen_max_length > generator.config.max_position_embeddings:
                    raise ValueError('Prompt plus --gen_max_length exceeds the base LLM context window')
                states = None
                if first_model is None:
                    embeds = generator.get_input_embeddings()(batch['input_ids'])
                elif args.save_features:
                    embeds, states = first_model(batch['input_ids'], batch['is_node'], batch['graph'], return_features=True)
                else:
                    embeds = first_model(batch['input_ids'], batch['is_node'], batch['graph'])
                # Both methods use inputs_embeds and the same decoding path.
                results = generator.generate(generation_config=deepcopy(generator.generation_config),
                                             inputs_embeds=embeds, attention_mask=batch['attn_mask'], **decoding)
                results = accelerator.pad_across_processes(results, dim=1, pad_index=tokenizer.pad_token_id)
                inputs = accelerator.pad_across_processes(batch['input_ids'], dim=1, pad_index=tokenizer.pad_token_id)
                masks = accelerator.pad_across_processes(batch['attn_mask'], dim=1, pad_index=0)
                payload = (results, batch['sample_index'], inputs, masks)
                if states is not None:
                    payload += (states['H_GNN'], states['T_graph'])
                gathered = accelerator.gather_for_metrics(payload)
            if recorder is not None:
                results, indices, inputs, masks = [t.detach().cpu() for t in gathered[:4]]
                output_tokens = []
                for row in results.tolist():
                    # Keep BOS and EOS in token logs, dropping only post-EOS padding.
                    if tokenizer.eos_token_id in row:
                        row = row[:row.index(tokenizer.eos_token_id) + 1]
                    output_tokens.append(row)
                input_tokens = [row[mask.bool()].tolist() for row, mask in zip(inputs, masks)]
                outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw = tokenizer.batch_decode(output_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                graph_features = dict(zip(('H_GNN', 'T_graph'), gathered[4:])) if states is not None else None
                recorder.write_batch(indices.tolist(), outputs, raw, output_tokens, input_tokens, graph_features)
        if recorder is not None:
            if len(recorder.seen) != len(loader.dataset):
                raise RuntimeError('Evaluation count differs from instruction count')
            metrics = recorder.finish(True)
            if metrics:
                accelerator.print(f"Accuracy={metrics['accuracy']:.6f}, Macro F1={metrics['macro_f1']:.6f}, "
                                  f"Valid answers={metrics['valid_answer_rate']:.6f}")
            accelerator.print(f'Saved complete inference artifacts to {recorder.folder}')
    except BaseException:
        if recorder is not None:
            failure = traceback.format_exc()
            try:
                recorder.finish(False, failure)
            except Exception:
                accelerator.print('Artifact finalization failed; flushed JSONL and feature_store files are retained.\n'
                                  + traceback.format_exc())
        raise


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested, but CUDA is unavailable')
    for exp in range(args.exp_num):
        run_args = deepcopy(args)
        seed = args.seed + exp if args.seed != -1 else exp
        if args.exp_num > 1:
            run_args.prefix = f'{args.prefix}_seed{seed}'
        accelerator = Accelerator(cpu=args.device == 'cpu', log_with=args.log_with,
                                  split_batches=False, dispatch_batches=False,
                                  kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False),
                                                   InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
                                  gradient_accumulation_steps=args.grad_steps)
        main(run_args, seed, accelerator)
        accelerator.free_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
