import argparse
import math
import pprint
import yaml
from os import path

module_path = path.dirname(path.abspath(__file__))


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--dataset", type=str, default='arxiv')
    parser.add_argument("--test_dataset", type=str, default='pubmed')
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument('--data_dir', default=path.join(module_path, 'data'))
    parser.add_argument('--instruction_dir', default=path.join(module_path, 'instruction'))
    parser.add_argument('--save_dir', default=path.join(module_path, 'saved_model'))
    parser.add_argument('--results_dir', default=path.join(module_path, 'results'))
    parser.add_argument('--projector_checkpoint', default='', help='Explicit first_model checkpoint; accepts the published TEA-GLM_*.pth files')
    parser.add_argument('--model_checkpoint', default='', help='Optional fine-tuned LLM checkpoint')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional deterministic subset for smoke tests; default uses all instructions')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
    parser.add_argument('--log_with', default=None, choices=['wandb', 'tensorboard'])
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--remove_target_edge', action='store_true', help='For LP, remove edges 0->1 and 1->0 from the supplied subgraph; changes the published-data protocol')

    # Model Config
    parser.add_argument('--backbone', type=str, required=True, help='Llama/Vicuna checkpoint path or Hugging Face model ID')
    parser.add_argument('--lora_weights', type=str, default='')
    parser.add_argument('--pretrain_gnn', type=str, default='')
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--prefix', type=str, default='trainable_llama_gnn')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--config_class', type=str, default='LlamaConfig')
    parser.add_argument('--model_class', type=str, default='InstructGLM')
    parser.add_argument('--gt_layers', type=int, default=2)
    parser.add_argument('--num_token', type=int, default=5)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--att_d_model', type=int, default=2048)
    parser.add_argument('--gnn_output', type=int, default=4096)
    parser.add_argument('--max_text_length', type=int, default=2215)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--freeze_llama', action='store_true', default=True)
    parser.add_argument('--train_llama', dest='freeze_llama', action='store_false', help='Opt in to LoRA tuning (outside the paper setting)')
    parser.add_argument('--optim', default='adam', help='Paper: Adam; released shell scripts: AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--best_epoch', type=int, default=0)

    # Inference
    parser.add_argument('--method', choices=['tea_glm', 'vicuna_text'], default='tea_glm',
                        help='vicuna_text removes only graph placeholders from the shared instruction')
    parser.add_argument('--save_features', action='store_true',
                        help='Persist X and, for TEA-GLM, H_GNN/T_graph for CPU diagnostics')
    parser.add_argument('--verbose', action='store_true', help='Print each prediction in addition to the durable JSONL log')
    parser.add_argument('--gen_max_length', type=int, default=80)
    parser.add_argument('--do_sample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=None, help='Default: inherit the base model generation_config.json, as in the author code')
    parser.add_argument('--top_k', type=int, default=None, help='Default: inherit the base model generation config')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)
    if kwargs['method'] == 'vicuna_text':
        if not kwargs['inference'] or not kwargs['freeze_llama']:
            parser.error('vicuna_text is a frozen, inference-only baseline')
        if any(kwargs[k] for k in ('projector_checkpoint', 'model_checkpoint', 'lora_weights')):
            parser.error('vicuna_text must not load a projector, fine-tuned LLM or LoRA checkpoint')
    if kwargs['save_features'] and not kwargs['inference']:
        parser.error('--save_features is for inference')
    for key in ('exp_num', 'batch_size', 'grad_steps', 'epoch', 'num_token', 'gt_layers', 'max_text_length', 'gen_max_length'):
        if kwargs[key] < 1:
            parser.error(f'--{key} must be positive')
    if kwargs['max_samples'] is not None and kwargs['max_samples'] < 1:
        parser.error('--max_samples must be positive')
    if not math.isfinite(kwargs['temperature']) or kwargs['temperature'] <= 0:
        parser.error('--temperature must be finite and positive')
    if kwargs['top_p'] is not None and not 0 < kwargs['top_p'] <= 1:
        parser.error('--top_p must be in (0, 1]')
    if kwargs['top_k'] is not None and kwargs['top_k'] < 0:
        parser.error('--top_k must be nonnegative')

    args = Config(**kwargs)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.safe_load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
