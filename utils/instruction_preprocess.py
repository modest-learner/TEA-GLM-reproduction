import pandas as pd
import re
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from .utils import load_checkpoint, select_instruction_indices, token_source_indices
from .conversation import conv_templates

IGNORE_TOKEN_ID = -100
ECOM_NAMES = {'children': 'book_children', 'history': 'book_history',
              'computer': 'computer', 'photo': 'photo', 'sports': 'sports'}


def get_instructions(data_path, max_samples=None):
    df = pd.read_json(data_path).reset_index(drop=True)
    # Keep the full ontology even when evaluating a small smoke-test subset.
    classes = sorted(set(df.loc[df['task'] == 'nc', 'output']))
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()
    if df.empty:
        raise ValueError(f'No instructions found in {data_path}')
    df['gpt'] = df['output']
    df.attrs['class_labels'] = classes
    return df


def preprocess(instruction, tokenizer, max_length, mode='train'):
    conv = conv_templates['vicuna_v1_1'].copy()
    conv.append_message(conv.roles[0], instruction['prompt'])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # Match the complete Vicuna conversation's tokenization exactly. Locate
    # the shared prefix instead of assuming tokenizer-specific +/-2 offsets.
    prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if mode == 'train':
        ids = tokenizer(prompt + ' ' + instruction['gpt'] + conv.sep2,
                        add_special_tokens=True).input_ids
        boundary = 0
        for prefix_token, full_token in zip(prefix_ids, ids):
            if prefix_token != full_token:
                break
            boundary += 1
        if boundary >= len(ids):
            raise ValueError('An instruction has no answer tokens')
        labels = [IGNORE_TOKEN_ID] * boundary + ids[boundary:]
    else:
        ids = prefix_ids
        labels = tokenizer(instruction['gpt'] + conv.sep2, add_special_tokens=False).input_ids
    if len(ids) > max_length:
        raise ValueError(f'Instruction needs {len(ids)} tokens, exceeding '
                         f'--max_text_length {max_length}. Increase the limit; '
                         'truncation can remove graph tokens or answer supervision.')
    return dict(input_ids=torch.tensor([ids], dtype=torch.long),
                target_ids=torch.tensor([labels], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.bool), text=prompt)


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train'):
        self.tokenizer, self.args, self.mode = tokenizer, args, mode
        name = args.test_dataset if args.inference else args.dataset
        instruction_path = Path(args.instruction_dir) / name / f'{name}_dataset_{mode}.json'
        self.instruction_path = instruction_path
        full = get_instructions(instruction_path)
        self.class_labels = full.attrs['class_labels']
        self.population_size = len(full)
        self.instruction_indices = select_instruction_indices(len(full), args.max_samples,
            getattr(args, 'sample_count', None), getattr(args, 'sample_seed', 2026))
        self.instructions = full.iloc[self.instruction_indices].copy().reset_index(drop=True)
        self.token_sources = None
        changed = getattr(args, 'graph_intervention', 'none') != 'none' or getattr(args, 'graph_token_scale', 1.) != 1.
        if changed and (set(self.instructions['task']) != {'nc'} or len(set(self.instructions['data'])) != 1):
            raise ValueError('Interventions require single-dataset node classification')
        if getattr(args, 'graph_intervention', 'none') == 'shuffle':
            if len({int(nodes[0]) for nodes in self.instructions['node_set']}) != len(self.instructions):
                raise ValueError('Shuffle requires distinct target node IDs')
            self.token_sources = token_source_indices(len(self.instructions), args.shuffle_seed)
        self.text_only = args.method == 'vicuna_text'
        if self.text_only and set(self.instructions['task']) != {'nc'}:
            raise ValueError('The text-only comparison is defined for node classification')
        self.node_token_ids = (torch.empty(0, dtype=torch.long) if self.text_only else
                              torch.tensor(tokenizer.convert_tokens_to_ids(
                                  [f'<Node {i}>' for i in range(1, args.num_token + 1)])))
        if not self.text_only and len(set(self.node_token_ids.tolist())) != args.num_token:
            raise ValueError('Register each <Node i> as a special token before creating the dataset')
        self.name2data = {}
        names = set(self.instructions['data'])
        for filename, selected in (
            ('graph_data_paper.pt', names - ECOM_NAMES.keys()),
            ('graph_data_ecommercial.pt', names & ECOM_NAMES.keys()),
        ):
            if selected:
                graphs = load_checkpoint(Path(args.data_dir) / filename, graph_data=True)
                for data_name in selected:
                    self.name2data[data_name] = graphs[ECOM_NAMES.get(data_name, data_name)]
        widths = {graph.x.shape[1] for graph in self.name2data.values()}
        if len(widths) != 1:
            raise ValueError(f'Incompatible node feature dimensions: {widths}')
        args.gnn_input = widths.pop()
        args.edge_dim = None

    def __len__(self):
        return len(self.instructions)

    def prepare_instruction(self, idx):
        instruction = self.instructions.iloc[idx].copy()
        prompt = instruction['prompt']
        # Citation instructions include abstracts; the paper uses titles only.
        if instruction['data'] not in ECOM_NAMES and 'Abstract: ' in prompt and 'Title: ' in prompt:
            before, remainder = prompt.split('Abstract: ', 1)
            prompt = before + 'Title: ' + remainder.split('Title: ', 1)[1]
        if self.text_only:
            # Matched baseline: remove graph tokens only. Keep every other
            # character, the same title/description, question and class list.
            instruction['prompt'] = re.sub(r'<Node \d+>', '', prompt)
        else:
            tokens = ' '.join(f'<Node {i}>' for i in range(1, self.args.num_token + 1))
            instruction['prompt'] = prompt.replace('<Node 1>', tokens)
        return instruction

    def prompt_text(self, idx):
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], self.prepare_instruction(idx)['prompt'])
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __getitem__(self, idx):
        instruction = self.prepare_instruction(idx)
        out = preprocess(instruction, self.tokenizer, self.args.max_text_length, self.mode)
        out['sample_index'] = idx
        if self.text_only:
            # No subgraph or graph-derived embeddings enter the baseline LLM.
            out['is_node'] = torch.zeros_like(out['input_ids'], dtype=torch.bool)
            return out
        out['graph'] = self.make_graph(idx)
        if self.token_sources is not None:
            out['source_graph'] = self.make_graph(self.token_sources[idx])
        out['is_node'] = (out['input_ids'].unsqueeze(-1) == self.node_token_ids).any(-1)
        if out['is_node'].sum().item() != self.args.num_token:
            raise ValueError(f'Instruction {idx} must contain exactly {self.args.num_token} graph tokens')
        return out

    def make_graph(self, idx):
        # Keep original instruction edges intact for post-hoc homophily bins.
        instruction = self.instructions.iloc[idx]
        node_list = torch.tensor(instruction['node_set'], dtype=torch.long)
        edge_index = torch.tensor(instruction['edge_index'], dtype=torch.long).reshape(2, -1)
        if node_list.numel() == 0 or (instruction['task'] == 'lp' and node_list.numel() < 2):
            raise ValueError(f'Invalid target nodes in instruction {idx}')
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= node_list.numel()):
            raise ValueError(f'Non-local edge indices in instruction {idx}')
        if instruction['task'] == 'lp' and self.args.remove_target_edge:
            target_edge = ((edge_index[0] == 0) & (edge_index[1] == 1)) | ((edge_index[0] == 1) & (edge_index[1] == 0))
            edge_index = edge_index[:, ~target_edge]
        return Data(x=self.name2data[instruction['data']].x[node_list].float(),
                    edge_index=edge_index, lp=instruction['task'] == 'lp')

    def collate_fn(self, batch):
        def pad(key, value, left=False):
            values = [entry[key].squeeze(0) for entry in batch]
            if left:
                values = [v.flip(0) for v in values]
            result = pad_sequence(values, batch_first=True, padding_value=value)
            return result.flip(1) if left else result
        left = self.mode != 'train'
        collated = dict(input_ids=pad('input_ids', self.tokenizer.pad_token_id, left),
                    target_ids=pad('target_ids', IGNORE_TOKEN_ID if not left else self.tokenizer.pad_token_id),
                    attn_mask=pad('attention_mask', False, left),
                    is_node=pad('is_node', False, left),
                    sample_index=torch.tensor([entry['sample_index'] for entry in batch], dtype=torch.long))
        if not self.text_only:
            collated['graph'] = Batch.from_data_list([entry['graph'] for entry in batch])
            if self.token_sources is not None:
                collated['source_graph'] = Batch.from_data_list([entry['source_graph'] for entry in batch])
        return collated
