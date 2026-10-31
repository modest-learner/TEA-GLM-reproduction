"""Original shared helpers, checkpoint loading and standalone NC evaluation.

Run this file directly to score saved generations without importing ML packages.
"""
import argparse
import hashlib
import inspect
import json
import os
import platform
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def seed_everything(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_trainable_params(first_model, model):
    trainable_params = 0
    all_param = 0

    for _, param in (first_model.named_parameters() if first_model is not None else []):
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_optimizer(optim):
    import torch
    # Bind the optimizer
    if optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def load_checkpoint(path, *, graph_data=False):
    import torch
    # PyG Data objects in the author's graph files require pickle deserialization.
    # Only use graph_data=True for trusted downloaded graph files.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = not graph_data
    return torch.load(path, **kwargs)


def resolve_checkpoint(args, kind):
    root = Path(args.save_dir)
    if kind == 'gnn':
        name = args.pretrain_gnn
        if name:
            candidate = Path(name)
            candidate = candidate if candidate.is_file() else root / 'gnn' / name
        else:
            source = args.dataset.removesuffix('_LP')
            candidate = root / 'gnn' / f'GraphSAGE_{source}_1000_tp.pth'
    else:
        explicit = getattr(args, 'projector_checkpoint' if kind == 'first_model' else 'model_checkpoint', '')
        if explicit:
            candidate = Path(explicit)
        else:
            marker = 'fm' if kind == 'first_model' else 'm'
            candidate = root / kind / f'{args.prefix}_{marker}_{args.dataset}_epoch{args.best_epoch}_end.pth'
            if not candidate.is_file() and kind == 'first_model':
                source = args.dataset.removesuffix('_LP')
                domain = 'citation' if source in ('arxiv', 'cora', 'pubmed') else 'ecom'
                candidate = root / kind / f'TEA-GLM_{domain}.pth'
    if not candidate.is_file():
        raise FileNotFoundError(f'{kind} checkpoint not found: {candidate}')
    return candidate


def output_decode(eval_output, eval_label, tokenizer):
    eval_decode_output = []
    eval_decode_label = []
    assert len(eval_output) == len(eval_label)
    for i in range(len(eval_output)):
        batch_output = eval_output[i]
        label_output = eval_label[i]
        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=True))
        eval_decode_label.extend(tokenizer.batch_decode(label_output, skip_special_tokens=True))
    assert len(eval_decode_label) == len(eval_decode_output)

    return eval_decode_output, eval_decode_label


def get_edge_num(x):
    return len(x['edge_index'][0])


def normalize_label(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip(' \t\n\r\"\'`*.,:;!?[](){}')


EVALUATION_VERSION = 'nc-unified-v1'


class LabelResolver:
    """Extract a prediction WITHOUT consulting the ground-truth answer.

    strict: the previous whole-answer matcher, retained for audit.
    unique: accept exactly one candidate mentioned with word boundaries.
    Conservative negation/uncertainty guards are not a semantic language model;
    rejected answers are retained for inspection, never guessed from the target.
    """
    def __init__(self, classes, policy='unique'):
        if policy not in ('strict', 'unique'):
            raise ValueError('Unknown evaluation policy: ' + str(policy))
        self.policy = policy
        self.classes = list(classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('Classes must be nonempty and unique')
        self.exact = {}
        aliases = defaultdict(set)
        for label in self.classes:
            key = normalize_label(label)
            if not key or key in self.exact:
                raise ValueError('Class names collide after normalization')
            self.exact[key] = label
            code = re.fullmatch(r'(cs\.[a-z]{2}),\s*(.+)', key)
            if code:
                aliases[code[1]].add(label)
                aliases[code[2]].add(label)
            # The released Cora instructions omit the leading AI hierarchy on
            # machine-learning categories, while their annotations include it.
            prefix = 'artificial intelligence, machine learning, '
            if key.startswith(prefix):
                aliases[key.removeprefix('artificial intelligence, ')].add(label)
        self.aliases = {alias: next(iter(labels)) for alias, labels in aliases.items() if len(labels) == 1}
        # Full names and the already documented arXiv/Cora aliases only.
        self.patterns = [(re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)'), label)
                         for name, label in {**self.aliases, **self.exact}.items()]

    def resolve_strict(self, response):
        key = normalize_label(response)
        # Accept a simple answer prefix and surrounding formatting. Do not guess
        # based on substrings/fuzzy similarity or select among multiple labels.
        key = re.sub(r'^(?:the\s+)?(?:most\s+likely\s+)?(?:answer|category|label|class|subcategory)\s*(?::|is\b)\s*:?\s*', '', key)
        key = normalize_label(key)
        return self.exact.get(key, self.aliases.get(key))

    def resolve_detail(self, response):
        if not isinstance(response, str):
            raise ValueError('An LLM response must be a string')
        strict = self.resolve_strict(response)
        if strict is not None:
            return dict(label=strict, reason='exact_or_alias', candidates=[strict])
        if self.policy == 'strict':
            return dict(label=None, reason='strict_unmatched', candidates=[])
        text = normalize_label(response)
        spans = [(match.start(), match.end(), label)
                 for pattern, label in self.patterns for match in pattern.finditer(text)]
        # A complete, more specific name wins over a name contained inside it;
        # separate mentions of different categories still remain ambiguous.
        spans = [(a, b, label) for a, b, label in spans if not any(
            c <= a and b <= d and (c < a or b < d) for c, d, _ in spans)]
        candidates = sorted({label for _, _, label in spans})
        if len(candidates) != 1:
            return dict(label=None, reason='multiple_candidates' if candidates else 'no_candidate',
                        candidates=candidates)
        # Reject uncertain or negative prose conservatively. This deliberately
        # sacrifices some recall instead of crediting "not Type 1 diabetes".
        negative = r"\b(?:not|no|never|neither|nor|cannot|can't|won't|without)\b|\b\w+n['\u2019]t\b"
        uncertain = (r'\b(?:unclear|unknown|unsure|uncertain|insufficient|unable|'
                     r'undetermined|ambiguous|whether|might|maybe|possibly|perhaps)\b|'
                     r'\b(?:could|may)\s+be\b|\b(?:rather than|instead of)\b')
        if re.search(negative, text):
            reason = 'negation_or_exclusion'
        elif re.search(uncertain, text):
            reason = 'uncertain_answer'
        else:
            return dict(label=candidates[0], reason='unique_candidate', candidates=candidates)
        return dict(label=None, reason=reason, candidates=candidates)

    def resolve(self, response):
        return self.resolve_detail(response)['label']


def _classification_counts(predicted, targets, classes):
    support, selected = Counter(targets), Counter(p for p in predicted if p is not None)
    correct = Counter(target for target, pred in zip(targets, predicted) if target == pred)
    per_class = {}
    for label in classes:
        tp, fp, fn = correct[label], selected[label] - correct[label], support[label] - correct[label]
        per_class[label] = dict(support=support[label], true_positive=tp, false_positive=fp, false_negative=fn,
                                precision=tp / (tp + fp) if tp + fp else 0.,
                                recall=tp / (tp + fn) if tp + fn else 0.,
                                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return dict(accuracy=sum(correct.values()) / len(targets) if targets else None,
                macro_f1=sum(row['f1'] for row in per_class.values()) / len(classes) if targets else None,
                per_class=per_class)


def node_classification_metrics(predictions, references, classes, policy='unique'):
    if not predictions or len(predictions) != len(references):
        raise ValueError('Predictions and references must have the same nonzero length')
    resolver = LabelResolver(classes, policy=policy)
    targets = []
    for reference in references:
        target = resolver.exact.get(normalize_label(reference))
        if target is None:
            raise ValueError(f'Unknown reference label: {reference}')
        targets.append(target)
    details = [resolver.resolve_detail(response) for response in predictions]
    predicted = [d['label'] for d in details]
    full = _classification_counts(predicted, targets, resolver.classes)
    legal = [(p, y) for p, y in zip(predicted, targets) if p is not None]
    legal_metrics = _classification_counts([p for p, _ in legal], [y for _, y in legal], resolver.classes)
    strict = [resolver.resolve_strict(response) for response in predictions]
    strict_metrics = _classification_counts(strict, targets, resolver.classes)
    strict_metrics.pop('per_class')
    strict_metrics['valid_answer_rate'] = sum(p is not None for p in strict) / len(targets)
    invalid = sum(label is None for label in predicted)
    report = dict(task='node_classification', num_samples=len(targets), num_classes=len(resolver.classes),
                  **full, macro_f1_all=full['macro_f1'], macro_f1_legal=legal_metrics['macro_f1'],
                  num_valid=len(legal), legal_per_class=legal_metrics['per_class'],
                  valid_answer_rate=1 - invalid / len(targets), invalid_answers=invalid,
                  evaluation_version=EVALUATION_VERSION, parser_policy=policy,
                  evaluation_policy='Accuracy and macro_f1/macro_f1_all: all examples, invalid=error/FN; '
                                    'macro_f1_legal: valid examples only; all dataset classes; zero_division=0; '
                                    'no legal examples => macro_f1_legal=null',
                  label_mapping='normalized exact/alias' if policy == 'strict' else
                                'exact/alias then unique boundary-matched candidate; conservative negation/uncertainty guards',
                  resolution_counts=dict(Counter(d['reason'] for d in details)), strict_metrics=strict_metrics)
    return report, predicted


def load_nc_references(instruction_path, max_samples=None):
    """Read the same instruction order as training, retaining the full ontology."""
    if max_samples is not None and max_samples < 1:
        raise ValueError('max_samples must be positive')
    data = json.loads(Path(instruction_path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get('output'), dict):
        keys = list(data.get('prompt', data['output']))
        if any(set(data[name]) != set(keys) for name in ('output', 'task')):
            raise ValueError('Instruction column indices do not match')
        rows = [dict(output=data['output'][key], task=data['task'][key]) for key in keys]
    elif isinstance(data, dict) and isinstance(data.get('output'), list):
        if len(data['output']) != len(data.get('task', [])):
            raise ValueError('Instruction columns have different lengths')
        rows = [dict(output=label, task=task) for label, task in zip(data['output'], data['task'])]
    else:
        raise ValueError('Expected a JSON instruction list or column-oriented instruction object')
    if not rows or any(row.get('task') != 'nc' for row in rows):
        raise ValueError('Instructions must contain node classification examples only')
    references = [row['output'] for row in rows]
    classes = sorted(set(references))
    return references[:max_samples] if max_samples is not None else references, classes


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


class FeatureStore:
    """Batch-durable CPU arrays; BF16 is stored losslessly as its int16 bits."""
    def __init__(self, folder, count, dataset, method):
        import numpy as np
        self.folder = Path(folder) / 'feature_store'
        self.folder.mkdir()
        self.count = count
        self.maps = {}
        self.manifest = dict(schema_version=1, dataset=dataset, method=method,
                             num_expected=count, num_saved=0, complete=False, arrays={},
                             X_definition='Released graph.x for the target node; LP uses the mean of its two targets',
                             H_GNN_definition='Exact pooled GNN output used in this inference forward; absent for vicuna_text',
                             T_graph_definition='Exact projected graph tokens injected into the LLM; absent for vicuna_text',
                             baseline_graph_features='not_applicable' if method == 'vicuna_text' else 'present')
        self.written = np.lib.format.open_memmap(self.folder / 'written.npy', mode='w+',
                                                dtype=np.bool_, shape=(count,))
        self.written[:] = False
        self.written.flush()
        atomic_json(self.folder / 'manifest.json', self.manifest)

    def write(self, indices, tensors):
        import numpy as np
        import torch
        new_array = False
        for name, tensor in tensors.items():
            tensor = tensor.detach().cpu().contiguous()
            array = tensor.view(torch.int16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            if name not in self.maps:
                new_array = True
                shape = (self.count,) + tuple(array.shape[1:])
                self.maps[name] = np.lib.format.open_memmap(self.folder / f'{name}.npy', mode='w+',
                                                           dtype=array.dtype, shape=shape)
                self.manifest['arrays'][name] = dict(file=f'{name}.npy', shape=list(shape),
                                                     torch_dtype=str(tensor.dtype).split('.')[-1],
                                                     numpy_dtype=str(array.dtype))
            spec = self.manifest['arrays'][name]
            if list(array.shape[1:]) != spec['shape'][1:] or str(tensor.dtype).split('.')[-1] != spec['torch_dtype']:
                raise ValueError(f'Feature shape/dtype changed during inference: {name}')
            self.maps[name][indices] = array
            self.maps[name].flush()
        if new_array:
            atomic_json(self.folder / 'manifest.json', self.manifest)

    def commit(self, indices):
        self.written[indices] = True
        self.written.flush()
        self.manifest['num_saved'] = int(self.written.sum())
        atomic_json(self.folder / 'manifest.json', self.manifest)

    def close(self, complete):
        self.manifest['complete'] = bool(complete)
        atomic_json(self.folder / 'manifest.json', self.manifest)
        self.maps.clear()
        del self.written


def recover_features(run_dir):
    """Finalize full or interrupted feature stores on CPU, without loading an LLM."""
    import numpy as np
    import torch
    run_dir = Path(run_dir)
    # A stale partial feature_store must never replace a complete final tensor
    # package (e.g. after copying files from the server at different times).
    existing = list(run_dir.glob('features_*.pt'))
    if existing:
        raise FileExistsError(f'Final feature file already exists: {existing[0]}; '
                              'recovery is only for runs without a final feature file')
    folder = run_dir / 'feature_store'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    written = np.load(folder / 'written.npy', mmap_mode='r')
    rows = np.flatnonzero(written)
    # Even after a hard stop, only rows committed after the JSONL flush are used.
    result = dict(X=None, H_GNN=None, T_graph=None, node_ids=None,
                  sample_indices=torch.from_numpy(rows.copy()), metadata=dict(manifest))
    result['metadata'].update(num_saved=len(rows), complete=len(rows) == manifest['num_expected'])
    for name, spec in manifest['arrays'].items():
        array = np.load(folder / spec['file'], mmap_mode='r+')
        tensor = torch.from_numpy(array)
        if spec['torch_dtype'] == 'bfloat16':
            tensor = tensor.view(torch.bfloat16)
        result[name] = tensor if len(rows) == manifest['num_expected'] else tensor[result['sample_indices']]
    labels = {}
    predictions_path = run_dir / f"predictions_{manifest['dataset']}.jsonl"
    with predictions_path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted last append is not a committed feature row.
                continue
            labels[record['sample_index']] = record['true_label']
    if any(int(i) not in labels for i in rows):
        raise ValueError('Committed feature rows are missing their prediction logs')
    result['true_labels'] = [labels[int(i)] for i in rows]
    target = run_dir / f"features_{manifest['dataset']}.pt"
    temporary = target.with_name(target.name + '.tmp')
    torch.save(result, temporary)
    os.replace(temporary, target)
    return target


class InferenceRecorder:
    """Preserve per-node outputs and forward features before the next batch."""
    def __init__(self, args, dataset, decoding_settings):
        self.args, self.dataset = args, dataset
        self.parent = Path(args.results_dir) / args.test_dataset
        self.folder = self.parent / args.prefix
        if self.folder.exists():
            raise FileExistsError(f'Run directory already exists: {self.folder}. '
                                  'Use a new --prefix or --results_dir to preserve earlier logs.')
        self.folder.mkdir(parents=True)
        self.references = dataset.instructions['output'].tolist()
        self.predictions = [None] * len(dataset)
        self.seen = set()
        self.nc = set(dataset.instructions['task']) == {'nc'}
        self.resolver = LabelResolver(dataset.class_labels if self.nc else sorted(set(self.references)),
                                      policy=getattr(args, 'evaluation_policy', 'unique') if self.nc else 'strict')
        self.settings = dict(decoding_settings)
        self.settings.update(seed=args.run_seed, backbone=args.backbone, source_dataset=args.dataset,
                             test_dataset=args.test_dataset, method=args.method, batch_size=args.batch_size,
                             max_text_length=args.max_text_length)
        versions = {}
        for package in ('torch', 'transformers', 'accelerate', 'torch-geometric', 'numpy', 'pandas', 'sentencepiece'):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        root = Path(__file__).resolve().parents[1]
        source_files = [root / 'config.py', root / 'train_glm.py']
        source_files += sorted((root / 'model').glob('*.py')) + sorted((root / 'utils').glob('*.py'))
        checkpoint = getattr(args, 'loaded_projector_checkpoint', '')
        self.run_config = dict(schema_version=1, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
            num_expected=len(dataset), num_completed=0, args=vars(args).copy(), generation=self.settings,
            python=sys.version, platform=platform.platform(), package_versions=versions,
            command=sys.argv, classes=dataset.class_labels,
            instruction_file=file_fingerprint(dataset.instruction_path),
            projector_checkpoint=file_fingerprint(checkpoint) if checkpoint else None,
            source_files={str(p.relative_to(root)): file_fingerprint(p)['sha256'] for p in source_files},
            evaluation_version=EVALUATION_VERSION, parser_policy=self.resolver.policy,
            text_baseline_definition='Same instruction preprocessing; remove <Node N> placeholders only; no GNN/projector',
            feature_note='X is the released node feature matrix, not LLM hidden states; H_GNN/T_graph do not exist for vicuna_text')
        atomic_json(self.folder / 'run_config.json', self.run_config)
        atomic_json(self.folder / 'generation_config.json', self.settings)
        self.log = (self.folder / f'predictions_{args.test_dataset}.jsonl').open('x', encoding='utf-8')
        self.features = FeatureStore(self.folder, len(dataset), args.test_dataset, args.method) if args.save_features else None

    def write_batch(self, indices, outputs, raw_outputs, output_tokens, input_tokens, graph_features=None):
        import torch
        if len({len(v) for v in (indices, outputs, raw_outputs, output_tokens, input_tokens)}) != 1:
            raise ValueError('Prediction batch columns have different lengths')
        if len(set(indices)) != len(indices) or any(i in self.seen or not 0 <= i < len(self.dataset) for i in indices):
            raise ValueError('Duplicate or invalid sample indices in inference output')
        records, xs, node_ids = [], [], []
        for i, output, raw, generated, inputs in zip(indices, outputs, raw_outputs, output_tokens, input_tokens):
            row = self.dataset.instructions.iloc[i]
            node_id = int(row['node_set'][0])
            resolution = self.resolver.resolve_detail(output)
            parsed = resolution['label']
            target = self.resolver.exact[normalize_label(row['output'])]
            records.append(dict(sample_index=i, node_id=node_id, true_label=row['output'],
                llm_raw_output=output, parsed_pred=parsed, is_correct=parsed == target,
                valid_answer=parsed is not None, llm_output_with_special_tokens=raw,
                parse_reason=resolution['reason'], mentioned_candidates=resolution['candidates'],
                evaluation_version=EVALUATION_VERSION, parser_policy=self.resolver.policy,
                generated_token_ids=generated, input_token_ids=inputs, prompt=self.dataset.prompt_text(i),
                target_node_ids=[int(n) for n in row['node_set'][:2 if row['task'] == 'lp' else 1]],
                dataset=self.args.test_dataset, source_dataset=self.args.dataset, method=self.args.method,
                seed=self.args.run_seed, feature_row=i if self.features else None))
            if self.features:
                graph = self.dataset.name2data[row['data']]
                selected = row['node_set'][:2 if row['task'] == 'lp' else 1]
                xs.append(graph.x[selected].float().mean(0).cpu())
                node_ids.append(node_id)
        if self.features:
            tensors = dict(X=torch.stack(xs), node_ids=torch.tensor(node_ids, dtype=torch.long))
            tensors.update(graph_features or {})
            self.features.write(indices, tensors)
        for record in records:
            self.log.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.log.flush()
        os.fsync(self.log.fileno())
        if self.features:
            self.features.commit(indices)
        for i, output in zip(indices, outputs):
            self.predictions[i] = output
            self.seen.add(i)
        atomic_json(self.folder / 'progress.json', dict(num_completed=len(self.seen), num_expected=len(self.dataset)))
        self.run_config['num_completed'] = len(self.seen)
        atomic_json(self.folder / 'run_config.json', self.run_config)
        if self.args.verbose:
            from tqdm import tqdm
            for record in records:
                tqdm.write(json.dumps({key: record[key] for key in
                    ('sample_index', 'node_id', 'true_label', 'llm_raw_output', 'parsed_pred', 'is_correct')}, ensure_ascii=False))

    def finish(self, success, error=None):
        self.log.close()
        if success and len(self.seen) != len(self.dataset):
            success = False
            error = 'Evaluation count differs from instruction count'
        self.run_config.update(status='finalizing' if success else 'failed', num_completed=len(self.seen),
                               error=error)
        if not success:
            self.run_config['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.folder / 'run_config.json', self.run_config)
        if error:
            (self.folder / 'error.txt').write_text(error, encoding='utf-8')
        if self.features:
            self.features.close(success)
            self.features = None
            recover_features(self.folder)
        if not success:
            return None
        for suffix, values in [('results', self.predictions), ('labels', self.references)]:
            atomic_json(self.parent / f'{self.args.prefix}_model_{suffix}.txt', values)
        atomic_json(self.parent / f'{self.args.prefix}_generation_config.json', self.settings)
        metrics = None
        if self.nc:
            metrics, mapped = node_classification_metrics(self.predictions, self.references, self.dataset.class_labels,
                                                           policy=self.resolver.policy)
            atomic_json(self.folder / 'metrics.json', metrics)
            atomic_json(self.parent / f'{self.args.prefix}_metrics.json', metrics)
            atomic_json(self.parent / f'{self.args.prefix}_predicted_labels.json', mapped)
        self.run_config.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.folder / 'run_config.json', self.run_config)
        return metrics


def read_predictions(path, instruction_path, references):
    """Accept legacy text arrays or validate/reorder the per-node JSONL log."""
    path = Path(path)
    if path.suffix.lower() != '.jsonl':
        predictions = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(predictions, list) or not all(isinstance(p, str) for p in predictions):
            raise ValueError('Predictions must be a JSON array of strings')
        return predictions
    with path.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != len(references):
        raise ValueError('JSONL sample count differs from instructions; do not score a partial run as a full test')
    ordered = {}
    for record in records:
        index = record.get('sample_index')
        if type(index) is not int or index in ordered or not 0 <= index < len(references):
            raise ValueError('JSONL contains duplicate or invalid sample indices')
        ordered[index] = record
    instructions = json.loads(Path(instruction_path).read_text(encoding='utf-8'))
    if isinstance(instructions, list):
        node_sets = [row['node_set'] for row in instructions]
    elif isinstance(instructions['node_set'], dict):
        keys = list(instructions.get('prompt', instructions['node_set']))
        node_sets = [instructions['node_set'][key] for key in keys]
    else:
        node_sets = instructions['node_set']
    predictions = []
    for i, reference in enumerate(references):
        record = ordered[i]
        if normalize_label(record.get('true_label', '')) != normalize_label(reference):
            raise ValueError(f'JSONL reference label does not match instruction {i}')
        if record.get('node_id') != int(node_sets[i][0]):
            raise ValueError(f'JSONL node ID does not match instruction {i}')
        if not isinstance(record.get('llm_raw_output'), str):
            raise ValueError(f'JSONL raw output is missing at instruction {i}')
        predictions.append(record['llm_raw_output'])
    return predictions


def evaluate_saved_predictions(prediction_path, instruction_path, output, *, policy='unique',
                               max_samples=None, parsed_output=None):
    """CPU-only scoring; preserve raw run files and record input fingerprints."""
    output = Path(output)
    parsed_output = Path(parsed_output) if parsed_output else None
    if parsed_output is not None and output.resolve() == parsed_output.resolve():
        raise ValueError('Metric JSON and per-node JSONL must have different paths')
    for path in (output, parsed_output):
        if path is not None and path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; choose a new output path')
    references, classes = load_nc_references(instruction_path, max_samples)
    predictions = read_predictions(prediction_path, instruction_path, references)
    metrics, mapped = node_classification_metrics(predictions, references, classes, policy=policy)
    metrics['inputs'] = dict(predictions=file_fingerprint(prediction_path),
                            instructions=file_fingerprint(instruction_path))
    metrics['evaluator_sha256'] = file_fingerprint(__file__)['sha256']
    if parsed_output:
        data = json.loads(Path(instruction_path).read_text(encoding='utf-8'))
        if isinstance(data, list):
            nodes = [row['node_set'][0] for row in data]
        elif isinstance(data['node_set'], dict):
            nodes = [data['node_set'][key][0] for key in data.get('prompt', data['node_set'])]
        else:
            nodes = [values[0] for values in data['node_set']]
        resolver = LabelResolver(classes, policy)
        parsed_output.parent.mkdir(parents=True, exist_ok=True)
        with parsed_output.open('x', encoding='utf-8') as handle:
            for i, (raw, target, pred) in enumerate(zip(predictions, references, mapped)):
                detail = resolver.resolve_detail(raw)
                handle.write(json.dumps(dict(sample_index=i, node_id=int(nodes[i]), true_label=target,
                    llm_raw_output=raw, parsed_pred=pred, is_correct=pred == target,
                    valid_answer=pred is not None, parse_reason=detail['reason'],
                    mentioned_candidates=detail['candidates'], parser_policy=policy,
                    evaluation_version=EVALUATION_VERSION), ensure_ascii=False) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, metrics)
    return metrics


def _evaluation_main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--predictions', type=Path, help='The *_model_results.txt JSON array or predictions_*.jsonl')
    parser.add_argument('--instructions', type=Path, help='The matching *_dataset_test.json')
    parser.add_argument('--max_samples', type=int, default=None, help='Use the same value as the generation command')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--policy', choices=['unique', 'strict'], default='unique')
    parser.add_argument('--parsed-output', type=Path, help='New per-node audit JSONL; original logs are not changed')
    parser.add_argument('--results-root', type=Path, help='Score all complete legacy output arrays under this experiment')
    parser.add_argument('--instruction-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'instruction')
    parser.add_argument('--datasets', nargs='+', default=['pubmed', 'history', 'children', 'sports'])
    parser.add_argument('--output-dir', type=Path, help='New directory for batch evaluation; refuses overwrite')
    parser.add_argument('--recover-features', type=Path, help='Finalize a retained feature_store in this run directory on CPU')
    args = parser.parse_args()
    if args.recover_features:
        print(f'Saved {recover_features(args.recover_features)}')
        return
    if args.results_root:
        if not args.output_dir or args.output_dir.exists():
            parser.error('--output-dir must name a new directory for batch evaluation')
        if any((args.predictions, args.instructions, args.output, args.parsed_output, args.max_samples)):
            parser.error('Do not combine batch evaluation with single-file/subset options')
        summary = []
        for dataset in args.datasets:
            instruction = args.instruction_dir / dataset / f'{dataset}_dataset_test.json'
            for prediction in sorted((args.results_root / dataset).glob('*_model_results.txt')):
                prefix = prediction.name.removesuffix('_model_results.txt')
                destination = args.output_dir / dataset / prefix
                try:
                    references, _ = load_nc_references(instruction)
                    labels_path = prediction.with_name(prefix + '_model_labels.txt')
                    if json.loads(labels_path.read_text(encoding='utf-8')) != references:
                        raise ValueError('Saved labels do not match the complete instruction order')
                    config_path = prediction.parent / prefix / 'run_config.json'
                    if config_path.exists():
                        config = json.loads(config_path.read_text(encoding='utf-8'))
                        expected_hash = config.get('instruction_file', {}).get('sha256')
                        if expected_hash and file_fingerprint(instruction)['sha256'] != expected_hash:
                            raise ValueError('Instruction hash differs from the generation run')
                    metrics = evaluate_saved_predictions(prediction, instruction, destination / 'metrics.json',
                        policy=args.policy, parsed_output=destination / 'predictions_evaluated.jsonl')
                    row = dict(dataset=dataset, run=prefix, **{k: metrics[k] for k in
                        ('num_samples', 'accuracy', 'macro_f1_all', 'macro_f1_legal', 'valid_answer_rate', 'strict_metrics')})
                    summary.append(row)
                    print(json.dumps(row, ensure_ascii=False))
                except (ValueError, FileNotFoundError) as exc:
                    parser.error(f'{dataset}/{prefix}: {exc}')
        if not summary:
            parser.error('No complete *_model_results.txt outputs found')
        atomic_json(args.output_dir / 'summary.json', dict(evaluation_version=EVALUATION_VERSION,
            parser_policy=args.policy, evaluator_sha256=file_fingerprint(__file__)['sha256'], runs=summary,
            note='Scored complete raw output/label arrays; this does not certify completeness of separate logs or feature stores'))
        return
    if not all((args.predictions, args.instructions, args.output)):
        parser.error('--predictions, --instructions and --output are required for scoring')
    try:
        metrics = evaluate_saved_predictions(args.predictions, args.instructions, args.output,
            policy=args.policy, max_samples=args.max_samples, parsed_output=args.parsed_output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Accuracy={metrics['accuracy']:.6f}, Macro F1={metrics['macro_f1']:.6f}, "
          f"Macro F1 (legal)={metrics['macro_f1_legal']}, "
          f"Valid answers={metrics['valid_answer_rate']:.6f}; saved {args.output}")


if __name__ == '__main__':
    _evaluation_main()
