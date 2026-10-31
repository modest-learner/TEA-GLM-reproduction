"""CPU-only diagnostics for retained TEA-GLM inference artifacts (Python 3.9).

Example:
    python scripts/analyze_results.py --results results/comparison_20261029_083107

只读取原结果；写入新的 analysis_* 目录。默认分析 History / PubMed。
复用 utils/utils.py 的正式标签解析口径；没有 Vicuna 时只做单模型诊断。
NumPy 为必需依赖；原始 .pt 图需要项目已有的 PyTorch/PyG 依赖。
Matplotlib 可选，有则输出 PNG/PDF 图。绝不重新加载 LLM 或修改模型。
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import inspect
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Load the standalone evaluator directly: utils/__init__.py also imports the
# training stack, which is unnecessary for CPU analysis of saved predictions.
_spec = importlib.util.spec_from_file_location('teaglm_saved_metrics', ROOT / 'utils' / 'utils.py')
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
node_classification_metrics = _metrics.node_classification_metrics
normalize_label = _metrics.normalize_label

ECOM = {'history': 'book_history', 'children': 'book_children',
        'computer': 'computer', 'sports': 'sports', 'photo': 'photo'}
BINS = ['[0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1]']
GLOSSARY = {
    'Accuracy / 准确率': '全部测试样本中答对的比例。大类样本多，对它的影响也更大。非法回答仍计错。',
    'Macro F1 / 宏平均 F1': '先计算每个类别的 F1，再对类别等权平均。小类和大类权重相同，适合观察长尾类别表现。',
    'Precision / 精确率': '预测成该类别的样本中，有多少确实属于它。低值常提示这个类别被过度预测。',
    'Recall / 召回率': '真实属于该类别的样本中，有多少被找回来。低值意味着该类样本经常被分到别处。',
    'F1': '精确率与召回率的调和平均。两者任何一个很低，F1 都会降低。',
    'Support / 类别样本数': '该类别在当前测试集里的真实数量。只有几条样本的类别，指标波动会很大。',
    'Imbalance ratio / 类别不平衡比': '最大类数量除以最小类数量，越大说明数量越不均衡。极小类别会显著放大此值，要结合每类数量看。',
    'Normalized label entropy / 归一化类别熵': '类别分布的熵除以 log(类别数)。接近 1 表示各类接近均衡，接近 0 表示样本集中在少数类；它不是预测准确率。',
    'Prediction inflation / 预测数量比': '预测为某类的数量除以该类真实数量。大于 1 表示过度预测，小于 1 表示预测不足；不等同于准确率。',
    '合法回答率': '按现有固定规则能解析到候选类别的比例。附带解释等文本可能被判为非法，需查看原始回答，不能自动改标签。',
    'Majority baseline / 多数类参照': '总是猜测试集中最大类时的 Accuracy，是利用真实标签计算的事后参照，不是可部署的零样本模型。',
    'Edge homophily / 边同质率': '连接同类节点的边占比。本脚本全图统计去自环、去重复的有向边；双向图含两个方向。',
    'Node homophily / 节点同质率': '先计算每个节点的同类入邻居比例，再对有邻居的节点等权平均。与按边加权的边同质率不同。',
    'Adjusted homophily / 校正同质率': '扣除按边端点类别占比随机连接时的同类概率，再归一化；接近 0 表示接近该随机参照。',
    '局部同质率': '使用本条指令子图中真正指向中心节点的边计算，沿用消息传递方向及重复边权重，排除中心自环；未知邻居标签不会当作异类。这是 1 跳指标，不能完整代表两层 GNN 的全部影响。',
    'Head / Medium / Tail': '按测试集类别规模从大到小划分，头部取前 ceil(20%) 类、尾部取后 floor(50%) 类，其余为中部；不足 5 类时不强行划分。',
    '有效秩 / effective rank': '中心化特征奇异值归一化后的熵的指数，反映信息分布在多少个数值方向上。这里是中心化 RankMe 变体，不是类别数量，也不是准确率。',
    '归一化余弦边能量': '固定测试诱导边上，单位化节点向量的平均 1-cosine，越低表示相邻表示更接近。分同类边与异类边报告；不是完整图能量，也不能单独证明有害过平滑。',
    '类内减类间余弦差': '每类内部的平均余弦相似度等权平均，减去不同类别对的平均余弦。较高说明按这个几何度量更容易区分类别，不保证 LLM 能正确利用。',
    'Delta / 差值': 'TEA-GLM 减 Vicuna。正值表示 TEA-GLM 更高；差 0.01 表示 1 个百分点。',
    '配对 bootstrap 区间': '对同一批节点成对重采样得到的 95% 区间；按真实类别分层、固定类别数量。图节点并非独立，区间仅为描述性参考，不能当作严格因果证据。',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def save_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or (list(rows[0]) if rows else [])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_json(v) for k, v in row.items()})


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def instruction_rows(path):
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj['prompt'], dict):
        keys = list(obj['prompt'])
        require(all(set(col) == set(keys) for col in obj.values()), '指令列索引不一致')
        return [{k: col[i] for k, col in obj.items()} for i in keys]
    require(len({len(col) for col in obj.values()}) == 1, '指令列长度不一致')
    return [{k: col[i] for k, col in obj.items()} for i in range(len(obj['prompt']))]


def results_root(path, datasets):
    path = Path(path).resolve()
    if any((path / d).is_dir() for d in datasets):
        return path
    choices = sorted(p for p in path.glob('*') if p.is_dir()
                     and any((p / d).is_dir() for d in datasets))
    require(len(choices) == 1, '请用 --results 指定实际 comparison_* 目录；不能自动合并不同实验。')
    return choices[0]


def groups_for_classes(classes, references):
    support = Counter(references)
    order = sorted(classes, key=lambda c: (-support[c], c))
    if len(order) < 5:
        return {c: '逐类分析' for c in order}
    head, tail = math.ceil(len(order) * .2), math.floor(len(order) * .5)
    return {c: 'Head' if i < head else 'Tail' if i >= len(order) - tail else 'Medium'
            for i, c in enumerate(order)}


def class_balance(counts):
    counts = np.asarray(list(counts), dtype=float)
    require(len(counts) > 0 and np.all(counts > 0), '类别统计必须具有正样本数')
    p = counts / counts.sum()
    return dict(num_classes=len(counts), largest_class=int(counts.max()), smallest_class=int(counts.min()),
                imbalance_ratio=float(counts.max() / counts.min()), majority_fraction=float(p.max()),
                normalized_label_entropy=float(-np.sum(p * np.log(p)) / np.log(len(counts))) if len(counts) > 1 else 0.)


def load_run(folder, dataset, instructions, instruction_path, policy='unique'):
    config = read_json(folder / 'run_config.json')
    require(config['status'] == 'complete', f'{folder.name} 未完整结束，不能作为全量测试')
    expected_hash = config.get('instruction_file', {}).get('sha256')
    require(not expected_hash or fingerprint(instruction_path) == expected_hash, '本地指令与运行时 SHA256 不同')
    args = config['args']
    require(args.get('max_samples') is None, '当前分析要求完整测试集，拒绝 max_samples 子集')
    require(args['test_dataset'] == dataset, '运行配置的目标数据集不符')
    n = len(instructions)
    require(config['num_expected'] == config['num_completed'] == n, '完成数量与完整指令集不符')
    path = folder / f'predictions_{dataset}.jsonl'
    records = [None] * n
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            i = row.get('sample_index')
            require(type(i) is int and 0 <= i < n and records[i] is None,
                    f'{path.name}:{line_number} sample_index 重复或越界')
            ins = instructions[i]
            require(row['node_id'] == int(ins['node_set'][0]), f'样本 {i} node_id 错位')
            require(row['true_label'] == ins['output'], f'样本 {i} true_label 错位')
            require(row.get('dataset') == dataset and row.get('method') == args['method'], '预测方法/数据集字段不一致')
            require(row.get('seed') == config['generation']['seed'], '预测种子不一致')
            require(isinstance(row.get('llm_raw_output'), str), '缺失原始生成文本')
            records[i] = row
    require(all(r is not None for r in records), '预测文件缺失测试节点')
    require(len({r['node_id'] for r in records}) == n, '测试目标节点重复，需单独定义多次观测协议')
    references = [r['true_label'] for r in records]
    classes = sorted(set(references))
    require(sorted(config['classes']) == classes, '类别列表与完整指令不一致')
    saved = read_json(folder / 'metrics.json')
    # First verify the original files under the rule that created them. Then
    # re-score both methods under the same requested rule without rewriting logs.
    original_policy = saved.get('parser_policy', 'strict')
    original_metrics, original_parsed = node_classification_metrics(
        [r['llm_raw_output'] for r in records], references, classes, policy=original_policy)
    for i, (row, pred) in enumerate(zip(records, original_parsed)):
        require(row.get('parsed_pred') == pred and row.get('is_correct') == (pred == references[i])
                and row.get('valid_answer') == (pred is not None),
                f'样本 {i} 重算解析结果与保存字段不一致；先确认评测版本')
    for key in ('accuracy', 'macro_f1', 'valid_answer_rate', 'num_samples', 'invalid_answers'):
        require(math.isclose(saved[key], original_metrics[key], abs_tol=1e-12), f'保存指标 {key} 与重算不一致')
    for label in classes:
        for key, value in original_metrics['per_class'][label].items():
            require(math.isclose(saved['per_class'][label][key], value, abs_tol=1e-12), '保存的每类指标与重算不一致')
    metrics, parsed = node_classification_metrics([r['llm_raw_output'] for r in records], references, classes,
                                                 policy=policy)
    for row, pred in zip(records, parsed):
        row.update(parsed_pred=pred, is_correct=pred == row['true_label'], valid_answer=pred is not None)
    return dict(name=folder.name, folder=folder, config=config, records=records, references=references,
                parsed=parsed, metrics=metrics, classes=classes, method=args['method'],
                seed=config['generation']['seed'], prediction_sha256=fingerprint(path))


def error_details(run):
    classes = run['classes']
    rows = []
    generation = run['config']['generation']
    for r, pred in zip(run['records'], run['parsed']):
        if pred == r['true_label']:
            continue
        raw = r['llm_raw_output']
        mentioned = [c for c in classes if normalize_label(c) in normalize_label(raw)]
        tokens = r.get('generated_token_ids', [])
        eos = generation.get('eos_token_id', 2)
        eos_ids = eos if isinstance(eos, list) else [eos]
        generated_count = len(tokens) - int(bool(tokens) and tokens[0] == 1)
        at_limit = bool(tokens) and tokens[-1] not in eos_ids and generated_count >= generation['max_new_tokens']
        reason = ('合法但分类错误' if pred is not None else '空输出' if not raw.strip() else
                  '达到生成上限且未见EOS' if at_limit else
                  '提到唯一候选但不满足解析规则_需人工核查' if len(mentioned) == 1 else
                  '提到多个候选_需人工核查' if len(mentioned) > 1 else '未识别到完整候选名称')
        rows.append(dict(sample_index=r['sample_index'], node_id=r['node_id'], true_label=r['true_label'],
                         parsed_pred=pred or '__INVALID__', error_type=reason,
                         llm_raw_output=raw, mentioned_labels=' | '.join(mentioned),
                         possible_generation_limit=at_limit, generated_tokens=generated_count,
                         prompt=r.get('prompt', '')))
    return rows


def confusion_rows(run):
    counts = Counter(zip(run['references'], [p or '__INVALID__' for p in run['parsed']]))
    support = Counter(run['references'])
    return [dict(true_label=c, predicted_label=p, count=counts[c, p],
                 fraction_of_true_class=counts[c, p] / support[c])
            for c in run['classes'] for p in run['classes'] + ['__INVALID__']]


def class_rows(run, groups):
    n = len(run['records'])
    rows = []
    for c, values in run['metrics']['per_class'].items():
        predicted = values['true_positive'] + values['false_positive']
        rows.append(dict(label=c, group=groups[c], **values, predicted_count=predicted,
                         true_share=values['support'] / n, predicted_share=predicted / n,
                         prediction_inflation=predicted / values['support'],
                         accuracy_contribution=values['true_positive'] / n))
    return sorted(rows, key=lambda r: (-r['support'], r['label']))


def score_subset(run, indices):
    if not indices:
        return dict(n=0, accuracy=None, valid_answer_rate=None)
    return dict(n=len(indices), accuracy=float(np.mean([run['parsed'][i] == run['references'][i] for i in indices])),
                valid_answer_rate=float(np.mean([run['parsed'][i] is not None for i in indices])))


def group_rows(run, groups):
    rows = []
    for group in sorted(set(groups.values())):
        labels = [c for c in run['classes'] if groups[c] == group]
        indices = [i for i, c in enumerate(run['references']) if c in labels]
        # F1 始终来自全测试集的混淆计数，保留其他类别误报到本组的 FP。
        rows.append(dict(group=group, num_classes=len(labels), **score_subset(run, indices),
                         macro_f1=float(np.mean([run['metrics']['per_class'][c]['f1'] for c in labels]))))
    return rows


def graph_stats(y, edges):
    """Directed, unique non-self edges; chance correction uses out/in marginals."""
    y = np.asarray(y, dtype=np.int64)
    src, dst = np.asarray(edges, dtype=np.int64)
    valid_labels = y >= 0
    keep = (src != dst) & valid_labels[src] & valid_labels[dst]
    pairs = np.unique(np.stack((src[keep], dst[keep]), axis=1), axis=0)
    src, dst = pairs.T if len(pairs) else (np.array([], dtype=np.int64),) * 2
    labels = sorted(set(y[valid_labels].tolist()))
    id_map = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    mixing = np.zeros((k, k), dtype=np.int64)
    if len(src):
        a = np.array([id_map[int(c)] for c in y[src]])
        b = np.array([id_map[int(c)] for c in y[dst]])
        np.add.at(mixing, (a, b), 1)
    degree = np.bincount(dst, minlength=len(y))
    same = np.bincount(dst, weights=(y[src] == y[dst]).astype(float), minlength=len(y))
    hi = np.full(len(y), np.nan)
    np.divide(same, degree, out=hi, where=degree > 0)
    m = len(src)
    baseline = float(np.dot(mixing.sum(0) / m, mixing.sum(1) / m)) if m else None
    hedge = float(np.trace(mixing) / m) if m else None
    adjusted = ((hedge - baseline) / (1 - baseline)) if m and baseline < 1 - 1e-12 else None
    summary = dict(num_nodes=len(y), labeled_nodes=int(valid_labels.sum()), num_classes=k,
                   original_directed_edges=int(len(edges[0])), unique_nonself_labeled_edges=m,
                   no_labeled_in_neighbors=int(np.sum(valid_labels & (degree == 0))),
                   edge_homophily=hedge, node_homophily=float(np.nanmean(hi)) if m else None,
                   chance_edge_homophily=baseline, adjusted_homophily=adjusted,
                   definition='directed unique non-self labeled edges; incoming neighbors; isolate excluded')
    summary['class_balance'] = class_balance(Counter(y[valid_labels].tolist()).values())
    per_class = []
    for c in labels:
        mask = y == c
        d = degree[mask]
        h = hi[mask]
        per_class.append(dict(label_id=c, support=int(mask.sum()), mean_in_degree=float(d.mean()),
                              median_in_degree=float(np.median(d)), max_in_degree=int(d.max()),
                              mean_node_homophily=float(np.nanmean(h)) if np.any(np.isfinite(h)) else None,
                              edge_weighted_homophily=float(same[mask].sum() / d.sum()) if d.sum() else None))
    return summary, per_class, mixing, labels, pairs, hi


class GraphReader:
    def __init__(self, data_dir, mode):
        self.folder, self.mode, self.loaded = Path(data_dir), mode, {}

    def get(self, name):
        if self.mode == 'off':
            return None
        filename = 'graph_data_ecommercial.pt' if name in ECOM else 'graph_data_paper.pt'
        if filename not in self.loaded:
            import torch
            kwargs = {'map_location': 'cpu'}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                kwargs['weights_only'] = False  # Only the user's trusted released graph assets.
            self.loaded[filename] = torch.load(self.folder / filename, **kwargs)
        graph = self.loaded[filename][ECOM.get(name, name)]
        require(getattr(graph, 'y', None) is not None, f'{name} 图文件没有节点标签')
        return graph


def profile_graph(graph, name, instructions, out):
    y = graph.y.detach().cpu().numpy().reshape(-1)
    require(np.all(np.isfinite(y)) and np.all(y == np.floor(y)), '图标签须为整数类别 ID')
    edges = graph.edge_index.detach().cpu().numpy()
    require(edges.ndim == 2 and edges.shape[0] == 2 and np.all(edges >= 0)
            and (not edges.size or edges.max() < len(y)), '全图 edge_index 无效')
    names, reverse = {}, {}
    if instructions is not None:
        for row in instructions:
            node = int(row['node_set'][0])
            require(0 <= node < len(y), '测试 node_id 超过图节点范围')
            code, label = int(y[node]), row['output']
            require(code >= 0, '测试目标缺少图标签')
            require(names.get(code, label) == label and reverse.get(label, code) == code,
                    '图的数字标签与测试字符串标签不能一一对应')
            names[code], reverse[label] = label, code
    summary, classes, mixing, ids, pairs, hi = graph_stats(y, edges)
    summary['dataset'] = name
    for row in classes:
        row['label'] = names.get(row['label_id'], f"class_id_{row['label_id']}")
    save_json(out / f'{name}_graph_profile.json', summary)
    save_csv(out / f'{name}_graph_classes.csv', classes)
    save_csv(out / f'{name}_graph_mixing.csv',
             [dict(source_class=names.get(a, str(a)), target_class=names.get(b, str(b)),
                   count=int(mixing[i, j]), source_row_fraction=float(mixing[i, j] / mixing[i].sum())
                   if mixing[i].sum() else None)
              for i, a in enumerate(ids) for j, b in enumerate(ids)])
    return dict(summary=summary, y=y.astype(np.int64), pairs=pairs, hi=hi, names=names, classes=classes)


def local_homophily(instructions, labels):
    rows = []
    for i, row in enumerate(instructions):
        nodes = np.asarray(row['node_set'], dtype=np.int64)
        edges = np.asarray(row['edge_index'], dtype=np.int64).reshape(2, -1)
        require(not edges.size or (edges.min() >= 0 and edges.max() < len(nodes)), '指令子图边不是局部索引')
        incoming = edges[0, (edges[1] == 0) & (edges[0] != 0)]
        target_label = labels.get(int(nodes[0]))
        observed = [labels.get(int(nodes[j])) for j in incoming]
        known = [label for label in observed if label is not None]
        coverage = len(known) / len(incoming) if len(incoming) else None
        partial_h = sum(label == target_label for label in known) / len(known) if known and target_label is not None else None
        h = partial_h if coverage == 1 else None
        bucket = BINS[min(int(h * 5), 4)] if h is not None else '无跨节点入邻居' if not len(incoming) else '邻居标签不完整'
        rows.append(dict(sample_index=i, node_id=int(nodes[0]), incoming_messages=len(incoming),
                         unique_in_neighbors=len(set(incoming.tolist())), neighbor_label_coverage=coverage,
                         partial_observed_homophily=partial_h, local_homophily=h, homophily_bin=bucket))
    return rows


class FeatureReader:
    """Read durable NumPy arrays; bfloat16 bits decode exactly, without Torch."""
    def __init__(self, run):
        self.folder = run['folder'] / 'feature_store'
        self.maps, self.fallback = {}, None
        n = len(run['records'])
        manifest_path = self.folder / 'manifest.json'
        final_path = run['folder'] / f"features_{run['config']['args']['test_dataset']}.pt"
        cache_complete = manifest_path.exists() and read_json(manifest_path).get('complete', False)
        if cache_complete or not final_path.exists():
            self.manifest = read_json(manifest_path)
            require(self.manifest['complete'] and self.manifest['num_expected'] == n
                    and self.manifest['num_saved'] == n, '特征缓存没有完整提交')
            written = np.load(self.folder / 'written.npy', allow_pickle=False)
            require(written.shape == (n,) and written.dtype == np.bool_ and written.all(), '特征缓存有效行标记不完整')
            for key, spec in self.manifest['arrays'].items():
                file = self.folder / spec['file']
                require(file.resolve().parent == self.folder.resolve(), '特征清单引用目录外文件')
                array = np.load(file, mmap_mode='r', allow_pickle=False)
                require(list(array.shape) == spec['shape'] and len(array) == n
                        and str(array.dtype) == spec['numpy_dtype'], f'{key} 特征形状/类型不符')
                self.maps[key] = array
            node_ids = np.asarray(self.maps['node_ids'])
        else:
            import torch
            path = final_path
            kwargs = {'map_location': 'cpu'}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                kwargs['weights_only'] = False
            self.fallback = torch.load(path, **kwargs)
            require(self.fallback['metadata']['complete'], '.pt 特征不完整')
            require(self.fallback['sample_indices'].tolist() == list(range(n)), '.pt 特征行序错误')
            require(self.fallback['true_labels'] == run['references'], '.pt 特征标签错位')
            self.manifest = self.fallback['metadata']
            node_ids = self.fallback['node_ids'].numpy()
        require(np.array_equal(node_ids, [r['node_id'] for r in run['records']]), '特征 node_ids 与预测不一致')
        require(all(r.get('feature_row') == i for i, r in enumerate(run['records'])), '预测中的 feature_row 错位')

    def shape(self, name):
        value = self.maps.get(name) if self.fallback is None else self.fallback.get(name)
        return tuple(value.shape) if value is not None else None

    def get(self, name, rows, slot=None):
        if self.fallback is not None:
            value = self.fallback[name][rows]
            if slot is not None:
                value = value[:, slot, :]
            result = value.float().numpy()
        else:
            value = self.maps[name][rows]
            if slot is not None:
                value = value[:, slot, :]
            if self.manifest['arrays'][name]['torch_dtype'] == 'bfloat16':
                require(value.dtype == np.int16, 'BF16 缓存不是 int16 原始位')
                result = (value.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                result = value.astype(np.float32)
        result = result.reshape(len(rows), -1)
        require(np.isfinite(result).all(), f'{name} 含 NaN/Inf')
        return result


def effective_rank(z):
    # Center first so a shared projector bias does not dominate the spectrum.
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    gram = z @ z.T if z.shape[0] <= z.shape[1] else z.T @ z
    eigenvalues = np.linalg.eigvalsh(gram)
    require(eigenvalues.min(initial=0) >= -1e-8 * max(1., eigenvalues.max(initial=0)), 'Gram 矩阵数值异常')
    singular = np.sqrt(np.maximum(eigenvalues, 0))[::-1]
    # Tiny negative/positive Gram roundoff is below reliable numerical rank.
    singular = singular[singular > (singular[0] * 1e-7 if len(singular) else 0)]
    if not len(singular):
        return dict(effective_rank=0., normalized_effective_rank=0., top1_variance_share=None)
    p = singular / singular.sum()
    erank = float(np.exp(-np.sum(p * np.log(p))))
    cap = min(z.shape[0] - 1, z.shape[1])
    return dict(effective_rank=erank, normalized_effective_rank=erank / cap if cap else None,
                top1_variance_share=float(singular[0] ** 2 / np.sum(singular ** 2)))


def cosine_separation(z, labels):
    norms = np.linalg.norm(z, axis=1)
    unit = z / np.maximum(norms[:, None], 1e-12)
    means, within = [], []
    for label in sorted(set(labels)):
        subset = unit[np.asarray(labels) == label]
        means.append(subset.mean(0))
        if len(subset) > 1:
            within.append(float((np.dot(subset.sum(0), subset.sum(0)) - np.sum(subset * subset)) /
                                (len(subset) * (len(subset) - 1))))
    cross = [float(np.dot(a, b)) for i, a in enumerate(means) for b in means[i + 1:]]
    return dict(class_balanced_within_cosine=float(np.mean(within)) if within else None,
                class_balanced_between_cosine=float(np.mean(cross)) if cross else None,
                cosine_margin=float(np.mean(within) - np.mean(cross)) if within and cross else None,
                zero_norm_rows=int(np.sum(norms <= 1e-12)))


def feature_diagnostics(run, profile, graph, out, sample_cap, edge_cap, seed):
    features = FeatureReader(run)
    n = len(run['records'])
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, min(sample_cap, n), replace=False))
    selected_labels = [run['references'][i] for i in indices]
    # One reproducible sample for every stage; small/rare classes may be absent.
    save_csv(out / 'feature_sample_indices.csv', [dict(sample_index=int(i), node_id=run['records'][i]['node_id'],
                                                    true_label=run['references'][i]) for i in indices])
    if graph is not None:
        nodes = [r['node_id'] for r in run['records']]
        reference_x = graph.x[nodes].detach().cpu().float().numpy()
        require(np.array_equal(features.get('X', np.arange(n)), reference_x), '保存的 X 与原始图目标节点特征不一致')
    pairs = np.empty((0, 2), dtype=np.int64)
    eligible = 0
    incident = 0
    if profile is not None:
        mapping = np.full(len(profile['y']), -1, dtype=np.int64)
        mapping[[r['node_id'] for r in run['records']]] = np.arange(n)
        endpoints = mapping[profile['pairs']]
        incident = int(np.sum(np.any(endpoints >= 0, axis=1)))
        pairs = endpoints[np.all(endpoints >= 0, axis=1)]
        eligible = len(pairs)
        if len(pairs) > edge_cap:
            pairs = pairs[np.sort(rng.choice(len(pairs), edge_cap, replace=False))]
    labels = np.array(run['references'])
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]] if len(pairs) else np.array([], dtype=bool)
    save_csv(out / 'feature_energy_edges.csv', [dict(source_sample_index=int(a), target_sample_index=int(b),
                                                  same_label=bool(labels[a] == labels[b])) for a, b in pairs])
    stages = [('X', 'X', None)]
    if features.shape('H_GNN') is not None:
        stages.append(('H_GNN', 'H_GNN', None))
        require(features.shape('T_graph') is not None and len(features.shape('T_graph')) == 3, '缺少三维 T_graph')
        stages += [('T_graph_flat', 'T_graph', None)]
        stages += [(f'T_graph_token_{i+1}', 'T_graph', i) for i in range(features.shape('T_graph')[1])]
    elif run['method'] == 'tea_glm':
        raise ValueError('TEA-GLM 特征文件缺少 H_GNN')
    results = []
    for stage, name, slot in stages:
        print(f"  features {run['name']} / {stage}", flush=True)
        z = features.get(name, indices, slot)
        norms = np.linalg.norm(z, axis=1)
        row = dict(stage=stage, sampled_nodes=len(indices), dimension=z.shape[1],
                   sampled_class_count=len(set(selected_labels)), mean_norm=float(norms.mean()),
                   median_norm=float(np.median(norms)), **effective_rank(z), **cosine_separation(z, selected_labels))
        energies, valid_edges = [], []
        for start in range(0, len(pairs), 128):
            block = pairs[start:start + 128]
            a, b = features.get(name, block[:, 0], slot), features.get(name, block[:, 1], slot)
            na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            valid = (na > 1e-12) & (nb > 1e-12)
            cosine = np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-24)
            energies.extend((1 - np.clip(cosine, -1, 1)).tolist())
            valid_edges.extend(valid.tolist())
        energy, valid = np.array(energies), np.array(valid_edges, dtype=bool)
        row.update(eligible_test_induced_edges=eligible, sampled_edges=len(pairs),
                   zero_endpoint_edges=int(np.sum(~valid)), incident_edges=incident,
                   test_induced_edge_coverage=eligible / incident if incident else None)
        for label, mask in [('all', valid), ('same_label', valid & same), ('different_label', valid & ~same)]:
            row[f'cosine_edge_energy_{label}'] = float(energy[mask].mean()) if mask.any() else None
            row[f'energy_edge_count_{label}'] = int(mask.sum())
        results.append(row)
    save_csv(out / 'feature_diagnostics.csv', results)
    save_json(out / 'feature_diagnostics.json', dict(stages=results, seed=seed,
              rank_note='Centered singular-value entropy on the sampled matrix, not full dataset rank; singular values <= 1e-7 of the largest are treated as numerical zero.',
              cosine_note='Uncentered unit vectors; equal weight for each represented class / class pair.',
              energy_note='Mean 1-cosine on fixed directed non-self test-induced edges; only sampled endpoints; not full graph Dirichlet energy.',
              sample_label_counts=dict(Counter(selected_labels)),
              limitation='Geometry alone does not establish oversmoothing, information loss, or causal effects.'))
    return results


def bucket_rows(run, local):
    buckets = BINS + ['无跨节点入邻居', '邻居标签不完整']
    return [dict(bucket=bucket, **score_subset(run, [i for i, r in enumerate(local) if r['homophily_bin'] == bucket]))
            for bucket in buckets]


def numeric_scores(target, pred, k):
    # Invalid predictions use column k and contribute FN, never a new macro class.
    cm = np.bincount(target * (k + 1) + pred, minlength=k * (k + 1)).reshape(k, k + 1)
    tp = np.diag(cm[:, :k])
    denom = cm.sum(1) + cm[:, :k].sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k, dtype=float), where=denom > 0)
    return float(tp.sum() / len(target)), float(f1.mean())


def paired_bootstrap(base, tea, repeats, seed):
    classes = tea['classes']
    ids = {c: i for i, c in enumerate(classes)}
    y = np.array([ids[c] for c in tea['references']])
    a = np.array([ids.get(p, len(ids)) for p in base['parsed']])
    b = np.array([ids.get(p, len(ids)) for p in tea['parsed']])
    strata = [np.flatnonzero(y == i) for i in range(len(ids))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        selection = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        aa, af = numeric_scores(y[selection], a[selection], len(ids))
        ba, bf = numeric_scores(y[selection], b[selection], len(ids))
        values.append([ba - aa, bf - af])
    return dict(bootstrap_repeats=repeats, seed=seed,
                accuracy_delta_ci95=np.quantile(values, [.025, .975], axis=0)[:, 0],
                macro_f1_delta_ci95=np.quantile(values, [.025, .975], axis=0)[:, 1],
                limitation='Class-stratified paired node bootstrap; graph dependence ignored; descriptive only.')


def compare_runs(base, tea, local, out, repeats, seed):
    a, b = base['config']['generation'], tea['config']['generation']
    keys = ['seed', 'backbone', 'source_dataset', 'test_dataset', 'batch_size', 'max_text_length',
            'do_sample', 'temperature', 'max_new_tokens', 'top_p', 'top_k', 'num_beams',
            'repetition_penalty', 'model_dtype', 'num_processes', 'pad_token_id', 'eos_token_id']
    mismatch = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    require(not mismatch, f'配对配置不一致，拒绝计算差值: {mismatch}')
    require(base['references'] == tea['references'], '配对参考标签不同')
    for ar, br in zip(base['records'], tea['records']):
        require(ar['node_id'] == br['node_id'], '配对节点不同')
        graph_removed = re.sub(r'<Node \d+>(?:\s+<Node \d+>)*', '', br['prompt'])
        require(ar['prompt'] == graph_removed, '两组 prompt 除图 token 外还有差异')
    rows = []
    for i, (ar, br) in enumerate(zip(base['records'], tea['records'])):
        ac, bc = base['parsed'][i] == ar['true_label'], tea['parsed'][i] == br['true_label']
        transition = 'both_correct' if ac and bc else 'corrected' if bc else 'harmed' if ac else 'both_wrong'
        rows.append(dict(**local[i], true_label=ar['true_label'], vicuna_pred=base['parsed'][i],
                         tea_pred=tea['parsed'][i], transition=transition,
                         vicuna_raw=ar['llm_raw_output'], tea_raw=br['llm_raw_output']))
    save_csv(out / 'paired_nodes.csv', rows)
    transitions = Counter(r['transition'] for r in rows)
    summary = dict(baseline=base['name'], tea=tea['name'], n=len(rows), transitions=dict(transitions),
                   accuracy_delta=tea['metrics']['accuracy'] - base['metrics']['accuracy'],
                   macro_f1_delta=tea['metrics']['macro_f1'] - base['metrics']['macro_f1'],
                   **paired_bootstrap(base, tea, repeats, seed))
    require(math.isclose(summary['accuracy_delta'], (transitions['corrected'] - transitions['harmed']) / len(rows), abs_tol=1e-12),
            '配对转移计数与 Accuracy 差值不一致')
    save_json(out / 'paired_summary.json', summary)
    bins = []
    for bucket in BINS + ['无跨节点入邻居', '邻居标签不完整']:
        indices = [i for i, row in enumerate(rows) if row['homophily_bin'] == bucket]
        ta, tb = score_subset(base, indices), score_subset(tea, indices)
        bins.append(dict(bucket=bucket, n=len(indices), vicuna_accuracy=ta['accuracy'], tea_accuracy=tb['accuracy'],
                         accuracy_delta=tb['accuracy'] - ta['accuracy'] if indices else None,
                         corrected=sum(rows[i]['transition'] == 'corrected' for i in indices),
                         harmed=sum(rows[i]['transition'] == 'harmed' for i in indices)))
    save_csv(out / 'paired_homophily_bins.csv', bins)
    save_csv(out / 'paired_classes.csv', [dict(label=c, support=tea['metrics']['per_class'][c]['support'],
             vicuna_f1=base['metrics']['per_class'][c]['f1'], tea_f1=tea['metrics']['per_class'][c]['f1'],
             f1_delta=tea['metrics']['per_class'][c]['f1'] - base['metrics']['per_class'][c]['f1']) for c in tea['classes']])
    groups = groups_for_classes(tea['classes'], tea['references'])
    grouped_a, grouped_b = group_rows(base, groups), group_rows(tea, groups)
    save_csv(out / 'paired_groups.csv', [dict(group=ar['group'], n=ar['n'], num_classes=ar['num_classes'],
             vicuna_accuracy=ar['accuracy'], tea_accuracy=br['accuracy'], accuracy_delta=br['accuracy'] - ar['accuracy'],
             vicuna_macro_f1=ar['macro_f1'], tea_macro_f1=br['macro_f1'], macro_f1_delta=br['macro_f1'] - ar['macro_f1'])
             for ar, br in zip(grouped_a, grouped_b)])
    # 同一类内部再分同质率，避免把类别构成差异误认为同质率作用。
    save_csv(out / 'paired_class_homophily.csv', [dict(label=c, **binrow) for c in tea['classes']
             for binrow in paired_class_bins(rows, c)])
    return summary


def paired_class_bins(rows, label):
    for bucket in BINS:
        selected = [r for r in rows if r['true_label'] == label and r['homophily_bin'] == bucket]
        counts = Counter(r['transition'] for r in selected)
        yield dict(bucket=bucket, n=len(selected), corrected=counts['corrected'], harmed=counts['harmed'],
                   accuracy_delta=(counts['corrected'] - counts['harmed']) / len(selected) if selected else None)


def render_table(rows, columns, limit=None):
    def cell(value):
        if value is None:
            return '—'
        if isinstance(value, (float, np.floating)):
            return f'{value:.4f}' if math.isfinite(value) else '—'
        return html.escape(str(value))
    selected = rows if limit is None else rows[:limit]
    return '<div class="table"><table><thead><tr>' + ''.join(f'<th>{html.escape(title)}</th>' for _, title in columns) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(r.get(k))}</td>' for k, _ in columns) +
        '</tr>' for r in selected) + '</tbody></table></div>'


def make_plots(run, classes, bins, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = [r['label'] for r in classes]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, .38 * len(labels))))
    y = np.arange(len(labels))
    axes[0].barh(y - .18, [r['support'] for r in classes], .36, label='True count')
    axes[0].barh(y + .18, [r['predicted_count'] for r in classes], .36, label='Predicted count')
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=9)
    axes[0].set_xlabel('Samples'); axes[0].set_title('Class distribution', pad=36)
    for values, name, offset in [('precision', 'Precision', -.24), ('recall', 'Recall', 0), ('f1', 'F1', .24)]:
        axes[1].barh(y + offset, [r[values] for r in classes], .24, label=name)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=9)
    axes[1].set_title('Per-class metrics', pad=36)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'class_diagnostics.{suffix}', dpi=160)
    plt.close(fig)
    matrix = np.array([r['count'] for r in confusion_rows(run)]).reshape(len(run['classes']), -1)
    matrix = matrix / matrix.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(7, .65 * len(labels)), max(5, .55 * len(labels))))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(np.arange(len(labels) + 1)); ax.set_xticklabels(run['classes'] + ['INVALID'], rotation=65, ha='right')
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(run['classes'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label'); ax.set_title('Confusion matrix (row-normalized)')
    fig.colorbar(image, ax=ax); fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'confusion_matrix.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = BINS + ['No in-neighbors', 'Unknown labels']
    x = np.arange(len(bins))
    axes[0].bar(x, [r['n'] for r in bins], color='#64748b')
    axes[0].set_title('Nodes per homophily bin'); axes[0].set_ylabel('Samples')
    axes[1].bar(x, [r['accuracy'] if r['accuracy'] is not None else np.nan for r in bins], color='#2563eb')
    axes[1].set_title('Single-model accuracy by bin'); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_xlabel('Incoming homophily in the actual instruction subgraph')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'homophily_bins.{suffix}', dpi=160)
    plt.close(fig)
    return True


def write_report(out, root, results, warnings):
    parts = ['<h1>TEA-GLM 结果诊断报告</h1>', '<p>所有指标均重新读取原始回答计算；原始结果未修改。</p>',
             f'<p>结果目录：{html.escape(str(root))}</p>',
             '<p>数值 0.01 相当于 1 个百分点。此报告是固定已发布权重的事后分析；相关性不能单独证明过平滑、负迁移或某模块的因果作用。</p>',
             '<p>Macro F1（全样本）将无效预测计入漏判；Macro F1（合法回答）只统计可解析回答。两者都固定使用完整类别集合，零分母取 0。合法回答指标存在样本筛选效应。</p>']
    if warnings:
        parts += ['<h2>当前数据限制</h2><ul>' + ''.join(f'<li>{html.escape(w)}</li>' for w in warnings) + '</ul>']
    overview = []
    for dataset, items in results.items():
        for item in items['runs']:
            run = item['run']
            overview.append(dict(dataset=dataset, run=run['name'], **{k: run['metrics'][k]
                for k in ('num_samples', 'accuracy', 'macro_f1', 'macro_f1_legal', 'parser_policy', 'valid_answer_rate', 'invalid_answers')},
                batch_size=run['config']['generation']['batch_size'], seed=run['seed']))
    parts.append(render_table(overview, [('dataset', '数据集'), ('run', '运行'), ('num_samples', '样本数'),
        ('parser_policy', '解析规则'), ('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1（全样本）'),
        ('macro_f1_legal', 'Macro F1（合法回答）'), ('valid_answer_rate', '合法回答率'),
        ('invalid_answers', '非法回答数'), ('batch_size', 'batch size'), ('seed', '种子')]))
    for dataset, items in results.items():
        parts.append(f'<h2>{html.escape(dataset)}</h2>')
        profile = items.get('profile')
        if profile:
            p = profile['summary']
            def number(value):
                return f'{value:.4f}' if value is not None else '未定义'
            parts.append('<p>完整图画像：' + html.escape(f"节点 {p['num_nodes']}；边同质率 {number(p['edge_homophily'])}；"
                f"节点同质率 {number(p['node_homophily'])}；校正同质率 {number(p['adjusted_homophily'])}。") +
                '它们描述图结构，不是模型准确率；推理分桶采用下方的实际指令子图。</p>')
        for item in items['runs']:
            run, classes = item['run'], item['classes']
            link = f"{dataset}/{run['name']}"
            n = len(run['records'])
            majority = max(r['support'] for r in classes) / n
            inflation = max(classes, key=lambda r: r['false_positive'])
            parts.append(f"<h3>{html.escape(run['name'])}</h3>")
            parts.append(f'<p>测试集最大类占比为 {majority:.2%}。真实样本最多的类别是 '
                f"{html.escape(classes[0]['label'])}（{classes[0]['support']} 条）。最大误报接收类是 "
                f"{html.escape(inflation['label'])}：真实 {inflation['support']} 条，预测 {inflation['predicted_count']} 条，"
                f"其中误报 {inflation['false_positive']} 条。请用混淆矩阵确认错误来自哪里，不能直接归因于邻居聚合。</p>")
            parts.append(render_table(classes, [('label', '类别'), ('group', '规模组'), ('support', '真实数量'),
                ('predicted_count', '预测数量'), ('prediction_inflation', '预测/真实'), ('precision', '精确率'),
                ('recall', '召回率'), ('f1', 'F1'), ('accuracy_contribution', '对总体 Accuracy 的贡献')]))
            parts.append('<p>规模组 Macro F1 从全测试集的每类 F1 求均值，保留来自组外的误报。样本不足 20 条的类别请谨慎解释。</p>')
            parts.append(render_table(item['groups'], [('group', '规模组'), ('num_classes', '类别数'),
                ('n', '样本数'), ('accuracy', 'Accuracy'), ('macro_f1', '组内类别平均 F1')]))
            top_errors = sorted((r for r in item['confusion'] if r['true_label'] != r['predicted_label'] and r['count']),
                                key=lambda r: -r['count'])[:10]
            parts.append('<h4>最常见的错误去向</h4>' + render_table(top_errors,
                [('true_label', '真实类别'), ('predicted_label', '预测类别'), ('count', '数量'), ('fraction_of_true_class', '占真实类别比例')]))
            reasons = Counter(r['error_type'] for r in item['errors'])
            parts.append('<h4>输出与解析诊断</h4>' + render_table([dict(reason=k, count=v) for k, v in reasons.most_common()],
                         [('reason', '错误类型'), ('count', '数量')]))
            parts.append('<p>“提到唯一候选”仅是人工复核线索，不代表答案正确；正式指标没有放宽解析规则。</p>')
            invalid = [r for r in item['errors'] if r['parsed_pred'] == '__INVALID__'][:6]
            parts.append(render_table(invalid, [('node_id', '节点 ID'), ('true_label', '真实类别'), ('llm_raw_output', '原始回答')]))
            parts.append('<h4>实际子图局部同质率分桶</h4>' + render_table(item['bins'],
                [('bucket', '区间'), ('n', '样本数'), ('accuracy', 'Accuracy'), ('valid_answer_rate', '合法回答率')]))
            parts.append('<p>这些是单模型相关性统计。桶之间的类别、度数和文本难度可能不同，不能把桶间差异直接当成同质率的因果作用。</p>')
            if item['features']:
                parts.append('<h4>特征生命周期（抽样诊断）</h4>' + render_table(item['features'],
                    [('stage', '阶段'), ('sampled_nodes', '抽样节点数'), ('sampled_class_count', '抽到的类别数'), ('dimension', '维度'),
                     ('effective_rank', '中心化有效秩'), ('normalized_effective_rank', '有效秩/理论上限'),
                     ('mean_norm', '平均范数'), ('cosine_margin', '类内减类间余弦差'),
                     ('cosine_edge_energy_same_label', '同类边能量'), ('cosine_edge_energy_different_label', '异类边能量')]))
                parts.append('<p>X 是发布的 128 维节点特征；H_GNN 是 GNN 输出；T_graph_flat 是五个 token 拼接，其他行分别对应各 token。'
                    '有效秩在同一抽样节点集上计算，受抽样数量和维度上限影响；有效秩下降不等于判别信息丢失。'
                    '边能量只覆盖具有已保存表示的测试节点之间的边，具体数量和覆盖率见 CSV。能量降低不等于有害过平滑。'
                    '维度较高阶段的满数据集 SVD 未计算，不应把此表称作全量秩。类内/类间统计也仅覆盖抽样里出现的类别，极少数类别可能未抽到；抽样类别数量见 JSON。</p>')
            parts.append(f'<p>明细：<a href="{link}/class_metrics.csv">每类指标 CSV</a> · '
                         f'<a href="{link}/errors.csv">全部错误与原始回答</a> · '
                         f'<a href="{link}/node_diagnostics.csv">逐节点诊断</a> · '
                         f'<a href="{link}/confusion.csv">混淆计数</a></p>')
            if item['plots']:
                parts.append(f'<img src="{link}/class_diagnostics.png" alt="class metrics">'
                             f'<img src="{link}/confusion_matrix.png" alt="confusion matrix">'
                             f'<img src="{link}/homophily_bins.png" alt="homophily bins">')
        for pair in items['pairs']:
            parts.append('<h3>Vicuna 与 TEA-GLM 配对结果</h3><pre>' + html.escape(json.dumps(clean_json(pair), ensure_ascii=False, indent=2)) + '</pre>')
        if not items['pairs']:
            parts.append('<p><strong>当前无可用的同配置配对结果，不能判断 TEA-GLM 相对纯文本的提升或负迁移。</strong></p>')
    parts += ['<h2>指标词典</h2><dl>' + ''.join(f'<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>' for k, v in GLOSSARY.items()) + '</dl>',
              '<h2>下一步能做什么</h2><p>补齐相同 batch size、种子、基座和解码配置的 Vicuna 结果后，重新运行本脚本即可得到配对改正/受损节点及差值。'
              'Self-only、token 打乱和高低频干预需要额外推理，本报告未执行。完整邻域特征和 GNN 第一层表示不在当前导出文件中。</p>']
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + \
        '<title>TEA-GLM 诊断报告</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:35px auto;padding:0 20px;color:#17212f}' + \
        'h1,h2,h3{line-height:1.3}h2{border-bottom:2px solid #cbd5e1;padding-top:25px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}' + \
        'th,td{border:1px solid #d5dce5;padding:8px;vertical-align:top}th{background:#eef3f9}tr:nth-child(even){background:#f8fafc}' + \
        'img{max-width:100%;margin:18px 0}dt{font-weight:bold;margin-top:14px}dd{margin-left:0}pre{white-space:pre-wrap;background:#f1f5f9;padding:15px}a{color:#1d4ed8}</style><body>' + ''.join(parts) + '</body></html>'
    (out / 'report.html').write_text(document, encoding='utf-8')
    save_csv(out / 'summary.csv', overview)
    save_json(out / 'metric_glossary.json', GLOSSARY)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--datasets', nargs='+', default=['history', 'pubmed'])
    parser.add_argument('--instruction-dir', type=Path, default=ROOT / 'instruction')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, help='新目录，拒绝覆盖；默认放在本次结果目录下')
    parser.add_argument('--graph-mode', choices=['auto', 'required', 'off'], default='auto')
    parser.add_argument('--no-features', action='store_true', help='仅做预测/结构分析，跳过特征统计')
    parser.add_argument('--no-plots', action='store_true', help='跳过可选 Matplotlib 图，仍生成 HTML/CSV')
    parser.add_argument('--feature-samples', type=int, default=512, help='每阶段使用同一批最多 512 个节点做谱统计')
    parser.add_argument('--edge-samples', type=int, default=10000, help='最多抽取的测试诱导有向边')
    parser.add_argument('--bootstrap', type=int, default=1000, help='有配对基线时的分层成对 bootstrap 次数')
    parser.add_argument('--analysis-seed', type=int, default=2026)
    parser.add_argument('--evaluation-policy', choices=['unique', 'strict'], default='unique',
                        help='先核验原始评分，再用此统一规则分析；不改写原始日志')
    args = parser.parse_args()
    require(args.feature_samples >= 2 and args.edge_samples > 0 and args.bootstrap >= 100, '抽样参数须满足 nodes>=2, edges>0, bootstrap>=100')
    root = results_root(args.results, args.datasets)
    out = (args.output or root / datetime.now().strftime('analysis_%Y%m%d_%H%M%S')).resolve()
    require(not out.exists(), f'输出目录已存在，请另选 --output：{out}')
    out.mkdir(parents=True)
    provenance = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                      options=vars(args), resolved_results=root, python=sys.version,
                      platform=platform.platform(), numpy=np.__version__, script_sha256=fingerprint(__file__),
                      evaluator_sha256=fingerprint(ROOT / 'utils' / 'utils.py'), runs=[])
    save_json(out / 'analysis_manifest.json', provenance)
    warnings, results = [], {}
    reader = GraphReader(args.data_dir, args.graph_mode)
    profiles_done = set()
    try:
        for dataset in args.datasets:
            print(f'Analyzing {dataset}', flush=True)
            destination = out / dataset
            destination.mkdir()
            instruction_path = args.instruction_dir / dataset / f'{dataset}_dataset_test.json'
            instructions = instruction_rows(instruction_path)
            require(all(r['task'] == 'nc' and r['data'] == dataset for r in instructions), '仅支持单目标数据集 NC 指令')
            folders = sorted(p.parent for p in (root / dataset).glob('*/run_config.json'))
            require(folders, f'{dataset} 没有 run_config.json，无法验证预测与特征来源')
            runs = []
            for folder in folders:
                if read_json(folder / 'run_config.json').get('status') != 'complete':
                    warnings.append(f'{dataset}/{folder.name} 未完成，已排除。')
                    continue
                runs.append(load_run(folder, dataset, instructions, instruction_path, args.evaluation_policy))
            require(runs, f'{dataset} 没有完整运行')
            groups = groups_for_classes(runs[0]['classes'], runs[0]['references'])
            profile, graph = None, None
            if args.graph_mode != 'off':
                try:
                    graph = reader.get(dataset)
                except (ImportError, FileNotFoundError) as exc:
                    if args.graph_mode == 'required':
                        raise
                    warnings.append(f'{dataset} 完整图读取不可用：{exc}；只统计已知邻居标签，缺失不会当作异类。')
                if graph is not None:
                    profile = profile_graph(graph, dataset, instructions, destination)
                    for source in sorted({r['config']['generation']['source_dataset'] for r in runs}):
                        if source not in profiles_done:
                            profile_graph(reader.get(source), source, None, out)
                            profiles_done.add(source)
            labels = ({i: int(c) for i, c in enumerate(profile['y']) if c >= 0} if profile else
                      {int(r['node_set'][0]): r['output'] for r in instructions})
            local = local_homophily(instructions, labels)
            save_csv(destination / 'local_homophily.csv', local)
            data_result = dict(profile=profile, runs=[], pairs=[])
            results[dataset] = data_result
            for run in runs:
                folder = destination / run['name']; folder.mkdir()
                classes = class_rows(run, groups)
                errors, confusion = error_details(run), confusion_rows(run)
                grouped, bins = group_rows(run, groups), bucket_rows(run, local)
                save_json(folder / 'metrics_recomputed.json', run['metrics'])
                save_json(folder / 'test_class_balance.json', class_balance(Counter(run['references']).values()))
                save_csv(folder / 'class_metrics.csv', classes)
                save_csv(folder / 'group_metrics.csv', grouped)
                save_csv(folder / 'errors.csv', errors, ['sample_index', 'node_id', 'true_label', 'parsed_pred', 'error_type',
                    'llm_raw_output', 'mentioned_labels', 'possible_generation_limit', 'generated_tokens', 'prompt'])
                save_csv(folder / 'confusion.csv', confusion)
                save_csv(folder / 'homophily_bins.csv', bins)
                save_csv(folder / 'node_diagnostics.csv', [dict(**local[i], true_label=r['true_label'],
                    parsed_pred=run['parsed'][i], is_correct=r['is_correct'], valid_answer=r['valid_answer'],
                    class_group=groups[r['true_label']], input_tokens=len(r.get('input_token_ids', [])))
                    for i, r in enumerate(run['records'])])
                features = [] if args.no_features else feature_diagnostics(run, profile, graph, folder,
                            args.feature_samples, args.edge_samples, args.analysis_seed)
                plotted = False if args.no_plots else make_plots(run, classes, bins, folder)
                if not args.no_plots and not plotted:
                    warnings.append('Matplotlib 不可用，已保留 HTML 表格和全部 CSV；安装 matplotlib 后可生成 PNG/PDF。')
                data_result['runs'].append(dict(run=run, classes=classes, errors=errors, confusion=confusion,
                                                groups=grouped, bins=bins, features=features, plots=plotted))
                provenance['runs'].append(dict(dataset=dataset, run=run['name'],
                    predictions_sha256=run['prediction_sha256'], config_sha256=fingerprint(run['folder'] / 'run_config.json'),
                    instruction_sha256=fingerprint(instruction_path), configuration=run['config']['generation']))
            for seed in sorted({r['seed'] for r in runs}):
                base = [r for r in runs if r['seed'] == seed and r['method'] == 'vicuna_text']
                tea = [r for r in runs if r['seed'] == seed and r['method'] == 'tea_glm']
                require(len(base) <= 1 and len(tea) <= 1, '同种子同方法存在多个运行，请分开结果目录')
                if base and tea:
                    pair_dir = destination / f'paired_seed{seed}'; pair_dir.mkdir()
                    try:
                        pair = compare_runs(base[0], tea[0], local, pair_dir, args.bootstrap, args.analysis_seed)
                        data_result['pairs'].append(pair)
                    except ValueError as exc:
                        warnings.append(f'{dataset} 配对被拒绝：{exc}')
                        save_json(pair_dir / 'pair_rejected.json', {'reason': str(exc)})
                else:
                    warnings.append(f'{dataset} seed={seed} 尚未同时具有完整 Vicuna 与 TEA-GLM；不计算模型差值。')
            print(f'  verified {len(instructions)} nodes, {len(runs)} runs', flush=True)
        warnings = list(dict.fromkeys(warnings))
        write_report(out, root, results, warnings)
        provenance.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(), warnings=warnings)
        save_json(out / 'analysis_manifest.json', provenance)
        print(f'Saved report: {out / "report.html"}', flush=True)
    except BaseException as exc:
        provenance.update(status='failed', error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(out / 'analysis_manifest.json', provenance)
        raise


if __name__ == '__main__':
    main()
