import argparse
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected, AddRemainingSelfLoops
from tqdm import tqdm

# Supports both python -m GNNPreTrain.train_valid_paper and direct execution.
ROOT = Path(__file__).resolve().parents[1]
# Prepend even when ROOT is already on PYTHONPATH: direct script execution
# otherwise finds GNNPreTrain/model.py instead of the root model package.
sys.path.insert(0, str(ROOT))
from GNNPreTrain.model import GraphSAGE
from GNNPreTrain.dataloader import NodeNegativeLoader
from GNNPreTrain.loss.contrastive_loss import ContrastiveLoss, GraceLoss
from utils.utils import load_checkpoint, seed_everything


def train_epoch(args, data, model, optimizer, criterion, principal_components, device):
    loader = NodeNegativeLoader(
        data, batch_size=args.batch_size, shuffle=True, neg_ratio=args.num_negs,
        num_neighbors=[int(n) for n in args.fan_out.split(',')], num_workers=args.num_workers,
        mask_feat_ratio_1=args.drop_feature_rate_1, mask_feat_ratio_2=args.drop_feature_rate_2,
        drop_edge_ratio_1=args.drop_edge_rate_1, drop_edge_ratio_2=args.drop_edge_rate_2)
    model.train()
    total, count = 0., 0
    dtype = next(model.parameters()).dtype
    for step, (original, view1, view2) in enumerate(tqdm(loader, desc='Pretraining')):
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        z1 = model(view1.x.to(dtype), view1.edge_index)[view1.node_label_index]
        z2 = model(view2.x.to(dtype), view2.edge_index)[view2.node_label_index]
        p1, p2 = model.projection(z1), model.projection(z2)
        if args.use_tp:
            components = principal_components[original.raw_nodes.to(device)] if args.self_tp else principal_components
            loss, _, _ = criterion(z1, z2, p1, p2, components)
        else:
            loss = criterion(p1, p2)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite GNN pretraining loss')
        loss.backward()
        optimizer.step()
        total += loss.item() * len(z1)
        count += len(z1)
        if step % args.log_every == 0:
            print(f'Step {step}: loss={loss.item():.6f}')
        if args.max_steps and step + 1 >= args.max_steps:
            break
    return total / count


def parse_args():
    parser = argparse.ArgumentParser('TEA-GLM GNN pretraining')
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    parser.add_argument('--dataset', default='arxiv', choices=['arxiv', 'computer', 'children'])
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--save_dir', type=Path, default=ROOT / 'saved_model' / 'gnn')
    parser.add_argument('--pca_path', type=Path, default=ROOT / 'GNNPreTrain' / 'PCA_1000_pc_llama.pt')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_hidden', type=int, default=2048)
    parser.add_argument('--num_out', type=int, default=4096)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_negs', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--fan_out', default='25,10')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_tp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--self_tp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--gnn_type', default='sage', choices=['sage', 'gat', 'gcn'])
    parser.add_argument('--max_steps', type=int, default=None, help='Optional smoke-test limit per epoch')
    args = parser.parse_args()
    if len(args.fan_out.split(',')) != args.num_layers:
        parser.error('--fan_out must specify one fanout per GNN layer')
    for name in ('num_epochs', 'num_runs', 'num_hidden', 'num_out', 'num_layers', 'batch_size', 'log_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    for name in ('drop_edge_rate_1', 'drop_edge_rate_2', 'drop_feature_rate_1', 'drop_feature_rate_2'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name} must be between zero and one')
    return args


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --gpu -1 for CPU')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    filename = 'graph_data_paper.pt' if args.dataset == 'arxiv' else 'graph_data_ecommercial.pt'
    datasets = load_checkpoint(args.data_dir / filename, graph_data=True)
    data = datasets['book_children' if args.dataset == 'children' else args.dataset]
    data = AddRemainingSelfLoops()(ToUndirected()(data))
    # Keep the graph on CPU for neighbor sampling; move sampled mini-batches only.
    data = data.cpu()
    components = None
    if args.use_tp:
        if not args.pca_path.is_file():
            raise FileNotFoundError(f'Generate token PCA first: python GNNPreTrain/get_pc.py --backbone <vicuna-path> --output {args.pca_path}')
        components = load_checkpoint(args.pca_path).to(device=device, dtype=dtype)
        if components.shape[-1] != args.num_out:
            raise ValueError('PCA width must equal --num_out and the LLM embedding width')
        if (components.ndim == 3) != args.self_tp:
            raise ValueError('--self_tp requires per-node PCA [N, P, D]; normal PCA has shape [P, D]')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for run in range(args.num_runs):
        seed_everything(run)
        model = GraphSAGE(data.x.shape[1], args.num_hidden, args.num_out, args.num_layers,
                          args.num_out, F.relu, args.dropout, gnn_type=args.gnn_type).to(device=device, dtype=dtype)
        criterion = ContrastiveLoss(args.tau, args.self_tp) if args.use_tp else GraceLoss(args.tau)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_loss, stale = float('inf'), 0
        for epoch in range(args.num_epochs):
            loss = train_epoch(args, data, model, optimizer, criterion, components, device)
            print(f'Run {run}, epoch {epoch}: loss={loss:.6f}')
            if loss < best_loss:
                best_loss, stale = loss, 0
                tag = f'{components.shape[-2]}_tp' if args.use_tp else 'grace'
                target = args.save_dir / f'GraphSAGE_{args.dataset}_{tag}_run_{run + 1}.pth'
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, target)
            else:
                stale += 1
                if stale >= args.patience:
                    break


if __name__ == '__main__':
    main(parse_args())
