"""Extract PCA axes from the base LLM's vocabulary embeddings."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.decomposition import PCA


def load_embeddings(backbone):
    folder = Path(backbone)
    key = 'model.embed_tokens.weight'
    # Reading just the embedding shard avoids loading a whole 7B decoder for PCA.
    for suffix in ('safetensors', 'bin'):
        index = folder / f'model.{suffix}.index.json' if suffix == 'safetensors' else folder / 'pytorch_model.bin.index.json'
        single = folder / ('model.safetensors' if suffix == 'safetensors' else 'pytorch_model.bin')
        if index.is_file():
            shard = folder / json.loads(index.read_text(encoding='utf-8'))['weight_map'][key]
        elif single.is_file():
            shard = single
        else:
            continue
        if suffix == 'safetensors':
            from safetensors import safe_open
            with safe_open(shard, framework='pt', device='cpu') as handle:
                return handle.get_tensor(key).float()
        return torch.load(shard, map_location='cpu', weights_only=True)[key].float()
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(backbone, torch_dtype=torch.float32)
    return model.get_input_embeddings().weight.detach().cpu().float()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--n_components', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('PCA_1000_pc_llama.pt'))
    args = parser.parse_args()
    embeddings = load_embeddings(args.backbone)
    if not 0 < args.n_components <= min(embeddings.shape):
        parser.error('--n_components must be within the embedding matrix dimensions')
    # Use the same sklearn PCA and float16 artifact as the released script.
    # Fix its previously unspecified randomized-SVD seed for repeatable runs.
    pca = PCA(n_components=args.n_components, random_state=0)
    pca.fit(embeddings.numpy())
    print(f'Explained variance ratio: {pca.explained_variance_ratio_.sum():.6f}')
    components = torch.from_numpy(pca.components_.astype(np.float16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(components, args.output)
    print(f'Saved {tuple(components.shape)} PCA axes to {args.output}')


if __name__ == '__main__':
    main()
