# LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings

Reproduction and audit of the authors' NeurIPS 2024 implementation, with documented logic fixes inside the original repository structure.

<img src="doc/framework.png" width="90%">

保留作者 GitHub master 提交 `6cb63667254aaeb9e73687f43ff0dbfdca0ef511` 的全部 30 个文件。根据后续独立分析脚本的要求，另增加 `scripts/analyze_results.py`，不删除作者文件。数据、指令、作者权重及论文另行保留。

## 1. 环境：以作者 GitHub 为基准

已撤回此前 PyTorch 2.8 / Transformers 4.46 的配置。基准为作者 [README](https://github.com/W-rudder/TEA-GLM#1-environment) 和 [requirements.txt](https://github.com/W-rudder/TEA-GLM/blob/master/requirements.txt)。

| 依赖 | 当前配置 | 与作者公开配置的关系 |
| --- | --- | --- |
| Python | 3.9 | 相同 |
| PyTorch / torchvision / torchaudio | 1.13.1 / 0.14.1 / 0.13.1 | 相同 |
| PyTorch CUDA 构建 | 11.7 | 相同 |
| Transformers / Accelerate / PEFT | 4.28.0 / 0.20.3 / 0.3.0 | 相同 |
| PyG | 2.6.1 | 相同 |
| pandas / scikit-learn | 1.5.2 / 1.3.0 | 相同 |
| NumPy | 1.24.4 | 必要兼容修正，作者写的是 2.1.3 |

**不能逐字照搬的一项：** 作者同时指定 Python 3.9 和 NumPy 2.1.3，但 NumPy 2.1 已不支持 Python 3.9，见 [NumPy 官方说明](https://numpy.org/doc/2.1/release/2.1.0-notes.html)。因此仅将作者依赖列表中的 NumPy 修正为 1.24.4。作者未明确锁定的 SciPy、tokenizers、sentencepiece、huggingface_hub、safetensors 和 protobuf 另在 requirements.txt 中锁定兼容版本；这些补充版本不能称为作者原始版本。

在 Linux 服务器项目根目录执行。已安装过新版环境时，新建环境避免混用：

```bash
cd /实际项目路径/TEA-GLM
nvidia-smi
conda create -n teaglm-author python=3.9 pip=25.3 -y
conda activate teaglm-author

# 作者 README 指定的 PyTorch/CUDA 组合。
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
  pytorch-cuda=11.7 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt

# 作者指定的 PyG 扩展版本；禁止回退到源码编译或其他 Torch 构建。
# PyTorch 1.13.1 的页面名是 torch-1.13.0，属于正常命名。
python -m pip install --no-index --only-binary=:all: --no-deps \
  torch_scatter==2.1.0+pt113cu117 \
  torch_sparse==0.6.16+pt113cu117 \
  torch_cluster==1.6.0+pt113cu117 \
  torch_spline_conv==1.2.1+pt113cu117 \
  pyg_lib==0.2.0+pt113cu117 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip check

python - <<'PY'
import sys
import torch, torchvision, torchaudio
import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv, pyg_lib
import torch_geometric, transformers, accelerate, peft, numpy, pandas
print('Python:', sys.version)
print('Torch/CUDA:', torch.__version__, torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('PyG:', torch_geometric.__version__)
print('Transformers/Accelerate/PEFT:', transformers.__version__, accelerate.__version__, peft.__version__)
print('NumPy/pandas:', numpy.__version__, pandas.__version__)
assert sys.version_info[:2] == (3, 9)
assert torch.__version__.split('+')[0] == '1.13.1'
assert torch.version.cuda == '11.7' and torch.cuda.is_available()
assert transformers.__version__ == '4.28.0' and accelerate.__version__ == '0.20.3'
PY
```

依据：[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/#v1131)、[PyG wheel 清单](https://data.pyg.org/whl/torch-1.13.0+cu117.html)。`nvidia-smi` 显示的是驱动可支持的 CUDA 版本，不要求其显示值恰好为 11.7；`torch.version.cuda` 才是此 PyTorch 构建的 CUDA 版本。

## 2. 数据、作者权重与基座

- 图：[下载地址](https://huggingface.co/datasets/W-rudder/graphs/tree/main)，放在 `data/`。
- 指令：[下载地址](https://huggingface.co/datasets/W-rudder/instructions/tree/main)，放在 `instruction/`。
- GNN/projector：[下载地址](https://huggingface.co/W-rudder/TEA-GLM/tree/main)，放在 `saved_model/`。
- 基座：论文使用 [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)，不是 Vicuna-8B。

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmsys/vicuna-7b-v1.5', local_dir='models/vicuna-7b-v1.5', local_dir_use_symlinks=False)"
```

需完整权重分片、索引、配置和 tokenizer。作者 projector 不包含基座 LLM。现有两份图文件、19 份指令和以下四份权重均保留：

```text
saved_model/gnn/GraphSAGE_arxiv_1000_tp.pth
saved_model/gnn/GraphSAGE_computer_1000_tp.pth
saved_model/first_model/TEA-GLM_citation.pth
saved_model/first_model/TEA-GLM_ecom.pth
```

## 3. 参数：论文与 GitHub 脚本的区别

论文第 3.1 节：GNN 为 2 层、batch size 512、60 epochs、学习率 0.02；projector 使用 Adam、学习率 0.001、1 epoch、每卡 batch size 2；硬件为两张 80GB A100，CUDA 11.7。种子 0--4，报告五次实验的均值和标准差。

GitHub 的 projector 脚本则写了学习率 0.002、每卡 batch size 2、累积 1 步，结合原 config.py，优化器为 AdamW。**论文描述与脚本并非完全一致。** 当前训练脚本默认采用论文的 Adam、0.001、1 epoch。GNN 保持 `128 -> 2048 -> 4096`、2 层、5 个图 token、线性 projector。LLM 与 GNN 冻结，只训练 projector；采用作者代码中的 BF16。

单张 24GB GPU 上训练默认 batch size 2、累积 2 步，有效 batch size 4，对应论文双卡每卡 2 的全局 batch。梯度检查点使用旧版接口，可用 `--no-gradient_checkpointing` 关闭。单卡累积与双卡计算不逐位等价；24GB 显存占用仍需服务器实测。

原 GitHub 长度：训练 arxiv 700、computer 1000；测试 arxiv/pubmed 700、cora 850、children/computer/photo 1000、history 1300、sports 500。当前仅将测试 Children 上限修正为 1100：对本地全部指令使用官方 Vicuna tokenizer、Transformers 4.28.0 实测，148 条超过 1000，最长 1035。其他长度保持原值。超长样本会明确报错，不截掉图 token 或答案；`MAX_TEXT_LENGTH` 可显式覆盖整组脚本的上限，须保留配置记录。

## 4. 使用作者权重进行跨数据集节点分类

已有作者 GNN/projector 时，跳过两个训练阶段。先检查四条样本：

```bash
python train_glm.py --backbone models/vicuna-7b-v1.5 \
  --dataset arxiv --test_dataset cora --inference --batch_size 4 --dtype bfloat16 \
  --projector_checkpoint saved_model/first_model/TEA-GLM_citation.pth \
  --max_samples 4 --max_text_length 850 --seed 0 --prefix smoke-cora

# 全量 NC；目标数据集保持作者原测试脚本的列表。
PROJECTOR=saved_model/first_model/TEA-GLM_citation.pth \
  bash scripts/test_citation.sh models/vicuna-7b-v1.5
PROJECTOR=saved_model/first_model/TEA-GLM_ecom.pth \
  bash scripts/test_ecom.sh models/vicuna-7b-v1.5
```

默认推理 batch size 4、种子 0；`BATCH_SIZE`、`SEED`、`PREFIX`、`PROJECTOR` 可显式覆盖。解码采用作者 g_step 的实际设置：采样、温度 0.7、最多生成 80 token；top_p/top_k 沿用基座 generation config，Vicuna top_p 为 0.6。原 config.py 虽写 gen_max_length=64，但原 g_step 实际固定 80；现在以实际解码为准。

每次 NC 运行保存 `results/<目标数据集>/<prefix>_model_results.txt`、`_model_labels.txt`、`_generation_config.json`、`_metrics.json` 和 `_predicted_labels.json`。前两个 txt 是 UTF-8 JSON 数组，按指令顺序对应。记录实际采样参数、种子、batch size 和长度上限。

最终统一评测协议为 **`nc-unified-v1`**。先从回答中独立提取唯一类别，再比较真实标签，绝不使用真实标签帮助选择预测。先做规范化精确/明确别名匹配，失败后检测具有词边界的完整候选名称；名称内嵌时取更完整名称；多个不同候选、没有候选、含否定或不确定标记的散文回答判为无效。该保守规则不是完整的语义理解器，拒绝原因和候选集合保留供人工检查。全数据集、两种模型共用同一规则，不按目标数据集调参。

默认 `--evaluation_policy unique`；`strict` 保留此前整句精确匹配，用于对照。主字段 `macro_f1` 与 `macro_f1_all` 均为全样本 Macro F1，非法回答计为真实类 FN；另报 `macro_f1_legal`，仅对可解析样本计算，与论文附录 B.2 的样本范围一致。两种 F1 都固定使用完整数据集类别集合、零分母取 0；没有合法回答时合法样本 F1 为 null。Accuracy 始终用全部测试样本，非法回答计错。`strict_metrics` 始终提供旧严格规则的 Accuracy/F1/合法率。

参考了 [LLaGA 节点分类评测](https://github.com/VITA-Group/LLaGA/blob/master/eval/eval_res.py) 对 PubMed/arXiv 的“唯一候选”思路，但未复制其各数据集不同的规则。GraphGPT 的公开主推理入口主要导出回答；其代码不能补齐 TEA-GLM 未公开的解析器。**本协议是统一且可审计的自定义评测，不能称作作者原始评测脚本。** 论文附录 B.2 已说明合法样本 F1；缺少的是具体答案解析等实现，而非完全没有指标定义。

单独重算 NC 指标，不加载 LLM：

```bash
python utils/utils.py \
  --predictions results/cora/tea-citation_model_results.txt \
  --instructions instruction/cora/cora_dataset_test.json \
  --policy unique --output results/cora/tea-citation_metrics_unified_v1.json \
  --parsed-output results/cora/tea-citation_predictions_unified_v1.jsonl
```

仅生成前 N 条时，离线评分也需 `--max_samples N`。评分拒绝覆盖已有输出；选择新的输出名，保留原始结果。

### 4.1 四个目标数据集：纯文本 Vicuna 与 TEA-GLM 配对对照

`--method vicuna_text` 是同一份 TEA-GLM 指令经相同预处理后，仅移除 `<Node N>` 图占位符；不改写其余措辞、候选类别、Vicuna 系统模板。引用网络两组都沿用标题输入，电商两组都保留原指令提供的信息。纯文本分支不实例化 GNN/projector、不注入图向量，也不加载微调 LLM/LoRA；读取 graph.x 仅用于保存诊断所需的 X。`--method tea_glm` 使用作者完整 GNN/projector 权重。

下面八次运行采用同一基座、BF16、batch size 4、种子 0，与现有 comparison_20261029_083107 的实际运行一致；采样温度 0.7、最多生成 80 token，top_p 沿用官方基座的 0.6。每次开始推理时重新设置记录的随机种子，避免额外图模块的初始化消耗随机数影响两组起点。推理没有梯度更新，不需要梯度累积。相同种子不保证两组逐步采样相同，也不保证不同硬件逐位一致。如需降低 batch size，应在新的结果目录对两种方法一起重新运行，不能混入当前配对比较。

| 目标 | 来源 / TEA-GLM 权重 | 测试节点数 | 实测最长输入 token | 本次长度上限 |
| --- | --- | ---: | ---: | ---: |
| pubmed | arxiv / citation | 3944 | 227 | 700 |
| history | computer / ecom | 7546 | 1252 | 1300 |
| children | computer / ecom | 14610 | 1035 | 1100 |
| sports | computer / ecom | 34611 | 405 | 500 |

最长长度基于本地发布指令、五个图 token 和旧版 tokenizer 的完整 TEA-GLM 输入。Children 的 1100 是防止全量实验中断的修正，不能称为作者原脚本的 1000；两种方法使用相同上限，动态 padding，不补齐到该上限。八条命令均不设置 `--max_samples`，运行全部测试节点。

在服务器项目根目录执行。`LLM` 必须指向直接包含 `config.json` 和模型分片的目录；若权重放在 `models/vicuna-7b-v1.5`，只修改这一行。先上传本次修改后的代码。

```bash
conda activate teaglm
cd ~/autodl-tmp/TEA-GLM
export LLM="$PWD/models"
export RESULTS_DIR="$PWD/results/comparison_$(date +%Y%m%d_%H%M%S)"
export BATCH_SIZE=4 SEED=0 EXP_NUM=1
echo "Results: $RESULTS_DIR"

# 正向对照：arxiv -> PubMed
METHOD=vicuna_text bash scripts/test_single.sh arxiv pubmed 5 700 \
  "vicuna_text_seed${SEED}" "$LLM"
METHOD=tea_glm bash scripts/test_single.sh arxiv pubmed 5 700 \
  "tea_glm_seed${SEED}" "$LLM" saved_model/first_model/TEA-GLM_citation.pth

# 正向对照：computer -> History
METHOD=vicuna_text bash scripts/test_single.sh computer history 5 1300 \
  "vicuna_text_seed${SEED}" "$LLM"
METHOD=tea_glm bash scripts/test_single.sh computer history 5 1300 \
  "tea_glm_seed${SEED}" "$LLM" saved_model/first_model/TEA-GLM_ecom.pth

# 诊断：computer -> Children
METHOD=vicuna_text bash scripts/test_single.sh computer children 5 1100 \
  "vicuna_text_seed${SEED}" "$LLM"
METHOD=tea_glm bash scripts/test_single.sh computer children 5 1100 \
  "tea_glm_seed${SEED}" "$LLM" saved_model/first_model/TEA-GLM_ecom.pth

# 诊断：computer -> Sports
METHOD=vicuna_text bash scripts/test_single.sh computer sports 5 500 \
  "vicuna_text_seed${SEED}" "$LLM"
METHOD=tea_glm bash scripts/test_single.sh computer sports 5 500 \
  "tea_glm_seed${SEED}" "$LLM" saved_model/first_model/TEA-GLM_ecom.pth
```

脚本强制启用 `--save_features --verbose`，通过 `python -u ... 2>&1 | tee` 保存全部 stdout/stderr，`pipefail` 保留失败退出码。已有运行目录或同名日志会拒绝覆盖；重跑失败实验请更换 `RESULTS_DIR` 或 prefix。重新登录服务器后，离线评测应将 `RESULTS_DIR` 设回已运行的实际目录，不要重新生成时间戳。

每次完成后自动评测。八组结果可再次离线评分，无需加载模型或使用 GPU：

```bash
for target in pubmed history children sports; do
  for method in vicuna_text tea_glm; do
    run_dir="$RESULTS_DIR/$target/${method}_seed${SEED}"
    python utils/utils.py \
      --predictions "$run_dir/predictions_${target}.jsonl" \
      --instructions "instruction/$target/${target}_dataset_test.json" \
      --output "$run_dir/metrics_recomputed.json" || break 2
  done
done
```

评测会重新解析原始回答、核对 sample_index/node_id/true_label，并拒绝将不完整预测当成全量实验。指标包括 Accuracy、Macro F1、合法回答率及每类 precision/recall/F1、TP/FP/FN/support；非法回答不丢弃。下面打印对照表：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root, seed = Path(os.environ['RESULTS_DIR']), os.environ['SEED']
print('dataset,method,n,accuracy,macro_f1,valid_answer_rate')
for target in ('pubmed', 'history', 'children', 'sports'):
    for method in ('vicuna_text', 'tea_glm'):
        path = root / target / f'{method}_seed{seed}' / 'metrics_recomputed.json'
        m = json.loads(path.read_text(encoding='utf-8'))
        print(f"{target},{method},{m['num_samples']},{m['accuracy']:.6f},"
              f"{m['macro_f1']:.6f},{m['valid_answer_rate']:.6f}")
PY
```

这是一对固定作者权重、推理种子 0 的对照实验，不等同于论文五次独立训练的均值和标准差。如需观察采样波动，将 `SEED` 依次设为 1、2、3、4 并分别重跑八条命令；prefix 已包含种子。不要在这组命令中把 `EXP_NUM` 改成 5，否则输出路径规则不同。Sports 在所附表中相对 Vicuna 的 Accuracy 和 Macro F1 均提高，诊断时不预设它必然存在负迁移。

### 4.2 必须保留的阶段 3 文件

以 `tea_glm_seed0` 的 PubMed 为例：

```text
$RESULTS_DIR/
  logs/tea_glm_seed0_arxiv_to_pubmed.log
  pubmed/tea_glm_seed0/
    predictions_pubmed.jsonl
    features_pubmed.pt
    feature_store/          # 运行中保留；完整 PT 经核验后可清理重复缓存
    run_config.json
    generation_config.json
    progress.json           # 运行中的进度；完成后可清理
    metrics.json
```

- JSONL 每行包括要求的 `node_id, true_label, llm_raw_output, parsed_pred, is_correct`，并保存 `sample_index`、实际 prompt、输入/生成 token IDs、包含特殊符号的原始解码、合法回答标记、方法与种子。node_id 来自 `node_set[0]`，不是测试集行号。`llm_raw_output` 是去除特殊 token 后、未做标签解析的生成文本；另一个字段保留带特殊 token 的解码。
- `features_*.pt` 包含 `X, H_GNN, T_graph, node_ids, sample_indices, true_labels, metadata`，按 sample_index 对齐。X 为作者发布 graph.x 的目标节点特征（128 维），不是重新编码的原始文本或 4096 维 LLM 隐状态。TEA-GLM 的 H_GNN 为 `[N,4096]`，T_graph 为 `[N,5,4096]`，直接来自本次推理的同一次前向；按实际 dtype 无损保存到 CPU。纯文本基线保存 X，H_GNN/T_graph 明确为 `None`。
- `run_config.json` 记录状态、样本数、运行参数、实际解码设置、依赖版本、硬件、指令与 projector 的 SHA256，以及源码 SHA256；只有全部文件完成后状态才为 `complete`。正常异常保存 `error.txt` 和已完成的部分特征。原有 `_model_results.txt` 等兼容输出仍保留。
- 每批先刷写特征、再刷写 JSONL，最后提交有效行标记。异常或强制终止后，只使用已提交行；缓存中未提交的预分配空间不能算作有效样本。四个数据集 TEA-GLM 的 BF16 特征本体约 2.8 GiB，连同恢复缓存约 5.6 GiB；再加基线 X 和日志，每个种子建议预留至少 8 GB 输出空间。

正常结束自动生成 `.pt`。完整结果经节点顺序、标签和特征逐位核验后，可以删除重复的 `feature_store/`；分析脚本会自动读取 `.pt`，需要 PyTorch，但不需要加载 LLM。运行未完成时不可删除缓存。若进程被强制终止而未完成最终打包，在原推理进程退出后，可只用 CPU 从缓存恢复：

```bash
CUDA_VISIBLE_DEVICES="" python utils/utils.py --recover-features \
  "$RESULTS_DIR/pubmed/tea_glm_seed${SEED}"
```

恢复结果的 `metadata.complete`、`metadata.num_saved` 和 `sample_indices` 标明是否完整及实际有效行。该命令只恢复特征，不恢复推理、不伪造缺失预测。后续 CPU 分析用 `torch.load(path, map_location='cpu')`，计算 SVD 等统计前可将 BF16 特征 `.float()`。

### 4.3 对已经保存的结果做 CPU 诊断分析

新增独立入口 `scripts/analyze_results.py`，默认分析 `history` 和 `pubmed`。它读取预测 JSONL、运行配置、指令、原图和已保存的特征，不加载 LLM，也不重新推理。所有分析结果写到新目录，已有目录拒绝覆盖；原预测和评测文件保持原样。

```bash
python scripts/analyze_results.py \
  --results results/comparison_20261029_083107 \
  --datasets history pubmed \
  --graph-mode required
```

如果 `results/` 下只有一组 comparison 目录，也可以直接运行 `python scripts/analyze_results.py`。存在多组实验时必须明确选择，避免混合不同运行。默认输出到 `comparison_*/analysis_日期_时间/`，终端最后打印报告路径。打开其中的 `report.html` 即可查看中文说明；CSV 使用 UTF-8 BOM，便于 Excel 打开。

报告及文件包括：

| 文件 | 用途 |
| --- | --- |
| `report.html` | 中文诊断报告，含指标解释、每类表现、错误去向、同质率和特征分析 |
| `summary.csv` | 各次运行的 Accuracy、Macro F1、合法回答率及实际 batch size / seed |
| `metric_glossary.json` | 指标词典，解释数值表示什么、不能推出什么 |
| `analysis_manifest.json` | 分析参数、输入预测/配置哈希、分析脚本/评测器哈希和完成状态 |
| `<数据集>/*_graph_profile.json`、`*_graph_classes.csv`、`*_graph_mixing.csv` | 完整目标图的同质率、类别不平衡、类别间连接和各类别度数；源图 arxiv/computer 的画像在报告根目录 |
| `<数据集>/local_homophily.csv` | 按实际指令子图、入边方向计算的中心节点局部同质率 |
| `<数据集>/<运行>/class_metrics.csv`、`group_metrics.csv` | 每类和 Head/Medium/Tail 指标；分组 F1 始终从全测试集每类 F1 计算，保留组外误报 |
| `<数据集>/<运行>/errors.csv` | 所有错误的节点 ID、真实类别、原始回答、错误类型及 prompt，可直接人工复核 |
| `<数据集>/<运行>/confusion.csv`、`node_diagnostics.csv` | 完整混淆计数、逐节点结构与预测诊断 |
| `<数据集>/<运行>/feature_diagnostics.csv` | X、H_GNN、五个 token 及其拼接的范数、中心化有效秩、类内/类间余弦差、测试诱导边能量 |
| `<数据集>/paired_seed*/` | 两方法齐全且配置相同时才生成配对改正/受损节点、分桶差值、每类差值和 bootstrap 区间 |

脚本首先核对运行完成状态、指令 SHA256、样本数量、node_id/true_label、重新解析结果和保存指标。特征分析还核对提交标记、特征行对应的节点 ID，以及保存的 X 是否等于原图目标节点输入。若不一致会报错，不能将部分结果默认为全量。两组配对还要求基座路径、种子、batch size、输入上限、解码配置相同，且实际 prompt 仅相差 graph tokens；不匹配时报告拒绝原因，保留单模型统计。

**2026-10-29 核验：History 和 PubMed 均已有完整 TEA-GLM 与 Vicuna 输出，实际 batch size 为 4、seed 为 0。** Children、Sports 仍需补齐纯文本基线。Children 的完整旧格式输出与最终 `.pt` 可核验，但详细 JSONL/进度/特征缓存仍混有旧快照；须同步最终日志包后再运行要求逐节点完整记录的分析脚本，不能把旧状态手工改为 complete。

特征统计默认使用同一批最多 512 个随机测试节点、最多 10000 条测试节点诱导边，抽样种子 2026。具体行号、边端点、各类抽样数量和边覆盖率均保存。中心化有效秩是抽样矩阵的统计，低于最大奇异值 `1e-7` 的方向按数值零处理；归一化余弦边能量使用单位向量的平均 `1-cosine`，不是完整图的原始狄利克雷能量。秩下降或边能量下降不能单独证明有害过平滑或判别信息丢失。

可调节分析规模或只做预测分析：

```bash
# 更大的特征抽样会增加 CPU 时间和内存；所有阶段仍使用相同样本。
python scripts/analyze_results.py --results results/comparison_20261029_083107 \
  --feature-samples 1024 --edge-samples 20000

# 仅分析预测，跳过图和特征；未知邻居不会被当作异类。
python scripts/analyze_results.py --results results/comparison_20261029_083107 \
  --graph-mode off --no-features --no-plots
```

基础分析需要 NumPy；读取图文件使用项目原有 PyTorch/PyG 依赖。Matplotlib 可选：安装后自动输出类别诊断、混淆矩阵、同质率分桶的 PNG/PDF；未安装仍有完整 HTML 表格与 CSV，不影响指标统计。保存的 BF16 NumPy 缓存可无损还原，无需 GPU；仅保留 `.pt` 特征时也支持通过 CPU 读取。

分析脚本先按原评测版本核验保存字段和指标，再默认用 `--evaluation-policy unique` 统一重算，不改写原始日志；可显式用 `strict` 做敏感性分析。正文分桶/配对 F1 使用全样本口径，概览同时列出合法样本 F1。对测试标签的使用仅用于事后诊断。Self-only、token 打乱和高低频干预需要新增推理，当前脚本不执行这些实验。

## 5. 重新训练与五个随机种子

使用作者 GNN 重新训练 projector，默认论文超参数：

```bash
EXP_NUM=5 SEED=0 bash scripts/train_arxiv.sh models/vicuna-7b-v1.5
EXP_NUM=5 SEED=0 bash scripts/train_computer.sh models/vicuna-7b-v1.5
```

种子依次为 0--4，权重分别保存到 `saved_model/first_model/tea-citation_seedN_fm_arxiv_epoch0_end.pth` 或 `tea-ecom_seedN_fm_computer_epoch0_end.pth`。使用每个种子对应的权重评估，例如 citation：

```bash
for seed in 0 1 2 3 4; do
  SEED="$seed" PREFIX="tea-citation_seed${seed}" \
    PROJECTOR="saved_model/first_model/tea-citation_seed${seed}_fm_arxiv_epoch0_end.pth" \
    bash scripts/test_citation.sh models/vicuna-7b-v1.5
done
```

对五次指标报告均值和标准差。**同一份作者 projector 仅换五个推理种子，只反映固定权重的采样波动，不能替代五次独立训练。**

若明确选择 GitHub 脚本超参数而非论文文字，显式运行：

```bash
LR=0.002 OPTIM=adamw GRAD_STEPS=1 bash scripts/train_arxiv.sh models/vicuna-7b-v1.5
LR=0.002 OPTIM=adamw GRAD_STEPS=1 bash scripts/train_computer.sh models/vicuna-7b-v1.5
```

这里只恢复脚本超参数，单卡全局 batch 与作者两卡仍不同。

可选 GNN 预训练，PCA 必须来自相同 Vicuna，使用作者 sklearn PCA 与 float16 保存格式：

```bash
python GNNPreTrain/get_pc.py --backbone models/vicuna-7b-v1.5
python GNNPreTrain/train_valid_paper.py --gpu 0 --dataset arxiv
```

输出 `saved_model/gnn/GraphSAGE_arxiv_1000_tp_run_1.pth`；训练 projector 时通过 `--pretrain_gnn` 显式选择新权重。

## 6. 代码兼容与保留的逻辑修复

- Accelerate 0.20.3：直接传 split_batches/dispatch_batches，移除新版 DataLoaderConfiguration；accumulate 只传一个模型，可选 LoRA 分支另行处理梯度同步；仅在初始化日志器后调用 end_training，避免无日志器模式退出时报错。
- Transformers 4.28.0：无参数 gradient_checkpointing_enable，采用该版本的 Llama forward、loss 和 generate；显式传 generation config，防止旧版隐式重建配置覆盖 top_p 等采样设置。
- 全部原有 Python 模块检查 Python 3.9 语法，不引入 Python 3.10 专用语法。
- 保留全量指令、真实 token ID、正确监督标签和 padding、冻结词嵌入 buffer、严格权重加载、逐轮保存、正确 beta2、梯度累积/调度与 PyG propagate 签名等修复。
- 保留论文公式 (6)/(8) 的 GNN 损失修复；重新预训练数值可能不同于原有偏差实现，不影响已发布权重。PCA 随机 SVD 固定种子 0，原脚本未指定种子。
- 不强制 wandb，不引用作者未上传的 accelerate YAML，不硬编码两张 GPU；这些属于启动方式适配。

原文件职责不变：config.py 配置；model/language_model.py 解码；model/graph_transformer.py 和 model/layers.py 图编码/投影；utils/conversation.py 保留原模板；utils/instruction_preprocess.py 数据；utils/trainer_base.py 模型/优化器；utils/utils.py 公共函数及独立 NC 评测；train_glm.py 主循环。

## 7. LP 与验证范围

NC 取子图首节点，LP 取前两个节点均值。需要 LP 时单独运行，例如：

```bash
bash scripts/test_single.sh arxiv cora_LP 5 2048 tea-citation-lp models/vicuna-7b-v1.5 \
  saved_model/first_model/TEA-GLM_citation.pth
```

下载的部分 LP 正样本子图含目标边。默认保留作者输入；`--remove_target_edge` 可移除局部 0->1/1->0 目标边，但会改变评估协议，不能替代严格划分和重新采样。LP 当前导出生成文本，不将硬 Yes/No 当概率计算论文 AUC。

此前新版环境的测试不能作为旧版兼容性证明。本轮在独立 Python 3.9.23 / PyTorch 1.13.1 CPU / Transformers 4.28.0 / Accelerate 0.20.3 环境完成 9 项回归检查，失败项修复后复测通过：版本与 Python 3.9 语法、梯度检查点反传、NC/LP 数据与 padding、小模型实际 CLI 两轮训练及保存重载/NC/LP 生成/独立评测、解码参数、GNN 采样训练、sklearn PCA、指标对照、两套作者权重严格加载和 8 个数据集 NC/LP 共 16 种真实子图前向。pip check 与五个脚本的 bash -n 均通过。测试依赖和测试文件仅置于系统临时目录，不改变项目文件清单，也不修改已有 conda 环境。

本次新增对照与诊断导出，在同一旧版依赖的临时 CPU 环境完成 8 项回归检查：纯文本只移除占位符且不创建 GNN；两组真实小模型 CLI 推理及逐节点/前向特征对齐；不导入 ML 包的离线评分和错位检测；拒绝覆盖；中断后恢复；BF16 位级无损及乱序行恢复；子集评估保留全类别；原训练梯度仍到达 projector。另用官方 Vicuna tokenizer 扫描上述四个数据集全部 60711 条指令，确认命令中的长度上限足够。

以上记录分别对应此前的验证轮次。2026-10-29 本轮在隔离目录使用 Python 3.9 / PyTorch 1.13.1 CPU / Transformers 4.28.0 / Accelerate 0.20.3 / PyG 2.6.1，完成 9 项模型/存储检查：对照作者 GraphEncoder 输出、GraphSAGE 邻居均值、对照作者 Llama logits/采样、冻结模型梯度、论文对比损失公式、真实指令提示文本、两组作者完整 checkpoint 与单独 GNN 参数逐项一致、BF16 特征无损恢复及拒绝覆盖、小模型实际 CLI 训练和两种推理分支。另有 9 项评测/语法检查，以及 PubMed/History 实际分析入口检查。完整六组已有输出按新规则离线重算；旧严格规则仍可复算原指标。

本轮恢复了作者 projector 的正态权重初始化（仅影响将来从头训练；推理会完整加载权重），没有改变已有权重的 GNN/投影前向路径。新日志附带评测版本、解析原因；特征清单和运行进度逐批更新，旧特征缓存不能覆盖已完成 `.pt`。所有原作者文件保留。

3090 / CUDA 11.7 / 完整 Vicuna-7B 的本次代码版本仍需服务器运行；本地验证没有代替全模型 GPU 实验。依赖一致不构成论文数值一致的保证。论文与代码的训练冲突包括：GNN 论文 60 epochs/lr=0.02，对照公开脚本 70/lr=0.002；projector 论文 Adam/lr=0.001，对照脚本 AdamW/lr=0.002；特征对比损失分母及平均系数也不同。当前训练沿用论文设置与已说明的公式修正，保留发布代码的早停机制；这些差异必须披露，不能承诺同时逐项等同两套互相冲突的设置。当前使用作者完整权重的阶段 3 推理不重新执行这些训练步骤。

## 8. 当前正式结果与后续分析（nc-unified-v1）

截至 2026-10-30，四个数据集、八组 seed=0 对照实验均已完成。Children 的正式 TEA 运行是 `tea_glm_seed0_recheck`；旧的不完整运行和临时诊断已清理。最新汇总见 [诊断报告](results/comparison_20261029_083107/diagnostics_nc_unified_v1_recheck/report.html)，保留文件的用途见 [results 目录说明](results/README.md)。不要因评测变化重跑已有完整模型输出。四步诊断中的第 4 步是额外模型干预实验，需要根据前面证据单独设计；相邻特征更相似或秩下降本身不等于有害过平滑，self-only 改善也不能单独证明唯一原因。

若需要重新离线评测，在项目根目录执行；无需再次生成已有回答：

```bash
conda activate teaglm
cd ~/autodl-tmp/TEA-GLM
export RESULTS_DIR="$PWD/results/comparison_20261029_083107"

# 重算全部已完成的旧格式原始回答；检查回答数量、标签顺序和指令指纹。
# 输出目录必须是新的；保留最初的 metrics.json/predictions*.jsonl。
python utils/utils.py --results-root "$RESULTS_DIR" \
  --datasets pubmed history children sports --policy unique \
  --output-dir "$RESULTS_DIR/evaluation_nc_unified_v1_all"
```

输出目录如果已存在，请另选新名称。批量评测只扫描存在的完整旧格式输出，必须核对 `summary.json` 有 **8 组**；清理后的结果目录已保留这八组回答和标签数组。

评分输出含每组 `metrics.json`、`predictions_evaluated.jsonl` 和总表 `summary.json`。后者的 JSONL 是从旧回答重建的评分记录，没有伪造原始 token 日志。解析与真实标签无关；多类别回答统一无效，不以“正确类别也出现了”为理由算对。否定/不确定规则可能拒绝一部分能人工解释的正确回答，因此同时保留严格指标、原文、拒绝原因；不得根据测试集涨跌选择规则。

同步完整的最终原始 JSONL、run_config、日志后，在具有项目依赖的 CPU 环境运行前三步分析：

```bash
python scripts/analyze_results.py --results "$RESULTS_DIR" \
  --datasets pubmed history children sports --evaluation-policy unique \
  --graph-mode required --output "$RESULTS_DIR/diagnostics_nc_unified_v1"
```

当前唯一正式综合诊断位于 `results/comparison_20261029_083107/diagnostics_nc_unified_v1_recheck/`，包含八组运行的统一指标及四组模型配对分析。旧分析与重复重评分目录已删除；必要的历史代码核查和清理记录保存在同级 `audit/`，原始预测、运行配置和最新报告保持不变。

脚本内置 Matplotlib 绘图，缺少该依赖时会跳过 PNG/PDF。已有诊断报告可在安装了 NumPy 和 Matplotlib 的环境中单独补图，无需重新分析特征或运行 LLM：

```bash
python scripts/analyze_results.py --render-only "$RESULTS_DIR/diagnostics_nc_unified_v1_recheck"
```

这会更新同目录 HTML 和图片，保留原指标与 CSV。每组运行生成类别分布/每类指标、行归一化混淆矩阵、最常见错误流向和同质率分桶图，分别保存 PNG 和 PDF。`__INVALID__` 是解析器未提取到唯一合法候选时的诊断占位符，不是模型新增的类别。

## 9. 推理干预诊断：缩放、去邻居、整组 token 置换

2026-10-31：复用原有 `train_glm.py`、`scripts/test_single.sh` 和 `scripts/analyze_results.py`，未增加代码文件或删除作者文件。默认 `--graph_intervention none --graph_token_scale 1` 保持正常模型前向；干预仅允许 NC 推理，不修改权重或训练流程。

| 条件名 | 操作 | 要回答的问题 |
| --- | --- | --- |
| `normal` | 正常 TEA-GLM，必须保留的对照 | 相同代码、样本和种子下的参照 |
| `scale_1e-3` / `scale_1e-4` / `scale_1e-5` | Projector 输出整体乘以固定系数，FP32 乘法后转回注入精度 | 输入尺度是否影响分类和类别偏好；不做逐 token 单位范数归一化 |
| `self_only` | 所有 GNN 层使用空边集，保留自身 root 分支及偏置 | 去除邻居消息后是否改善；不是添加自环令 A=I |
| `shuffle` | 固定无固定点随机置换，将其他节点的五个 token 整组注入，文本不变 | 分类收益是否依赖图表示与当前节点对应 |

抽样与置换分别使用独立种子，默认均为 2026，不消耗解码 RNG。筛查在每个数据集固定均匀抽取 2048 个节点，按原指令行号排序；置换在这一批节点内部进行，保持抽样群体的 token 来源分布。全量阶段置换整个测试集。来源子图使用同一冻结 GNN/projector 重新前向，有限精度下不同 batching 可能有微小数值差异；保存实际输出以便检查。不得按真实类别挑选置换来源。

服务器在项目根目录运行（`models` 必须直接包含 `config.json`、tokenizer 和 Vicuna-7B-v1.5 模型分片；如果实际在子目录，修改 `LLM_DIR`）：

```bash
conda activate teaglm
cd ~/autodl-tmp/TEA-GLM
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export LLM_DIR="$PWD/models"
export RESULTS_DIR="$PWD/results/interventions_screen_$(date +%Y%m%d_%H%M%S)"

# 可选：只打印 12 条推理命令，不加载模型、不创建输出。
DRY_RUN=1 bash scripts/test_single.sh --interventions "$LLM_DIR" screen

# Children、History，各 2048 节点，六条件，seed=0，共 12 组。
BATCH_SIZE=4 SAMPLE_COUNT=2048 SAMPLE_SEED=2026 SHUFFLE_SEED=2026 SEEDS="0" \
  bash scripts/test_single.sh --interventions "$LLM_DIR" screen

# 独立重评分：读取原始回答和已核验抽样清单；应有 12 组，每组 2048 条。
python utils/utils.py --results-root "$RESULTS_DIR" --datasets children history \
  --policy unique --output-dir "$RESULTS_DIR/evaluation_unique"

# 配对诊断：差值 = 干预 - 同种子 normal；按干预前原图同质率分桶。
python scripts/analyze_results.py --results "$RESULTS_DIR" --datasets children history \
  --interventions --evaluation-policy unique --graph-mode required \
  --output "$RESULTS_DIR/diagnostics"
```

沿用已有实验的 batch=4、BF16、温度 0.7、采样解码、最多生成 80 token；top_p/top_k 继承基座配置，并记录最终值（已有正式实验是 0.6/50）。Children/History 输入上限 1100，Pubmed 700。不要在不同条件间更改 batch size。若某组中断，保留失败目录，另选新结果目录或单独使用新的 prefix；不会自动覆盖或把部分输出当作完成。运行日志逐组保存在 `logs/`，JSONL 和特征逐批刷新。

筛查结束后再固定全量条件。下面示例验证去邻居和置换；如果决定验证某个缩放系数，将对应条件名加入 `VARIANTS`。完整验证不是重新运行纯文本基线。

```bash
export RESULTS_DIR="$PWD/results/interventions_full_$(date +%Y%m%d_%H%M%S)"
unset SAMPLE_COUNT

# 示例：三数据集，正常/去邻居/置换，三个推理种子；共 27 组。
# 第一轮可设 SEEDS="0"，先验证九组，再用独立目录补后续种子。
DATASETS="children history pubmed" VARIANTS="normal self_only shuffle" \
  SEEDS="0 1 2" BATCH_SIZE=4 SAMPLE_SEED=2026 SHUFFLE_SEED=2026 \
  bash scripts/test_single.sh --interventions "$LLM_DIR" full

python utils/utils.py --results-root "$RESULTS_DIR" --datasets children history pubmed \
  --policy unique --output-dir "$RESULTS_DIR/evaluation_unique"
python scripts/analyze_results.py --results "$RESULTS_DIR" --datasets children history pubmed \
  --interventions --evaluation-policy unique --graph-mode required \
  --output "$RESULTS_DIR/diagnostics"
```

单项命令保留原用法，通过环境变量设置干预，例如去邻居；也支持 `GRAPH_INTERVENTION=self_only GRAPH_TOKEN_SCALE=1e-4` 的后续组合对照，但第一轮不要组合：

```bash
GRAPH_INTERVENTION=self_only GRAPH_TOKEN_SCALE=1 SEED=0 SAMPLE_COUNT=2048 \
  bash scripts/test_single.sh computer children 5 1100 self_only_seed0 "$LLM_DIR" \
  saved_model/first_model/TEA-GLM_ecom.pth
```

新增记录全部在各运行目录内：

- `sample_manifest.json`：完整指令 SHA256、原指令行号、node_id、抽样种子；`sample_index` 是当前子集内行号，`instruction_index` 是原完整指令行号。
- `intervention.json`：确切干预参数、实现定义及特征含义；shuffle 另保存 `token_permutation.json` 的目标/来源节点映射，预测 JSONL 也保留来源。
- `features_*.pt`：`X` 是目标节点原始特征；`H_GNN` 是目标子图的 GNN 输出（self-only 已去边）；`T_graph` 始终是实际注入的 token。非单位缩放保存 `T_graph_unscaled`；shuffle 保存目标原 token `T_graph_original`、来源 GNN 输出 `H_GNN_source`。这些额外张量同样逐批保存到可恢复的 feature_store。

诊断目录中，`report.html` 展示指标、图和配对结果；`summary.csv` 汇总每组指标；`intervention_summary.csv` 给出相对 normal 的差值及纠正/损害数；`intervention_seed_summary.csv` 汇总各推理种子的差值均值与样本标准差（单种子不伪造标准差）。每个 `paired_<条件>_seed*/` 保存逐节点、每类、规模组、同质率桶和类别×同质率表；`paired_class_details.csv` 含两组 Precision/Recall/F1 和预测占比。

分析会拒绝错误的抽样/置换指纹、节点错位、不匹配的模型权重/代码/依赖/解码参数、重复正常对照。干预比较中的 harmed 是“正常 TEA 答对、干预答错”，不是早期“Vicuna 答对、TEA 答错”的那批节点。常规分析入口不接受干预组或随机子集，必须明确加 `--interventions`。小样本可能没有某些类别，评测仍固定使用完整候选类别集合；Head/Medium/Tail 按原完整测试集定义。

筛查属于探索性诊断，不能把按测试集选择出来的缩放系数宣称为未经调参的最终泛化成绩。多推理种子仅衡量固定权重的生成波动，不等于多次独立训练。几何或单项干预改善不唯一证明过平滑、类别信息丢失或 token 功能冗余。

本次在临时隔离的 Python 3.9 / PyTorch 1.13.1 CPU / Transformers 4.28.0 / Accelerate 0.20.3 / PyG 2.6.1 环境验证：正常前向逐值一致、空边集自身分支递推、整组置换、FP32/BF16 缩放与实际注入对齐、正常训练梯度；六条件真实小型 Llama CLI 推理、随机子集独立评测、全流程诊断和 24 组 PNG/PDF、非零纠正/损害计数、错配与篡改拒绝。原八组正式结果仍全部通过只读核验。shell 语法、筛查 12 条/全量示例 27 条命令展开、Python 3.9 语法检查通过。完整 Vicuna-7B / CUDA 推理仍需在服务器执行；未改变本地原 conda 环境或原结果。

## Acknowledgements

Many previous works, such as [Vicuna](https://github.com/lm-sys/FastChat), [GraphGPT](https://github.com/HKUDS/GraphGPT), and [LLaGA](https://github.com/VITA-Group/LLaGA), have greatly inspired our work. The code is developed based on [InstructGLM](https://github.com/agiresearch/InstructGLM/tree/main). We are grateful for their wonderful contributions.


## Citation
```
@inproceedings{
wang2024llms,
title={{LLM}s as Zero-shot Graph Learners: Alignment of {GNN} Representations with {LLM} Token Embeddings},
author={Duo Wang and Yuan Zuo and Fengzhi Li and Junjie Wu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=32g9BWTndc}
}
```


