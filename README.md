# LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings

The official implementation of the NeurIPS 2024 paper, with logic fixes inside the author's original repository structure.

<img src="doc/framework.png" width="90%">

保留作者 GitHub master 提交 `6cb63667254aaeb9e73687f43ff0dbfdca0ef511` 的全部 30 个文件，文件名与数量不变。数据、指令、作者权重及论文另行保留。

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

单张 4090 上默认 batch size 2、累积 2 步，有效 batch size 4，对应论文双卡每卡 2 的全局 batch。梯度检查点使用旧版接口，可用 `--no-gradient_checkpointing` 关闭。单卡累积与双卡计算不逐位等价；24GB 显存占用仍需服务器实测。

原 GitHub 长度恢复到脚本中：训练 arxiv 700、computer 1000；测试 arxiv/pubmed 700、cora 850、children/computer/photo 1000、history 1300、sports 500。若样本超长，代码会明确报错，避免截掉图 token 或答案；`MAX_TEXT_LENGTH=2048` 可显式提高上限，但须记录为偏离原脚本长度设置。

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

按用户确定的常规口径评测全样本 Accuracy、真实类别集合上的 Macro F1（零分母取 0）；非法回答计错并计入真实类 FN，另报合法回答率。标签只作规范化精确匹配和明确的 arXiv/Cora 别名，不模糊猜测，不按合法回答过滤测试集。作者未提供论文统计代码，因此不能认定统计口径完全相同。

单独重算 NC 指标，不加载 LLM：

```bash
python utils/utils.py \
  --predictions results/cora/tea-citation_model_results.txt \
  --instructions instruction/cora/cora_dataset_test.json \
  --output results/cora/tea-citation_metrics.json
```

仅生成前 N 条时，离线评分也需 `--max_samples N`。

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

4090 / CUDA 11.7 / 完整 Vicuna-7B 全量实验仍需服务器执行。依赖一致不构成论文数值一致的保证。

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


