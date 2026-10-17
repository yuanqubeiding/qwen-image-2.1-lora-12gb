# 在 12 GB 显卡 + 32 GB 内存上训练 Qwen-Image-2.1 的 LoRA

**核心结论：不要用 CPU offload。用量化。**

**许可：Apache-2.0 —— 可商用，不用打招呼。**

这是一份实测记录 + 可复现配置。不是理论推导，是在一台 **12 GB 显存 / 31.7 GB 内存**
的机器上，用 10 张图跑了 500 步、拿到可用的 LoRA 之后的完整记录。

**两个数字是一个整体。** 这条配方针对的是「两头都小」的机器。如果只有显卡小、内存很宽裕，
CPU offload 其实是个可行的选项（慢，但能跑）——本配方仍然更快，但「offload 是错的」这句话
在那种机器上不成立。

---

## 为什么这件事值得单独写

Qwen-Image-2.1 是 20B 级别的 DiT，加上 text encoder 和 VAE，bf16 权重合计约 33 GB。
常规做法只有两条：

| 常规做法 | 在这台机器上的结果 |
|---|---|
| 直接训练 | 显存不够，加载就失败 |
| 开 CPU offload | **主机内存不够，加载阶段就 OOM** |
| 换更大显存 | 要花钱，而且很多人只有消费级卡 |

第三条路是本项目要说的：**把 transformer 和 text_encoder 量化到 NF4，权重常驻显存，offload 全关。**

它和"低显存就 offload"这条通行直觉是相反的。下面的实测说明为什么直觉在这里失效。

---

## 关键对照

| | 常规 offload 路线 | 本配方 |
|---|---|---|
| 权重精度 | bf16 | **NF4**（VAE 保持 bf16） |
| 显存常驻 | 小 | 权重约 9.4 GB，**实测显存增量 11,148 MB** |
| 主机内存峰值 | **~2× 模型大小（约 66 GB）** | 不再复制第二份，内存不参与 |
| 分页文件能否救 | **不能** | 不需要 |
| 12 GB 卡 + 32 GB 内存 | 加载阶段就失败 | 500 步跑完 |

分页文件不能救的原因：offload 会把每个参数复制一份到锁页内存（`pin_memory`），
锁页内存不允许换出，所以只有物理内存算数。而且这是**第二份分配**，峰值约为模型的两倍。

详见 [references/pitfalls.md](references/pitfalls.md)。

---

## 遇到慢 / 爆显存时的决策顺序

**按这个顺序，不要跳步：**

1. **量化大模块**（transformer、text_encoder）
2. **降低 `max_pixels`**
3. 还不行，才考虑 offload 或换卡 —— 而且第 3 步只有在**内存真的够**时才成立

实测中第 3 步在这台机器上从来不是解法。

---

## 适用面

| 你的机器 | 意味着什么 |
|---|---|
| 12–24 GB 显存，**32 GB 以上内存** | 就是实测的配置，照抄即可 |
| 12–24 GB 显存，**64 GB 以上内存** | 能跑；而且 offload 对你来说也是合法选项。本配方仍然更快（省掉每步 CPU→GPU 搬运） |
| 12 GB 显存，**不足 32 GB 内存** | **没验过。** 本配方的硬件下限就是实测的这台 31.7 GB。再往下，offload 同样会失败，第三条路只剩租卡 |
| 24 GB 以上显存、内存充足 | 可以直接 bf16 不量化（约 33 GB）训练，上面这些都用不上 |

反直觉的地方在于：**卡顿的元凶是分辨率，不是显存占用率。**

- `max_pixels = 1048576`（约 1024²）：显存 97–98%，每隔几步卡死 5 分钟
- `max_pixels = 262144`（约 512²）：显存**同样是 95–98%**，但 1.31–1.38 秒/步跑完 500 步，零卡顿（两次独立实测）

同样的占用率，完全不同的行为。所以"显存满了 → 上 offload"这个推理在这里会把你带进沟里。

---

## 实测环境

| 项 | 版本 |
|---|---|
| GPU | RTX 5070 Ti Laptop，11.94 GB 显存，sm_120（Blackwell） |
| 内存 | 31.7 GB |
| Python | 3.11.0 |
| torch | 2.10.0+cu130 |
| CUDA | 13.0 |
| bitsandbytes | 0.50.2 |
| DiffSynth-Studio / WebUI | 上游 main 分支，随附 commit 见 `assets/` |

以上版本是**跑通过**的那一组。torch 和 bitsandbytes 的版本组合比较敏感，
升版本前先看一眼 [references/pitfalls.md](references/pitfalls.md)。

---

## 安装

两条路，选一条即可。**命令行路径不需要 Node.js。**

### A. 命令行（推荐，云 GPU 也用这条）

```bash
git clone --recurse-submodules https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
pip install bitsandbytes --no-deps
```

两个坑：

1. **`--recurse-submodules` 不能省。** 普通 clone 会让 WebUI 子模块是空的，
   到时候报错会指向一个你根本没写过的文件。
2. **`bitsandbytes` 必须加 `--no-deps`。** 不加的话 pip 可能顺带换掉你的 torch，
   然后整个环境回到"编译不匹配"的状态。它的 `quant` 可选依赖组里还有
   `comfy-kitchen` / `entropack[cuda13]` / `torchao`，没有一个是训练必需的。

也可以用一键脚本（会 clone + 装包 + 下载权重）：

```bash
bash scripts/setup.sh
```

### B. WebUI（图形界面）

```bash
git clone --recurse-submodules https://github.com/modelscope/DiffSynth-WebUI.git
cd DiffSynth-WebUI
pip install -e .
pip install bitsandbytes --no-deps
powershell -ExecutionPolicy Bypass -File .\training_ui\launch_windows.ps1
```

打开 `http://127.0.0.1:8100/dashboard`。首次启动会编译前端，可能要几分钟。

点选路径见 [references/manual-webui.md](references/manual-webui.md)。

---

## 下载模型

权重 **31 GB**，第一次运行会自动下载。想先下好：

```bash
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir ./models/Qwen/Qwen-Image-2.1
export DIFFSYNTH_MODEL_BASE_PATH=./models
```

### ⚠ 每个 shell 都要设

这个变量**只从环境变量读，不从任何配置文件读**。WebUI 会自己设好，你手动跑的 CLI
不会继承它——它会回落到相对工作目录的默认位置，找不到模型，然后**默默重下 31 GB**，
耗时远比训练本身长。

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/路径/models              # 下面要有 Qwen/Qwen-Image-2.1/
du -sh "$DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image-2.1"   # 应该是 31G
```

看日志就能分辨：`xxx.safetensors: NN%` 是在下模型，`NN/100 [mm:ss<mm:ss, 1.3s/it]`
是在训练。跑了几分钟 GPU 还上不了 2%，停掉，别等——半截下载不会复用。

---

## 磁盘

| 项 | 大小 |
|---|---|
| 模型权重 | 31 GB |
| 依赖环境 | ~8 GB |
| 每个 epoch 的 LoRA | 167 MB |
| 数据集 | 小 |
| **建议预留** | **50 GB** |

---

## 数据

自备。格式是一个目录加一个 `metadata.jsonl`：

```json
{"image": "0001.jpg", "prompt": "<固定的风格描述前缀>: 画面内容描述。"}
{"image": "0002.jpg", "prompt": "<固定的风格描述前缀>: 画面内容描述。"}
```

**风格靠固定前缀承载，内容才进正文。** 这一条比步数、rank 都重要。
写法和挑选标准见 [references/dataset.md](references/dataset.md)。

本仓库不含任何数据集 —— 理由见同一篇的「权利」一节。

---

## 完整配置

| 参数 | 值 | 说明 |
|---|---|---|
| `quant_options` | transformer + text_encoder → `bitsandbytes_nf4` | VAE 不在列表里，保持 bf16 |
| `enable_model_cpu_offload` | **`false`** | 整个配方的核心 |
| `lora_base_model` | `dit` | 只训 DiT |
| `lora_rank` | `32` | 风格 LoRA 的合理默认 |
| `max_pixels` | `262144` | 约 512²。**卡顿的那个旋钮** |
| `dataset_repeat` | `10` | |
| `num_epochs` | `5` | 10 张图 → 500 步 |
| `learning_rate` | `1e-4` | AdamW |
| `use_gradient_checkpointing` | `true` | |
| `remove_prefix_in_ckpt` | `"pipe.dit."` | 去掉 pipeline 前缀 |
| 训练后采样 | **关闭** | 采样会额外加载完整 bf16 模型 → 12 GB 上必炸 |

完整命令：

```bash
bash scripts/train.sh <dataset-name>
```

或者直接看 [references/manual-cli.md](references/manual-cli.md) 的逐参数解释。

---

## 结果

| 项 | 实测 |
|---|---|
| 数据 | 10 张图 |
| 步数 | 500（10 repeat × 5 epoch） |
| 单步耗时 | 1.31–1.38 秒 |
| 总训练时间（含加载） | **12 分 04.7 秒** |
| 输出 | 5 个 `epoch-*.safetensors`，各 167 MB |
| 卡顿 | 0 次 |

### 峰值（2 秒采样，全程 355 个样本）

| 指标 | 峰值 | 时刻 | 稳态 |
|---|---|---|---|
| 训练进程 RSS | **20,444 MB** | **t=25 秒（加载阶段）** | 12,254 MB |
| 系统内存已用 | **28,700 / 32,485 MB（88.3%）** | **t=25 秒** | 19,702 MB（60.6%） |
| 可用内存最低 | **3,785 MB** | 加载阶段 | 约 12,700 MB |
| 页面文件占用 | **1,188 MB** | 全程 | 基线 1,179 MB |
| 显存占用 | **11,691 / 12,227 MB（95.6%）** | t=43 秒 | 11,616 MB |

这三条是关键：

1. **最紧的时候是加载，不是训练。** 内存峰值出现在前 25 秒，之后落到 60%。
   扛过加载就稳了——所以 **32 GB 是硬下限**，不是舒适线。
2. **页面文件全程没涨。** 1,178.9 → 1,188.5 MB，12 分钟只漂移了 9.6 MB。
   这就是「分页文件救不了」的实测证据，不再是推理。
3. **显存余量只有约 0.5 GB。** 不是宽裕，是贴着天花板。训练时任何其他吃显存的
   程序（浏览器里跑着别的模型、没关的 ComfyUI）都会直接变成 OOM。

### Loss 按 epoch

| epoch | 平均（A 次） | 平均（B 次） |
|---|---|---|
| 0 | 0.2498 | 0.2680 |
| 1 | 0.2656 | **0.2238** |
| 2 | **0.2245** | 0.2292 |
| 3 | 0.2294 | **0.2235** |
| 4 | 0.2292 | 0.2475 |

**别拿这张表挑 epoch。** 同一条命令、同一个数据集跑两次，A 次说 epoch 2 最低，
B 次说 epoch 3 最低，epoch 0 差 0.018。epoch 之间的差距**比两次运行之间的差距还小**，
所以排序是噪声。

这张表能支持的只有一条：**第一个 epoch 之后 loss 基本不再改善**，10 张图的数据集
跑 300 步就够，后 200 步主要是在增加僵硬感。至于用哪个 checkpoint，**看图像，不看 loss**。

---

## 四条实测结论

### 1. offload 在低内存机器上是死路，不是慢路

`pin_memory` 复制了一份模型到锁页内存，峰值约为模型的两倍，分页文件不参与。
bf16 需要约 66 GB，FP8 需要约 34 GB（差 2–3 GB），NF4 约 19 GB —— 能过。

**这是"跑不跑得动"的问题，不是"跑得快不快"。**

而且这一次不再只是推理：跑通的这次训练全程页面文件从 1,178.9 MB 走到 1,188.5 MB，
**12 分钟只涨了 9.6 MB**——根本没发生换页。所以分页文件再大也不参与，
因为它压根没被用上。详见 pitfalls §1。

### 2. NF4 和 offload 不能同时开

两个都开会加载成功，然后在第一次前向时死在
`RuntimeError: invalid argument to getCurrentStream`。
但 `bnb.nn.Linear4bit` 单独在 sm_120 上测试是正常的 —— 所以这不是硬件不支持，
是 offload 链路的问题。详见 pitfalls §2。

### 3. 卡顿由分辨率触发，不由显存百分比触发

见上面「决策顺序」一节。这个坑最贵，因为它会把你推向 1 和 2 这两条死路。

### 4. checkpoint 不要改名

输出里的 `lora_A.default.weight` 看起来像 PEFT 的内部命名，其实 ComfyUI
有专门的分支处理它（`comfy/weight_adapter/lora.py:166`），原样放进去就能加载。
改名反而可能造出它不认识的 key。

验证方法：看控制台的 `192 patches attached`（32 个 block × 6 个模块，其中
`gate_layer` 和 `proj` 融合成一个 `gate_up`）。是 `0` 就是没生效。

---

## 在 ComfyUI 里用

把 `epoch-N.safetensors` 原样丢进 `ComfyUI/models/loras/`。

起点：`strength_model = 1.5`，`resolution = 1024`，提示词带上训练时的风格前缀。

强度 2.0 风格最强但会掉色；1.8–1.9 反而更容易出现解剖错误 —— 原因在
[references/comfyui.md](references/comfyui.md) §3。

---

## 这个仓库里没有什么

- **没有数据集** —— 训练数据的来源和授权由你自己负责
- **没有 LoRA 权重** —— 同上的原因
- **没有模型权重** —— 31 GB，从这里拿：`Qwen/Qwen-Image-2.1`
- **没有 ComfyUI** —— 这是上游项目

---

## 引用与许可

| 组件 | 许可 | 可商用 |
|---|---|---|
| **本仓库**（文本 + 脚本） | **Apache-2.0**，见 [LICENSE](LICENSE) | **可以，不用问** |
| DiffSynth-Studio | Apache-2.0 | 可以 |
| DiffSynth-WebUI | Apache-2.0 | 可以 |
| ComfyUI | GPL-3.0（仅按 file:line 引用，未收录源码） | 按 GPL-3.0 |
| Qwen-Image-2.1 权重 | Qwen Research License | **不可以，需单独许可** |

第三方署名见 [NOTICE.txt](NOTICE.txt)。

**本仓库：Apache-2.0，可以商用。** 拿去卖课、嵌进产品、改了自己发，都不用问。
唯一要做的只有 Apache 的例行公事：留着 LICENSE 和 NOTICE，说明你改了什么。
赚到钱了愿意回链接最好，不回也不违约。

**下面这段限制来自 Qwen 的许可，不是我们的。** 我们不拥有那套权重，也就无权授权
它的商用。许可原文是这么写的：

1. **授权范围只有「研究或评估」。** §1(i) 把 "Non-Commercial" 定义成
   "for research or evaluation purposes only" —— **比日常说的「非商用」更窄**。
   §2(a) 授予的使用、复制、分发、创作衍生作品、修改，全部限定在这个范围内。
   §2(b)：任何商业用途都要单独许可，申请地址 model-business@notice.qwencloud.com。
2. **你训出来的东西继承这个限制。** §5(a) 说你对自己创作的衍生作品享有所有权，
   但前提是 "conditioned upon compliance"——遵循本协议。所以用这套权重训出的
   LoRA 同样只能用于研究/评估；卖它、或者放进商业产品，需要商业许可。
3. **生成图片没有被单独豁免。** 协议里没有任何一条把输出物的权利授予你；出图属于
   "use of the Materials"，而这项授权只覆盖研究/评估。**不要把生成结果当成不受
   限制**——保守做法是同样按研究/评估对待。
4. **分发训好的模型要署名。** §4(b)：训练或微调出模型并分发时，产品文档要写
   "Built with Qwen" 或 "Improved using Qwen"。**免费分发（研究/评估用途）是允许的**
   （§3 允许再分发），商业使用不允许。
5. **"Qwen" 不能当衍生产品的主名称。** §4(c)。描述性用法（如"Qwen-Image-2.1 的
   LoRA"）是明确允许的。

本仓库只记录*怎么训练*，不分发权重，也不分发任何 LoRA。

---

## 文件索引

| 文件 | 内容 |
|---|---|
| [SKILL.md](SKILL.md) | 给 AI 助手读的版本 |
| [PROVENANCE.md](PROVENANCE.md) | 数据来源：两次运行、测量方法、哪些是实测哪些是推算 |
| [NOTICE.txt](NOTICE.txt) | 第三方署名 |
| [scripts/measure.py](scripts/measure.py) | 峰值采样器（跑任何命令都行） |
| [references/setup.md](references/setup.md) | 环境、依赖、验证命令 |
| [references/manual-cli.md](references/manual-cli.md) | 命令行逐参数说明 |
| [references/manual-webui.md](references/manual-webui.md) | 界面点选路径 |
| [references/pitfalls.md](references/pitfalls.md) | 四个坑的机理 |
| [references/comfyui.md](references/comfyui.md) | 加载、验证、强度调法 |
| [references/dataset.md](references/dataset.md) | 格式、文案、筛选 |
| [assets/training_args.json](assets/training_args.json) | 那份跑通的配置（已脱敏） |
| [assets/loss.csv](assets/loss.csv) | 500 步的原始 loss（A 次） |
| [assets/measured_summary.json](assets/measured_summary.json) | 峰值：内存 / 页面文件 / 显存（B 次） |
| [assets/measured_resources.csv](assets/measured_resources.csv) | 每 2 秒一个采样，共 355 个（B 次） |
| [assets/measured_loss.csv](assets/measured_loss.csv) | 500 步的原始 loss（B 次） |
