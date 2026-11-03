# K3 上跑 Laya 贪吃蛇（AI Core）：110 ms/步，以及三个装不上的坑

> 【实测】2026-11-03，`pingji-k3` · Bianbu 4.0.1 · 内核 `6.18.3-generic` ·
> Python 3.14.4 · `python3-spacemit-ort` **2.0.6+1**（ORT `1.24.2+spacemit.a1`）·
> `spacemit-tcm` **3.0.1** · 15 GiB 内存 · 无 swap。
>
> 项目：[`JiongLab/laya-snake-k3`](https://github.com/JiongLab/laya-snake-k3) v0.1.0，
> 官方帖子见 [t/1664](https://forum.spacemit.com/t/1664)。

---

## 0. 一句话结论

| 问题 | 结论 |
| --- | --- |
| 能跑吗 | **能**，600 步无干预、无死亡，得分 37 |
| 多快 | **110.6 ms/步（8.79 步/秒）**，峰值 RSS **1.91 GiB**，平均 CPU **164%**（单核=100%） |
| 和官方数字比 | 官方 110 ms / 1.88 GiB —— **实测完全吻合** |
| 能用 X100 跑吗 | **不能**：EP 的 `INTRA_THREAD_AFFINITY` 只接受 8–15，写 `0` 直接报错 |
| 最大的坑 | deb 把依赖版本**写死**了；`setup.sh` 的 `pip check` 失败会让**模型下载静默跳过** |
| 性能最大的干扰 | 同时跑任何 CPU 推理（哪怕在 X100 上）→ **305 ms/步（慢 2.8×）** |

---

## 1. 安装：deb 的版本钉死 + setup 静默中断

### 1.1 deb 依赖要求「精确等于」，而板上都高一个修订版

```bash
$ sudo apt install ./laya-snake-k3_0.1.0_riscv64.deb
 laya-snake-k3 : 依赖: python3-spacemit-ort (= 2.0.6)
                 但无法安装它
```

板上实际版本：

| 包 | deb 要求 | 板上实际 |
| --- | --- | --- |
| `python3-spacemit-ort` | `= 2.0.6` | **2.0.6+1** |
| `spacemit-onnxruntime` | `= 2.0.6` | **2.0.6+1** |
| `spacemit-tcm` | `= 3.0.0+5` | **3.0.1** |
| `python3-torch` | `= 2.9.1+dfsg-1~exp1ubuntu2` | 一致 ✅ |

`+1` 只是重建号，API 相同。为了**不改动系统包**，我直接解包运行：

```bash
dpkg-deb -x laya-snake-k3_0.1.0_riscv64.deb ~/laya-app
# 启动脚本用相对路径，解包后可直接跑
```

`setup.sh` 里同样有 `dpkg-query` 的精确比对，把解包副本里的那几行放宽：

```bash
case "$got" in "$want"|"$want"+*) : ;; *) echo "WARN: $entry -> $got" >&2 ;; esac
```

### 1.2 `pip check` 失败 → 模型下载被跳过（最坑的一处）

`setup.sh` 装了 `set -eu`，venv 建好之后执行：

```bash
PYTHONPATH=/usr/lib/python3.14/dist-packages "$LAYA_K3_HOME/venv/bin/python" -m pip check
```

板上系统包元数据不全（`laya 0.3.4 requires safetensors`、`pynacl requires cffi` 等），
`pip check` 返回 1 → **脚本直接退出**，`models.py` 根本没执行，日志里只有一堆 pip 警告。

表现：`~/.local/share/laya-snake-k3/venv` 有了，`models/` 是空的，脚本却"看起来跑完了"。

手动补下载（模型清单锁定在 `models.lock.json`，共 **1.33 GB**）：

```bash
python3 ~/laya-app/usr/share/laya-snake-k3/scripts/models.py \
  --directory ~/.local/share/laya-snake-k3/models
```

下载完成后的目录：

```
models/multilingual/                    647 MB  (safetensors + tokenizer)
models/multilingual-onnx-comparison/    581 MB  (model.onnx, FP16)
```

---

## 2. 只能跑在 A100 上（X100 会被 EP 拒绝）

绑核不是 Python 层做的，而是写死在 `snake_k3/onnx_backend.py`：

```python
affinity = '8;10;12;14' if threads == 4 else ';'.join(str(i) for i in range(8, 8+threads))
ep_options = {'SPACEMIT_EP_INTRA_THREAD_NUM': str(threads),
              'SPACEMIT_EP_INTRA_THREAD_AFFINITY': affinity, ...}
```

把 `8` 改成 `0` 想让它跑 X100，结果是**初始化就报错**：

```
EP Error Invalid intra_thread_affinity id 0. when using ['SpaceMITExecutionProvider']
Falling back to ['CPUExecutionProvider'] and retrying.
RuntimeError: AI Core provider failed to initialize; refusing silent CPU-only fallback
```

也就是说：**SpaceMIT EP 的线程亲和性只接受 8–15（A100）**，
"X100 vs A100" 在这个应用上没法做对照——不是慢，是 EP 不给开。

> 副作用：EP 还会把一批算子丢回 CPU（启动日志里 `SpineOpManager` 逐个报
> `operator [Transpose]/[Concat]/[Div]/[Erf]/[Where]/[IsNaN] has been disabled`），
> 所以这个 110 ms 已经是 "A100 + CPU 混合" 的结果。

---

## 3. 实测：600 步稳态

```bash
export LAYA_K3_HOME=$HOME/.local/share/laya-snake-k3
sh ~/laya-app/usr/share/laya-snake-k3/start-snake.sh \
   --headless --steps 600 --record /tmp/laya600.jsonl
```

默认参数：`--width 12 --height 8 --max-speed --backend spacemit`（4 线程，绑 8/10/12/14）。

| 指标 | 600 步全程 | 后 80%（480 步） |
| --- | ---: | ---: |
| `inference_ms` 均值 | 113.1 | **110.6** |
| 中位数 | 110.4 | **110.2** |
| p95 | 116.1 | **116.0** |
| 步/秒 | 8.79 | — |
| 得分 / 长度 | 37 / 43 | — |

资源（每 2 s 采样）：

| 项 | 值 |
| --- | ---: |
| 峰值 RSS | **1,957 MiB（1.91 GiB）** |
| 平均 RSS | 1,676 MiB |
| 平均 CPU | **164%**（4 个 AI 线程 + 少量 CPU 算子） |
| 温度 | 62–66 °C |

对比官方帖子：

| 指标 | 官方（v0.1.0, 1200 步） | 本次（600 步） |
| --- | ---: | ---: |
| 单步推理 | 110 ms | **110.6 ms** ✅ |
| 稳定 RSS | 1,927.61 MiB | **1,957 MiB** ✅ |
| 平均 CPU | 246.32% | 164%（采样口径不同）|

**结论：官方数字可复现，误差 < 2%。**

---

## 4. 干扰实验：别的推理一开，直接慢 2.8 倍

同一份 600 步，在后台同时跑 `llama-perplexity`（Qwen3-4B，8 线程绑 X100 0–7）：

| 场景 | `inference_ms` 均值 | 步/秒 | 相对 |
| --- | ---: | ---: | ---: |
| 独占 | **110.6** | 8.79 | 1.00× |
| 与 X100 上的 perplexity 并发 | **305.6** | 3.27 | **2.76× 慢** |

注意：Laya 跑在 A100（8/10/12/14），perplexity 跑在 X100（0–7），**核不重叠**，
但两者都吃 DDR 带宽 —— 这类小模型推理是**内存带宽敏感**的，
"核没抢到" 不代表 "不受影响"。

> 所以测这类数字时，**板上不要同时跑任何模型**；我第一次测出 210 ms / 305 ms
> 都是被这件事污染的。

---

## 5. 复现清单

```bash
# 1) 取代码与包
git clone --depth 1 https://github.com/JiongLab/laya-snake-k3.git
# 从 Releases 下 laya-snake-k3_0.1.0_riscv64.deb（14.5 MB）

# 2) 解包（不改系统包）
dpkg-deb -x laya-snake-k3_0.1.0_riscv64.deb ~/laya-app

# 3) 建 venv + 装 wheelhouse（离线）
sh ~/laya-app/usr/share/laya-snake-k3/setup.sh --skip-models

# 4) 手动下模型（1.33 GB）
python3 ~/laya-app/usr/share/laya-snake-k3/scripts/models.py \
  --directory ~/.local/share/laya-snake-k3/models

# 5) 跑
LAYA_K3_HOME=$HOME/.local/share/laya-snake-k3 \
  sh ~/laya-app/usr/share/laya-snake-k3/start-snake.sh --headless --steps 600
```

---

## 6. 给上游的三条建议

1. **依赖别用 `=`**：`python3-spacemit-ort (>= 2.0.6)` 就够了，`+1` 这种重建号不该挡安装。
2. **`pip check` 不要放在 `set -e` 流水线里**，或者改成只告警；现在会让模型下载静默跳过，
   且退出码是 0 的话用户根本看不出。
3. **报错说清楚"X100 不支持"**：`Invalid intra_thread_affinity id 0` 太隐晦，
   建议直接提示 "AI Core EP 仅支持 A100 (8–15)"。
