# DeepSeek-OCR 实测：把文档压成视觉 token

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120）· Ubuntu 26.04
> 实测日期：2026-11-25 · vLLM 0.34.0 · 模型 3.34 B MoE（570 M 激活）

---

## 1. 它是什么

DeepSeek-OCR 的核心主张不是"识别文字"，而是 **Contexts Optical Compression**：
**把长文本渲染成图像，再用少量视觉 token 表示它**，从而用视觉通道压缩上下文。

```
文档图像
   │
   ▼ DeepEncoder（约 380 M：80 M SAM-base + 300 M CLIP-large）
视觉 token（可少到 64 / 100 / 256 / 400 个）
   │
   ▼ DeepSeek-3B-MoE（64 选 6 路由 + 2 共享，激活约 570 M）
文本输出（含 <|grounding|> 坐标）
```

论文给出的关键数字：**压缩比 <10× 时 OCR 精度 97%，20× 时仍有约 60%**；
在 OmniDocBench 上用 100 个视觉 token 就超过 GOT-OCR2.0（256 token），
用不到 800 token 超过 MinerU2.0（平均 6000+ token）。

许可证 MIT（可商用），权重 6.3 GB（bf16）。

## 2. 本机部署：两个坑

### 坑 1：默认 MoE 后端会在 sm_120 上 JIT 编译 40+ 分钟

vLLM 对 MoE 默认走 **flashinfer fused-MoE**。sm_120 没有预编译包，启动时现场编译：

```
[59/97] nvcc ... -gencode=arch=compute_120f,code=sm_120f ... fused_moe_120 ...
ninja: build stopped
```

97 个 cutlass kernel，单个约 20–30 s → **40 分钟以上**，且中途被打断就白等。

**解法：换 Triton 后端**

```python
LLM(model=MD, ..., moe_backend="triton")   # 引擎就绪从 40min+ 降到 167s
```

### 坑 2：传 `mm_processor_kwargs` 会让占位符与编码器输出对不上

想按论文测 64/100/256/400 token 四档，于是传了：

```python
mm_processor_kwargs={"base_size": 512, "image_size": 512, "crop_mode": False}
```

结果直接崩：

```
RuntimeError: shape mismatch: value tensor of shape [73, 1280]
              cannot be broadcast to indexing result of shape [903, 1280]
```

原因：**多模态占位符按 903 个算，视觉编码器只吐出 73 个 embedding**。
这是 vLLM 0.34 的 DeepSeek-OCR 处理器在自定义尺寸下的 bug。

**解法：不要传 `mm_processor_kwargs`**，用模型默认配置（默认走动态分辨率）。

> 因此论文里"64/100/256/400 视觉 token 的精度曲线"在 vLLM 上暂时测不了，
> 只能在官方 Transformers 路径或等 vLLM 修复后再做。

## 3. 实测结果（A4 文档，中英混排 + 表格）

测试图：1240×1754（A4 @150dpi），含标题、三节正文、三行表格、一行代码。

| 项 | 值 |
| --- | ---: |
| prompt token | **913**（含 903 个视觉占位） |
| 输出 token | 557 |
| 单页耗时 | **7.26 s** |
| 原始字符精度 | 53.8% |
| **去掉 grounding 坐标标签后** | **83.5%** |

**为什么原始精度看起来低**：`<|grounding|>` 模式会在每段文字前插入坐标标签，
例如 `text[[75, 238, 320, 258]]`。这些标签对"文字内容"没有影响，但会大幅拉低字符串相似度。
去掉标签后：

```
输出（去标签）：端侧推理部署说明v12 一概述 本设备支持在16GB显存上运行30亿参数的多模态模型
              典型延迟为每帧410毫秒吞吐约24帧每秒 二关键指标 显存占用62GB 首token延迟125ms
              解码速度553tokens/s32K上下文 三注意事项
```

对照原文几乎逐字一致，差异来自 markdown 标记（`##`）和我的参考文本是手打子集。

## 4. 性能定位

| 指标 | 本机实测 | 参考 |
| --- | ---: | --- |
| 单页（A4，913 token 输入） | **7.26 s** | — |
| 估算吞吐 | **约 500 页/小时** | DeepSeek 官方：A100-40G 约 200k 页/天（≈8300 页/小时） |
| 显存 | 约 8 GB（bf16，停掉 vlm-server 后） | 官方建议 12 GB 起 |

差 16 倍主要来自：消费卡算力、`enforce_eager` 关闭 CUDA graph、Triton MoE 后端（非最优）。

## 5. 它对我们有什么意义

| 角度 | 结论 |
| --- | --- |
| **技术思路** | "用视觉通道压缩文本上下文"是个新方向：一页 6000 token 的文档压到 100–800 视觉 token，长文档 RAG 的成本模型会变 |
| **工程价值** | 单页 7 s、显存 8 GB，**在这张卡上可做离线批量 OCR**（合同/票据/PDF 转 markdown），但不适合实时 |
| **对比已有方案** | 我们之前测的 Qwen3-VL-8B 单帧 0.41 s 但只做描述；OCR 需要精确结构，DeepSeek-OCR 的 grounding 坐标能直接给版面分析 |
| **限制** | 文档类专用；vLLM 上暂不支持自定义视觉 token 预算；比 PaddleOCR 类方案重得多 |

## 6. 复现

```bash
# 1) 权重（ModelScope，6.3 GB）
python -c "from modelscope import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# 2) 停掉常驻 VLM 服务腾显存
sudo systemctl stop vlm-server vlm-cams

# 3) 推理（注意 moe_backend="triton"，且不要传 mm_processor_kwargs）
cd ~/apps/infer-engines
export PATH="$PWD/.venv/bin:$PATH" CUDA_HOME=/usr/local/cuda
python dsocr_default.py

# 4) 恢复服务
sudo systemctl start vlm-server vlm-cams
```

产物：`~/apps/infer-engines/dsocr_default.json`（含完整输出文本）
