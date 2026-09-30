# Mage-VL 视频链路在 K3 上跑通 + ggml f16 漏洞修复

> 板子 pingji-k3 · riscv64 · 2026-09-30

## 结论

Mage-VL 的**视频理解链路已在 K3 上端到端跑通**，输出 StreamMind 的逐帧 silent/speak 决策：

```json
{"segment":0,"frame":65,"seconds":2.708,"silent_logit":-0.0959,"speak_logit":0.0435,"speak_probability":0.5348,"decision":"speak"}
{"segment":0,"frame":68,"seconds":2.833,"silent_logit":-0.0719,"speak_logit":0.0106,"speak_probability":0.5206,"decision":"speak"}
...
{"event":"suppressed","segment":0,"peak_probability":0.9258,"vision_encode_count":4}
STREAMMIND_TIMING {"segment":0,"timestamps":55,"container_ms":26.6,"vision_ms":68690.2,"epfe_ms":176.6,"classifier_ms":2230.4,"native_total_ms":71145.7}
```

（3 秒片段 → 16 个 codec-aware canvas）

## 全链路（全部 riscv64 从源码构建）

```
raw mp4
  └─ cv-preinfer            # codec-aware canvas 生成（补丁版 FFmpeg + cv_reader_fast）
       └─ pack_mage_codec   # 打包成 MAGECV1 (.mcv)
            └─ llama-streammind-e2e   # Mage-ViT 编码 + StreamMind 门控（Mamba EPFE + 分类器）
```

## 修复的漏洞：ggml f16 binary op

**文件**：`llama.cpp-sm/ggml/src/ggml-cpu/binary-ops.cpp` 的 `binary_op<>` 分派函数。

**现象**：门控 decode 时 abort：
```
ggml/src/ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types: dst: f32, src0: f32, src1: f16
ggml_abort -> ggml_compute_forward_mul
```

**原因**：该分派只支持以下组合——
`(F32,F32,F32)`、`(F16,F16,F16)`、`(BF16,BF16,BF16)`、`(BF16,F32,BF16)`、`(BF16,F32,F32)`、`(F16,F32,F16)`、`(F16,F32,F32)`，
**漏了 `(dst=F32, src0=F32, src1=F16)`**（第二个操作数是 f16，第一/输出是 f32）。门控图里有一个 f16 张量与 f32 直接相乘，于是命中 `else -> GGML_ABORT`。

**修改**：在最后的 `else { GGML_ABORT(...) }` 之前插入一个分支：

```cpp
    } else if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, float, ggml_fp16_t, float>(params, dst);
    } else {
```

模板参数顺序是 `<op, src0_t, src1_t, dst_t>`，与已有分支一致（例如 `(F16,F32,F32)` 对应 `<op, ggml_fp16_t, float, float>`），所以 `(F32,F16,F32)` 就是 `<op, float, ggml_fp16_t, float>`。

**重编**：
```bash
cd ~/mage-vl/llama.cpp-sm
cmake --build build --target llama-streammind-e2e -j8
```
重跑即通过（上面 JSON 就是修复后的真实输出）。

## 复现命令

```bash
# 1) 视频 → codec-aware canvas（尺寸必须能被 32 整除）
cd ~/mage-vl/cvp && export LD_LIBRARY_PATH=$PWD/src/codec_video_prep/libs
cv-preinfer --video ~/mage-vl/models/soccer3s.mp4 --out_dir ~/mage-vl/vp4 \
  --patch 16 --max_pixels 60000 --target_canvas 4 --images_per_group 4
# 2) 打包成 MAGECV1
cd ~/mage-vl && python3 src/tools/pack_mage_codec.py vp4 soccer3s.mcv
# 3) 视频运行时
export LD_LIBRARY_PATH=$PWD/llama.cpp-sm/build/bin
./llama.cpp-sm/build/bin/llama-streammind-e2e \
  models/mage-vl-backbone-Q4_K_M.gguf models/mage-vit-mmproj-Q8_0.gguf \
  models/mage-streammind-epfe-Q8_0.gguf models/mage-streammind-cls-Q8_0.gguf soccer3s.mcv
```

## 性能

3 秒 / 16 canvas 片段：`vision_ms=68.7s`（Mage-ViT 编码，瓶颈）、`epfe_ms=0.18s`、`classifier_ms=2.2s`。
→ 视频"能跑通"，但 CPU 上**远非实时**（视觉编码最慢）。
