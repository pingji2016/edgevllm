# 脚本与复现

这 5 个脚本对应正文里的每一张实测表。它们都是**独立可跑**的：不依赖任何仓库外的代码，
只需要 ffmpeg（带 librubberband）和几个 pip 包。

## 环境

本机实测环境（数字都出自这里）：

| 项 | 值 |
| --- | --- |
| 系统 | Windows 11 Pro 26100 |
| Python | 3.12.2 |
| numpy / scipy | 1.x / 1.x（numpy 走的是 `AppData\Roaming\Python\Python312` 那个用户目录） |
| mediapipe | 1.1.0（**只有 Tasks API，`mp.solutions` 已删除**） |
| onnxruntime | 1.20.1，CPU Execution Provider |
| onnx | 1.23.2（只为从 onnx 里把 emap 抠出来） |
| opencv-python | 4.x |
| ffmpeg | 带 `librubberband`（变调）和 `libmp3lame`（出样音） |
| GPU | RTX 5060 Ti 16G —— **本目录的实验全部没用它**，全 CPU |

```bash
pip install numpy scipy opencv-python mediapipe onnx onnxruntime
```

## 素材

合成素材自带生成器，不需要下载任何真人数据：

```bash
python make_test_face.py face.png                 # 程序化合成一张能被检测器认出的人脸
```

人脸模型（只下这一个就够跑滤镜）：

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

换脸还需要两个权重（**只授权非商业研究用途**）：

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `inswapper_128.onnx`（554 MB） | 换脸网络本身 | 社区镜像，如 HF 上的 `ezioruan/inswapper_128.onnx` |
| `w600k_r50.onnx`（174 MB） | ArcFace 身份编码器 | insightface `buffalo_l` 包内的识别模型 |

> huggingface.co 在部分网络下不通，`HF_ENDPOINT=https://hf-mirror.com` 可用。

语音素材用系统 TTS 生成即可（本机用的是 SAPI 的「Microsoft Huihui Desktop」，
一段 9.47 s 的中文），不需要真人录音。

## 脚本清单

| 脚本 | 做什么 | 正文 |
| --- | --- | --- |
| [make_test_face.py](make_test_face.py) | 生成合成人脸 | 素材准备 |
| [face_landmark_filter.py](face_landmark_filter.py) | 478 点 + 表情系数，量「全图变形 vs ROI 变形」 | [01-face-filters.md](../01-face-filters.md) |
| [face_swap_pipeline.py](face_swap_pipeline.py) | 换脸全流水线，逐阶段计时（自换脸 smoke test） | [02-face-swap.md](../02-face-swap.md) |
| [dsp_formant_synthetic.py](dsp_formant_synthetic.py) | 合成元音 + LPC 真值校验 + 倒谱包络移植 | [03-voice-pitch-and-formant.md](../03-voice-pitch-and-formant.md) |
| [dsp_pitch_formant.py](dsp_pitch_formant.py) | 真实语音上量 F0 比与包络搬移量 α | [03-voice-pitch-and-formant.md](../03-voice-pitch-and-formant.md) |
| [realtime_chunking.py](realtime_chunking.py) | 分块拼接的伪影 + 块长 ↔ 延迟/RTF | [05-realtime-engineering.md](../05-realtime-engineering.md) |

## 运行

```bash
# 人脸滤镜：检测 10.3ms / 全图变形 97.6ms / ROI 变形 7.4ms
python face_landmark_filter.py face.png face_landmarker.task out/

# 换脸：自换脸（源 = 目标），逐阶段耗时
python face_swap_pipeline.py face.png models/ out/

# 合成元音：真值 700/1200/2500 Hz，对比 rubberband 的两种 formant 模式
python dsp_formant_synthetic.py out_synth/

# 真实语音：F0 比 + 包络搬移量 α
python dsp_pitch_formant.py speech_zh.wav out_var/

# 分块：接缝伪影 + 块长 ↔ 延迟/RTF
python realtime_chunking.py speech_zh.wav out_chunk/
```

## 本机踩到的坑

都是会重复踩的，记下来：

1. **MediaPipe 1.1.0 删掉了 `mp.solutions`。** 网上绝大多数人脸教程还在用
   `mp.solutions.face_mesh.FaceMesh(...)`，在这个版本会直接 `AttributeError`。
   现在只有 Tasks API（`mediapipe.tasks.python.vision`），而且输入要的是
   `mp.Image` 而不是 numpy 数组。
2. **`/tmp/...` 这类路径只在 bash 命令行参数里被自动转换。** 写在 heredoc 或
   Python 源码里的 `/tmp/x` Windows 上的 Python 不认识 —— 要么用 `C:\...`，要么
   让路径从 `sys.argv` 进来。
3. **双引号里套 ASCII 双引号 = 语法错误。** 中文注释里想引用词语，用「」而不是 `"`。
   用脚本重写文件时统一 `io.open(..., newline="\n")`，免得平台间换行符漂移。
4. **换脸时 `latent` 必须归一化到模长 1。** 身份向量本身模长在 20 上下，直接拿去做
   emap 投影会让 AdaIN 的风格强度放大 20 倍 —— 贴回去是一张花脸。正确顺序是
   「先单位化 → 投影 → 再单位化」。
5. **倒谱包络的 lifter 必须短于谐波间距（`sr/F0`）。** 取长了谐波梳会留在包络里，
   量出来的「包络」其实跟着音高走，于是 `formant=preserved` 和 `shifted` 量出同一个数。
