# WhisperLive 学习笔记

把一个**批处理**语音模型改造成**近实时**字幕服务的完整拆解。

Whisper 本身只吃固定长度（标称 30 秒）音频、吐完整结果；WhisperLive 不改模型，
只在服务端加一层"流式化包装"，就把它变成了边收边出字幕的 WebSocket 服务。
这套包装里全是工程取舍——滚动窗口、双游标、重复输出检测——值得逐条读。

## 目录

| 文档 | 内容 |
| --- | --- |
| [QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md) | 项目说明与运行指南：组件一览、通信协议、安装（含 CUDA 版 torch）、服务端/客户端参数表、浏览器插件、Docker、FAQ |
| [DESIGN.zh-CN.md](DESIGN.zh-CN.md) | 处理流程与设计要点：端到端数据流、服务端循环、13 条特殊处理逐条拆解、两个后端差异、调优旋钮、代码粗糙处、走读顺序 |

## 阅读顺序建议

1. 先看 **QUICKSTART** 把服务端和客户端跑起来——没有跑通过就读设计，很多现象（字幕抖动、
   延迟累积）会没有体感。
2. 再看 **DESIGN 第 1~3 章**建立数据流的骨架：客户端推 float32 帧 → 缓冲区 → 识别线程 → 回推 JSON。
3. 然后重点读 **DESIGN 第 4 章**，这是全文的核心。尤其 4.2「为什么最后一段不算数」和
   4.3「重复输出检测」——这两条是"近实时"与"字幕不抖动"能同时成立的真正原因。
4. 最后按 **DESIGN 第 8 章**的顺序走读源码，核心部分不到 400 行。

## 贯穿两份文档的结论

**1. 模型不改，包装来凑。** 实时性技巧全部落在 `server.py` 的 `ServeClientBase` 子类里，
`transcriber.py` 只是 faster-whisper 的一个可调用封装。想改行为，改包装层，别动模型。

**2. 延迟的来源不是推理速度，而是"什么时候敢定稿"。** 每积累约 1 秒就把「尚未定稿的全部音频」
再喂一次模型，但只有确认说完了的段才落库。所以调延迟本质是在调 `same_output_threshold`
这类"等多久算说完"的阈值，而不是调模型。

**3. 流式服务的真正难点是边界情况，不是主流程。** 主流程只有几十行；篇幅最大的是静音幻觉、
时间轴落后、缓冲区裁剪、半句话被切断这些异常路径。读流式系统，异常路径才是主体。

## 关键路径速查

```
服务端入口      run_server.py                        命令行参数 → 启动 WebSocket 服务
连接与协议      whisper_live/server.py:225           首条 JSON 配置 → SERVER_READY → 二进制帧
缓冲与双游标    whisper_live/server.py:407           ServeClientBase：frames_offset / timestamp_offset
识别主循环      whisper_live/server.py:979           speech_to_text → transcriber.transcribe
定稿逻辑        whisper_live/server.py:1045          update_segments（全项目最关键的一段）
结果回推        whisper_live/server.py:544           send_transcription_to_client（只回最近 10 段）
模型封装        whisper_live/transcriber.py          faster-whisper 的 vendored 副本（有两处改动）
客户端          whisper_live/client.py               TranscriptionClient / Client
重采样与 SRT    whisper_live/utils.py                PyAV 转 16kHz、生成 .srt
VAD             whisper_live/vad.py                  Silero VAD（ONNX），TensorRT 后端用
```

代码引用一律是 `路径:行号`，对应上面提到的上游版本。

## 来源与本地对照

这两份文档是**本地整理的学习笔记，不是上游 Collabora/WhisperLive 的内容**。
底本是上游 `v0.6.2`（tag，commit `379bd146fc0bcf64b2c464c6568bbe47a74a8b80`）。

文中所有 `whisper_live/server.py#L1007` 形式的引用**都已改写为上游 `v0.6.2` 的永久链接**
（目录用 `tree/`、文件用 `blob/`）。这样做的原因：原文用的是 `../whisper_live/server.py`
这类相对仓库根的路径，搬进本仓库后会全部失效，而"所有结论都标了出处、可对照代码验证"
正是这两份文档的价值所在。

要在本地对照源码（同样是 v0.6.2）：

```bash
git clone --depth 1 --branch v0.6.2 \
    https://github.com/collabora/WhisperLive.git third_party/WhisperLive
```

`third_party/` 已在 [.gitignore](../../.gitignore) 中，不会污染 `git status`；
本地拉取而非纳入版本控制的理由见仓库根 [README](../../README.md) 的「第三方依赖」一节。

## 环境提示

上游要求 Python 3.8–3.12。两个最容易踩的版本坑：

```bash
# 1) 必须按 pin 装 faster-whisper 1.1.0，旧版（0.10.0）导入就失败
pip install "faster-whisper==1.1.0" "tokenizers==0.20.3" soundfile

# 2) requirements 里装的是 CPU 版 torch；要用 40/50 系新卡（sm_120）得换 cu128
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

细节与 FAQ 见 [QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md) 第 4、10 节。
