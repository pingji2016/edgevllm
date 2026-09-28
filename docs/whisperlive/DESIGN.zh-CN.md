# WhisperLive 处理流程与设计要点（深入解析）

> 配套文档：[QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md)（安装、参数表、FAQ）
> 本文基于 `v0.6.2` 源码逐行阅读整理，所有结论都标了出处，可对照代码验证。
>
> 文中 `whisper_live/server.py#L1007` 形式的引用均已改写为上游 `v0.6.2` 的永久链接，
> 点开即是所引的那一行。本文是本地整理的学习笔记、**非上游内容**，
> 来源与本地对照方式见 [README.md](README.md)。

---

## 目录

- [0. 先跑起来（运行方式速查）](#0-先跑起来运行方式速查)
- [1. 整体设计取舍](#1-整体设计取舍)
- [2. 端到端数据流](#2-端到端数据流)
- [3. 服务端循环详解](#3-服务端循环详解)
- [4. 特殊处理逐条拆解（重点）](#4-特殊处理逐条拆解重点)
- [5. 两个后端的差异](#5-两个后端的差异)
- [6. 调优旋钮速查](#6-调优旋钮速查)
- [7. 代码里粗糙的地方](#7-代码里粗糙的地方)
- [8. 建议的代码走读顺序](#8-建议的代码走读顺序)

---

## 0. 先跑起来（运行方式速查）

完整的环境说明见 [QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md)，这里给一条**从零到出字幕**的最短路径。

### 0.1 装依赖

```bash
cd WhisperLive

# 服务端：包含 faster-whisper / openai-whisper / onnxruntime / PyAudio 等
pip install -r requirements/server.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 客户端：只需要 PyAudio + websocket-client + av + scipy
pip install -r requirements/client.txt
```

两个容易忽略的点：

- `requirements/server.txt` 里**没有直接写 torch**，它是通过 `openai-whisper` 间接拉进来的。想要 GPU 支持，请另按 [QUICKSTART 4.3 节](QUICKSTART.zh-CN.md) 装 CUDA 版 torch。
- 本仓库代码是按 **faster-whisper 1.1.0** 的 API 写的（[transcriber.py:21](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L21) 用 `pad_or_trim`，1.1.0 才有）。如果机器上已有旧版（比如 0.10.0），**必须**按 requirements 重装一次，否则导入就失败：

  ```bash
  pip install "faster-whisper==1.1.0" "tokenizers==0.20.3" soundfile
  ```

  冒烟自检（能打印 `ok` 才说明环境对了）：

  ```bash
  python -c "from whisper_live.server import TranscriptionServer; print('ok')"
  ```

### 0.2 启动服务端

```bash
# 国内网络建议先设 HF 镜像，模型下载会快很多
export HF_ENDPOINT=https://hf-mirror.com          # Git Bash / Linux / macOS
# set HF_ENDPOINT=https://hf-mirror.com           # Windows CMD

python run_server.py --port 9090 --backend faster_whisper
```

期望日志（第一次连接时才会下载模型，会卡一会儿）：

```
INFO:whisper_live.server:Single model mode currently only works with custom models.
INFO:whisper_live.server:New client connected
INFO:whisper_live.server:Running faster_whisper backend.
```

### 0.3 启动客户端

新开一个终端，**仍然在仓库根目录**（`whisper_live` 包要能被导入）：

```bash
# 转写仓库自带的示例音频（会自动重采样成 ./jfk_resampled.wav）
python -c "
from whisper_live.client import TranscriptionClient
c = TranscriptionClient('localhost', 9090, lang='en', model='small')
c('assets/jfk.flac')
"
```

也可以用麦克风实时转写：

```bash
python -c "
from whisper_live.client import TranscriptionClient
c = TranscriptionClient('localhost', 9090, lang='zh', model='small')
c()          # Ctrl+C 结束
"
```

### 0.4 验证结果

- 终端会实时刷新最近 3 段字幕（[client.py:115](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L115) `process_segments`）。
- 转写结束后生成 **`output.srt`**（默认路径），内容形如：

  ```
  1
  00:00:00,000 --> 00:00:02,400
  And so my fellow Americans, ask not what your country can do for you...
  ```

- 服务端日志里能看每块的耗时信息 `Running faster_whisper backend.` / `Processing audio with duration ...`。

### 0.5 跑测试

```bash
python -m unittest discover -s tests        # 测试用 mock 掉了 websocket 和音频设备，不需要真的麦克风
```

---

## 1. 整体设计取舍

Whisper 本身是**批处理模型**：吃一段（标称 30 秒）音频，吐完整结果。要把它做成"实时字幕"，绕不开三个矛盾：

| 矛盾 | 常规做法 | WhisperLive 的做法 |
| --- | --- | --- |
| Whisper 需要 ≥30 秒上下文才准，但实时要求低延迟 | 等满 30 秒再出结果 | **滚动窗口**：每累积约 1 秒就把"尚未定稿的全部音频"再喂一次模型，每次只把**已定稿的段**追加进字幕（[server.py:1007](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1007)） |
| 一句话说到一半，音频被切断，最后一个词必然是残缺的 | 也把它当结果输出，用户会看到错字反复跳动 | **最后一个 segment 永不立即定稿**：先以 `completed=false` 发给前端的"临时结果显示"，只有它连续多轮不再变化（说明说完了）才落库（[server.py:1097](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1097)） |
| 静音/噪声时 Whisper 会"幻觉"出重复文本，音频却一直在涨，时间轴会无限落后 | 无 | **重复输出检测** + **无有效输出时强制推进时间轴**（[server.py:476](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L476)、[server.py:1111](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1111)） |

一句话概括架构：**服务端做"流式化包装"，模型本身不改**。所有实时性技巧都在 `server.py` 的 `ServeClientBase` 子类里，`transcriber.py` 只是 faster-whisper 的一个可调用封装。

---

## 2. 端到端数据流

```
┌── 客户端进程 ─────────────────────────┐   ┌── 服务端进程 ────────────────────────┐
│                                       │   │                                      │
│ ①麦克风/文件/RTSP/HLS → 16kHz mono    │   │                                      │
│ ②int16 → float32(/32768) 二进制帧	  │──►│ ④websocket 收帧 → np.frombuffer      │
│   每帧 4096 采样 = 256ms              │   │   → client.add_frames() 追加进缓冲   │
│                                       │   │              │                       │
│                                       │   │              ▼                       │
│                                       │   │ ⑤识别线程（每连接一条）              │
│                                       │   │   frames_np 尾部未定稿音频            │
│                                       │   │   → transcriber.transcribe()        │
│                                       │   │   → update_segments() 决定谁定稿     │
│ ⑦打印最近3段 + 写 output.srt          │◄──│ ⑥send_transcription_to_client()      │
│   （只收 completed=true 的段）        │   │   只回最近 10 段 JSON                │
└───────────────────────────────────────┘   └──────────────────────────────────────┘
```

### 各步骤对应代码

| 步骤 | 位置 |
| --- | --- |
| ① 采集与分块（4096 样本/包） | [client.py:298-301](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L298) |
| ② int16 → float32 归一化 | [client.py:646-661](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L646) `bytes_to_float_array` |
| ③ 文件重采样到 16kHz wav | [utils.py:45](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/utils.py#L45) `resample`（用 PyAV，产物 `<原名>_resampled.wav`） |
| ④ 收帧入缓冲 | [server.py:210-223](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L210)、[server.py:445](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L445) `add_frames` |
| ⑤ 识别循环 | [server.py:979](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L979) `speech_to_text` |
| ⑥ 回推结果 | [server.py:544](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L544) `send_transcription_to_client` |
| ⑦ 客户端消费与落盘 | [client.py:115](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L115) `process_segments`、[client.py:259](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L259) `write_srt_file` |

### 握手与收尾协议

```
客户端                                    服务端
  │── JSON{uid,language,task,model,       │
  │        use_vad,max_clients,           │
  │        max_connection_time} ─────────►│  作为"第一个 websocket 消息"被解析
  │                                       │  server.py:229
  │◄──────── {message:"SERVER_READY",     │
  │            backend:"faster_whisper"} ─│  server.py:840
  │── 二进制 float32 PCM 帧 ─────────────►│  循环 body: recv()
  │◄──────── {segments:[{start,end,text,  │
  │            completed}]} ──────────────│
  │── b"END_OF_AUDIO" ───────────────────►│  结束哨兵（文件/流模式）
  │◄──────── 最后一波 segments ───────────│
  │  等 15s 无新响应后关闭连接             │  client.py:274 wait_before_disconnect
```

几个细节：

- 握手没有子协议、没有鉴权，就是"第一条消息必须是 JSON 配置"（[server.py:229](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L229)），协议很脆弱但也极简。
- `uid` 由客户端 `uuid4()` 生成，服务端每条消息都原样带回来，客户端用它做校验（[client.py:153](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L153)）。**这等于没有鉴权**：知道服务端地址的任何人都能连上去消耗算力。
- `END_OF_AUDIO` 是个**字符串字节**（不是 JSON），在二进制流里被当成特殊帧识别（[server.py:221](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L221)）。

---

## 3. 服务端循环详解

每个客户端连接会产生 **2 条线程 + 1 把锁**：

```
线程 A（websockets 库的 handler，即 recv_audio）
  while True:
      frame = websocket.recv()          # 阻塞收帧
      add_frames(frame)                 # 加锁写 frames_np（server.py:461）
      # tensorrt 后端还会在这里做 VAD 门控

线程 B（per-client 识别线程 speech_to_text）
  while not self.exit:
      if frames_np is None: 跳过
      clip_audio_if_no_valid_segment()  # 兜底：防止时间轴无限落后
      input_bytes, duration = get_audio_chunk_for_processing()
      if duration < 1.0: 睡 0.1s 后重试   # 攒够 1 秒才识别
      result = transcriber.transcribe(input_bytes, ...)
      handle_transcription_output(result, duration)
```

关键在 `get_audio_chunk_for_processing()`（[server.py:487](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L487)）：

```python
samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
input_bytes = self.frames_np[int(samples_take):].copy()   # 从"已定稿位置"到缓冲末尾的全部音频
```

也就是说：**每次识别的输入是"尚未定稿的全部音频"**（可能是 1 秒，也可能是 40 秒），而不是固定的 1 秒切片。这就是"近实时"的来源——音频越积越多时，单次推理会变慢，但字幕在持续往外滚。

---

## 4. 特殊处理逐条拆解（重点）

### 4.1 双游标：`frames_offset` 与 `timestamp_offset`

这两个变量最容易看糊，它们回答的是两个不同问题：

| 变量 | 含义 | 场景 |
| --- | --- | --- |
| `frames_offset` | **缓冲区**的时间原点。缓冲区被裁剪时整体前移 | 缓冲超 45s，丢最旧 30s → `frames_offset += 30`（[server.py:462](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L462)） |
| `timestamp_offset` | **已定稿/已放弃**的音频位置。它决定了"下一块要识别哪段音频"，也是字幕时间戳的基准 | 每次定稿后 `timestamp_offset += offset`（[server.py:1130](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1130)） |

所以：

```
frames_np 缓冲区:   [==========|====================]
                     ↑frames_offset  ↑timestamp_offset
                     已经丢弃/已定稿   ↑ 待识别的音频
```

字幕里的绝对时间是 `timestamp_offset + segment.start`（[server.py:1076](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1076)），因为模型只会给出"相对当前块的秒数"，必须加上基准才能拼出**整场会话的连续时间轴**。

另外裁剪缓冲区时有个保护：如果 `timestamp_offset` 落在被丢弃区间里，就把它抬到新的 `frames_offset`（[server.py:468](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L468)），否则减出负数会直接崩。

### 4.2 为什么"最后一段不算数"

`update_segments()`（[server.py:1045](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1045)）是整个项目的灵魂，逻辑是：

```python
# 1) 除最后一段外，全部定稿
for s in segments[:-1]:
    transcript.append({start: ts_offset + s.start, end: ts_offset + min(duration, s.end), text: s.text, completed: True})
    offset = min(duration, s.end)          # 记录"消费到哪儿了"

# 2) 最后一段只作为临时结果发给前端，不定稿
last_segment = {..., completed: False}     # 因为它的结尾很可能是被切掉半个词
```

为什么？因为音频是**边收边喂**的，缓冲区末尾是"刚说到一半"的音频。Whisper 对半句话经常给出残缺结果（`"the quick brown fo"`）。如果直接落库，用户就会看到文字被反复改写。所以设计是：

- **前端**拿到 `completed=false` 的段，可以当作"正在输入的字幕"渲染（插件就是这么做的）。
- **后端**只在同一段文本**连续 10 次以上不再变化**时，才认为说话人停下来了，把它定稿（见 4.3）。

`offset = min(duration, s.end)` 的 `min` 也不是随手写的：`s.end` 是模型给的段尾，可能因为幻觉超出实际音频长度，取 `min` 防止时间轴冲过头。

### 4.3 重复输出检测：防"幻觉死循环"

这是最容易踩坑的地方。假设用户说完一句话后闭嘴，但客户端还在推流：

```
第 1 轮音频 0-2s   → "今天天气不错"        ← 未定稿
第 2 轮音频 0-3s   → "今天天气不错"        ← 未定稿（重复）
第 3 轮音频 0-4s   → "今天天气不错"        ← 未定稿（还是重复）
...  音频越来越长，模型对越来越长的静音继续输出同一句
```

如果不干预，`timestamp_offset` 永远不前进，缓冲区越堆越长，最后触发裁剪，字幕彻底乱掉。

对策（[server.py:1097-1128](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1097)）：

```python
if current_out.strip() == prev_out.strip() and current_out != '':
    same_output_count += 1
    if end_time_for_same_output is None:
        end_time_for_same_output = segments[-1].end     # 记住"第一次重复"时的段尾
    time.sleep(0.1)     # 故意等一下，给说话人继续说的机会（利于标点）
else:
    same_output_count = 0
    end_time_for_same_output = None

if same_output_count > self.same_output_threshold:      # 默认 10
    transcript.append({end: ts_offset + min(duration, end_time_for_same_output), text: current_out, completed: True})
    offset = min(duration, end_time_for_same_output)    # 关键：推进时间轴
    current_out = ''                                    # 清空，下一轮重新开始
    same_output_count = 0
```

三个值得注意的巧思：

1. **用 `end_time_for_same_output` 而不是当前的 `segments[-1].end`**：注释里写得很清楚——第一次出现重复时的段尾，比当前值更保守，避免把"还没说完但已经开始重复"的音频误判成说完（[server.py:1100](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1100)）。
2. **`time.sleep(0.1)` 是刻意的**：不是为了省 CPU，而是为了"等说话人把话说完"，让标点更准（[server.py:1104](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1104)）。
3. **阈值可调**：`same_output_threshold = 10`（[server.py:802](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L802)）。调小 → 定稿快但容易把长句斩断；调大 → 更稳但延迟高。

### 4.4 无有效输出时的兜底：`clip_audio_if_no_valid_segment`

如果模型对这 25 秒音频**一个段都没吐出来**（嘈杂环境、纯噪声、极端 hallucination），`offset` 保持 `None`，`timestamp_offset` 就不动，下一轮喂进去的音频更长，形成恶性循环。兜底逻辑（[server.py:476](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L476)）：

```python
if frames_np[int((timestamp_offset - frames_offset) * RATE):].shape[0] > 25 * RATE:
    duration = frames_np.shape[0] / RATE
    timestamp_offset = frames_offset + duration - 5      # 抛弃旧音频，只留最近 5 秒
```

含义是：**"25 秒都没识别出东西，那我认了，把这堆音频丢掉，只保留最近 5 秒作为上下文。"** 这是防雪崩的最后一道闸。

### 4.5 缓冲区裁剪：45 秒上限 / 丢 30 秒

[server.py:462](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L462)：`frames_np` 超过 45 秒就丢掉最旧的 30 秒。为什么丢 30 而不是丢到刚好 45？因为 Whisper 的窗口就是 30 秒，丢 30 秒能让剩余音频仍是一个完整的窗口边界，省掉一次重新对齐；而且**能长期跑不爆内存**。

### 4.6 两层静音过滤：VAD 与 `no_speech_prob`

两层是**互补**的，不是二选一：

| 层 | 作用位置 | 说明 |
| --- | --- | --- |
| faster-whisper 的 `vad_filter` | 模型输入前，切掉静音段再拼接 | 服务端把客户端的 `use_vad` 透传过来（[server.py:923](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L923)），VAD 参数默认 `{"onset": 0.5}`（[server.py:800](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L800)） |
| `no_speech_prob` 阈值 0.45 | 模型输出后，丢弃"模型自己认为没人说话"的段 | [server.py:801](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L801)，在 `update_segments` 里对每个段判断 |

注意第一层 VAD 会**改变时间轴**：被切掉的静音消失了，所以 faster-whisper 内部要用 `restore_speech_timestamps` 把时间戳映射回原始时间轴（[transcriber.py:1794](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L1794)）。这是"VAD 开着但字幕时间轴仍然对得上"的原因。

### 4.7 静音时"回显上一段"和插入空行

人说话有停顿。停顿期间模型没有输出，但前端不能就这么干等着（否则界面像卡死）。于是（[server.py:932](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L932)）：

- **停顿 < 5 秒**（`show_prev_out_thresh`）：把已定稿的段**重发一遍**（内容没变，客户端会去重，不会重复显示）。
- **停顿 > 3 秒**（`add_pause_thresh`）：往 `self.text` 里塞一个空字符串，作为"这里有个停顿"的标记（[server.py:954](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L954)）。
- 如果长时间没有有效输出（`t_start` 超时），就不发任何东西。

### 4.8 语言处理

三处相关逻辑，容易混：

1. **模型名以 `.en` 结尾 → 强制英文**：`self.language = "en" if self.model_size_or_path.endswith("en") else language`（[server.py:797](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L797)）。注意 `distil-large-v3` 这类不以 `.en` 结尾的多语言 distil 模型不在此列。
2. **客户端不指定语言 → 服务端自动检测**：faster-whisper 会在音频里探测语言，`language_probability > 0.5` 时通过 `{"language": ..., "language_prob": ...}` 消息告知客户端（[server.py:884](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L884)）。
3. **检测只做一次**：一旦 `self.language` 被设定，后续调用都会显式传语言，不再探测。

### 4.9 设备与精度选择

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    major, _ = torch.cuda.get_device_capability(device)
    self.compute_type = "float16" if major >= 7 else "float32"   # Volta 及以上用 fp16
else:
    self.compute_type = "int8"                                   # CPU 用 int8 量化提速
```

见 [server.py:805](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L805)。**这里没有给用户任何覆盖入口**——想强制 fp32/int8 只能改代码。

### 4.10 "单模型模式"与它的锁

| 模式 | 触发条件 | 行为 | 代价 |
| --- | --- | --- | --- |
| 多模型（默认） | 不传 `-fw` / `-trt` | 每个连接按客户端请求的 `model` 新建模型 | 连接时等待长、(V)RAM 随连接数线性增长 |
| 单模型 | 传了 `-fw` / `-trt` | 类变量 `SINGLE_MODEL` 懒加载一次，后续连接复用 | 所有连接只能用同一个模型 |

复用的实现是**类属性 + 类级锁**：`SINGLE_MODEL` / `SINGLE_MODEL_LOCK`（[server.py:767](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L767)、[server.py:768](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L768)）。因为 CTranslate2 的模型实例不是线程安全的，每次推理都要拿锁（[server.py:916](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L916)）——**这意味着单模型模式下多个客户端实际上是串行识别的**，一个慢会话会拖住所有人。这是"省显存"换来的代价。

### 4.11 客户端侧的几个处理

- **重采样**：非 wav 输入先用 PyAV 重采样成 16kHz / 单声道 / `s16`，产物写在**当前工作目录**（[utils.py:67](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/utils.py#L67)）。注意这个中间文件不会被清理。
- **文件转写是"边播边传"**：`play_file` 读 wav 分块发送，同时通过 PyAudio 播放出来（[client.py:376](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L376)）。`mute_audio_playback=True` 时不播放但仍按分块时长 `sleep`，以维持实时节奏。
- **收尾要等**：发完 `END_OF_AUDIO` 前会先 `wait_before_disconnect()` 忙等最多 15 秒，确保服务端把最后一波字幕吐完（[client.py:274](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L274)）。
- **SRT 去重规则**：只接受 `completed=true` 且 `start >= 上一段 end` 的段（[client.py:123](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L123)），避免临时结果和重复推送污染字幕。
- **多个客户端同时喂**：`TranscriptionTeeClient` 支持把同一份音频 `multicast` 给多个服务端（例如同时跑 faster-whisper 和 TensorRT 对比效果），`TranscriptionClient` 只是它包装的"单客户端版"。

### 4.12 连接管理：满员排队与超时踢人

`ClientManager`（[server.py:24](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L24)）：

- **满员**：达到 `max_clients`（默认 4）时，回一条 `{"status": "WAIT", "message": <预计等待分钟数>}` 然后关连接（[server.py:91](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L91)）。等待时间算法是**所有在线客户端里"最早到期"的那个的剩余时间**（[server.py:77](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L77)）——注意这是个乐观估计，没有考虑排队顺序。
- **超时**：每轮收帧前检查，超过 `max_connection_time`（默认 600 秒）就主动断开并告警（[server.py:109](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L109)）。
- `max_clients` / `max_connection_time` 由**第一个连上来的客户端**决定，之后不再变（[server.py:232](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L232)）。

### 4.13 `transcriber.py` 相对上游 faster-whisper 的改动

这个文件是 faster-whisper 的 **vendored 副本**（文件头注明来源），为了配合流式场景做了改动，最重要的两处：

1. **`generate_segments` 返回列表而不是生成器**（[transcriber.py:1339](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L1339) `return all_segments`）。上游是 `yield` 的惰性生成器，而服务端需要 `len(result)`、`segments[:-1]`、`segments[-1]`（[server.py:968](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L968)、[server.py:1071](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1071)）——**一刀切列表化**才能按下标取"最后一段"。这也解释了为什么不能直接把上游 faster-whisper 换进来用。
2. **显式暴露 `no_speech_prob`**：`Segment` 数据类带上了 `no_speech_prob` 字段（[transcriber.py:60](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L60)），服务端才能按 4.6 的阈值过滤。

另外服务端调用时用的是默认参数 `without_timestamps=False`，所以段里**有 `start`/`end`**——这是生成 SRT 时间轴的前提。

---

## 5. 两个后端的差异

| 维度 | `faster_whisper`（默认） | `tensorrt` |
| --- | --- | --- |
| 推理引擎 | CTranslate2 | TensorRT-LLM（需预编译 engine） |
| 依赖门槛 | `pip install` 即可 | 需按 [TensorRT_whisper.md](https://github.com/collabora/WhisperLive/blob/v0.6.2/TensorRT_whisper.md) 搭建 + 编译 |
| VAD 位置 | 模型内部 `vad_filter` | 服务端逐帧 Silero VAD（[vad.py:143](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/vad.py#L143)），连续 3 帧无语音判定句尾 |
| 定稿策略 | 看 segment 的 `no_speech_prob` 和重复计数 | 看 `eos` 标志（无语音时置位）才推进时间轴（[server.py:685](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L685)） |
| 时间戳 | 模型给 start/end，能做精确 SRT | 用 `<|notimestamps|>` 前缀，**没有细粒度时间戳**，靠累加块时长推算（[server.py:708](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L708)） |
| 最小识别音频 | 1.0 秒（[server.py:1007](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1007)） | 0.4 秒（[server.py:753](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L753)） |
| 启动开销 | 首次下载模型 | 要 warmup 10 次推理（[server.py:651](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L651)） |
| 语言 | 自动检测 | 需客户端传 `-m` 声明多语言模型，否则固定 `en` |
| 降级行为 | — | 初始化失败会发 `WARNING` 并**自动回退到 faster_whisper**（[server.py:173](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L173)） |

两者共享 `ServeClientBase` 里的缓冲区管理、时间轴推进、客户端通信这部分代码，差异集中在"怎么调用模型"和"什么信号代表一句话说完了"。

---

## 6. 调优旋钮速查

代码里写死但值得知道的常量（都在 `server.py` 的 `ServeClientBase.__init__` 和 `ServeClientFasterWhisper.__init__`）：

| 常量 | 值 | 作用 | 调大的后果 |
| --- | --- | --- | --- |
| `show_prev_out_thresh` | 5 秒 | 停顿多久之内重发上一段 | 界面更"粘"，但不易闪空 |
| `add_pause_thresh` | 3 秒 | 停顿多久插入空行标记 | 段落分隔更碎 |
| `send_last_n_segments` | 10 | 每次回推最近几段 | 网络流量变大、前端有更多上下文 |
| `no_speech_thresh` | 0.45 | 判定"这是静音"的概率阈值 | 更激进地丢结果，可能漏词 |
| `same_output_threshold` | 10 | 重复几轮才定稿 | 延迟高，但句子更完整 |
| `RATE` | 16000 | 全链路采样率，**不可改**（模型硬要求） | — |

运行时可调的是 `run_server.py` 的命令行参数（端口、后端、模型路径、OMP 线程数）和客户端构造函数参数（模型大小、VAD、语言、超时），见 [QUICKSTART.zh-CN.md](QUICKSTART.zh-CN.md) 的参数表。

---

## 7. 代码里粗糙的地方

阅读时值得留意的几处（不是 bug 报告，只是读代码时容易困惑的点）：

1. **两处忙等**：
   - `ServeClientFasterWhisper.speech_to_text` 里 `if self.frames_np is None: continue`（[server.py:1001](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L1001)）**没有 sleep**，连接刚建立还没音频时会空转占 CPU。TensorRT 版本有 `time.sleep(0.02)`（[server.py:747](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L747)），可见是遗漏而非设计。
   - `wait_before_disconnect` 用 `while ...: continue` 忙等最多 15 秒（[client.py:277](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L277)）。
2. **`initialize_client` 的 `except Exception: return`**（[server.py:202](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L202)）把异常整个吞掉，不打印日志。客户端表现就是"连上了但永远不 ready"，排查困难。
3. **模型加载失败的提示不精确**：`ServeClientFasterWhisper` 只回 `Failed to load model: <模型名>`（[server.py:830](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L830)），不区分"网络下载失败"和"显存不足"。
4. **`Client.__init__` 有初始化竞态**：websocket 线程在 [client.py:96](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L96) 就启动了，而 `self.transcript`、`self.server_backend` 等属性在之后才赋值（[client.py:100](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L100)），理论上极快返回的 `SERVER_READY` 之后的消息可能撞上未初始化的属性。
5. **默认模式显存放大**：不传自定义模型时，每个连接一套模型（[README.md:65](https://github.com/collabora/WhisperLive/blob/v0.6.2/README.md#L65) 里官方也承认了这点，只是把它当作"可自由选模型尺寸"的代价）。
6. **单模型模式实际是串行**：见 4.10，锁的粒度是整个推理过程。
7. **依赖 pin 不一致**：`setup.py` 写 `onnxruntime==1.16.0`，`requirements/server.txt` 写 `1.17.0`。
8. **无鉴权、无加密**：裸 `ws://`，任何能访问端口的人都能连（[client.py:79](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L79)）。生产部署务必放在反代/内网后面。

---

## 8. 建议的代码走读顺序

按这个顺序读，能最快建立完整心智模型（总量约 2900 行，核心部分不到 400 行）：

1. **[run_server.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/run_server.py)**（51 行）— 看清有哪些可配项。
2. **[client.py:76-230](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L76)** — 先当"协议说明书"看：客户端发什么、期待收到什么。
3. **[server.py:225-330](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L225)** `handle_new_connection` / `recv_audio` / `run` — 连接生命周期。
4. **[server.py:407-590](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L407)** `ServeClientBase` — 缓冲区与双游标（对照本文 4.1、4.5）。
5. **[server.py:979-1134](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L979)** `speech_to_text` + `update_segments` — **本文 4.2、4.3 的落点，全项目最关键的一段**。
6. **[server.py:590-763](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/server.py#L590)** TensorRT 分支，对照第 5 章的差异表看。
7. **[transcriber.py:694](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/transcriber.py#L694)** `WhisperModel.transcribe` — 只看参数和返回值形状即可，内部是 faster-whisper 原版；重点确认 4.13 的两个改动点。
8. 最后看 **[client.py:376-431](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/client.py#L376)** `play_file` 和 [utils.py](https://github.com/collabora/WhisperLive/blob/v0.6.2/whisper_live/utils.py) — 文件转写与 SRT 生成。
