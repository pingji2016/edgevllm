
---

## 9. 追加：判据工具库（把今天的 bug 变成资产）

今天三个实验一共出 **5 个"假失败"**，全部来自判据或集成：

| # | 现象 | 根因 |
| ---: | --- | --- |
| 1 | agent 50% | 三处判据错（112≠116、`37*18` 取了 37、data.csv 金标算错） |
| 2 | CLIP 0% | processor 传了文件路径而不是 PIL 图 |
| 3 | CLIP 0% | `get_*_features` 返回结构体，取错字段 |
| 4 | CLIP 0% | 对已投影向量又做了一次 `text_projection` |
| 5 | `rag_r1` 91.7% | 名字叫 R@1，实际算的 R@5 |

于是新增两个文件：

### 9.1 `bench/judge.py` —— 等价判据

```python
from judge import args_equiv, num_eq, time_key, mcnemar

args_equiv({"target_lang": "中文"}, {"target_lang": "zh"})   # True
args_equiv({"time": "明天 08:00"}, {"time": "明天早上 8 点"})  # True
mcnemar(only_a=32, only_b=12)                                # 3.02
```

覆盖：语言码、单位别名、时间（12/24 小时制 + 中文数字）、
表达式归一（空格/引号/列名同义词）、自由文本包含、数值容差、McNemar。

### 9.2 `tool_selection` 检查

```bash
python run.py --only tool_selection
# tool_selection  0.90  ≥0.9  PASS  (18/20 条（等价判据）, 22.5s)
```

同一套 20 条指令，**等价判据 90%，严格判据会明显更低**——
这就是把它放进 bench 的意义：以后改 prompt/schema，能立刻看出是"模型变了"还是"判据变了"。

### 9.3 张量集成的三条断言

CLIP 那三个 bug 任何一个都能被下面三行拦住：

```python
assert x.dtype == torch.float32, x.dtype
assert x.shape[-1] == 512, x.shape
print("norm:", x.norm(dim=-1)[:3])   # 应在 5–20 之间，不应是 0 或 1
```

> **规则：任何 embedding 代码，先打印 shape / dtype / norm 再往下写。**
> 三个 bug 的共同点就是"没看中间张量"。
