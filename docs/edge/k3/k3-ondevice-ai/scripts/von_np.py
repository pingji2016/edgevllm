#!/usr/bin/env python3
"""
Von 的板端预处理（纯 numpy + tokenizers，不碰 torch / onnxruntime）。

从 `von/models/option_marker.py` 逐行复刻，用于在 K3 上跑 int8 ONNX：
  - pack_sequence   ：拼 [prefix] [SEP] [MASK] opt0 [MASK] opt1 ...
  - build_mask      ：build_independent_option_masks + build_option_invariant_position_ids

与 SDK 的一致性由 von_parity_check.py 逐位验证。
"""
import re

import numpy as np

MASK, SEP = "[MASK]", "[SEP]"
_DIGIT_RUN = re.compile(r"\d+")


def split_digits(t: str) -> str:
    return _DIGIT_RUN.sub(lambda m: " ".join(m.group(0)), t)


def neutralise(text: str, *specials: str) -> str:
    """把用户文本里出现的特殊 token 字面量用零宽连接符打断。"""
    for sp in specials:
        if sp and sp in text:
            text = text.replace(sp, sp[0] + "\u200d" + sp[1:])
    return text


def pack_sequence(state: str, question: str, options, digit_split: bool = False) -> str:
    state = neutralise(state, MASK, SEP)
    question = neutralise(question, MASK, SEP)
    options = [neutralise(o, MASK, SEP) for o in options]
    prefix = f"{question} {state}".strip() if question else state.strip()
    opts = " ".join(f"{MASK} {o.strip()}" for o in options)
    packed = f"{prefix} {SEP} {opts}"
    return split_digits(packed) if digit_split else packed


def build_mask(ids_row: np.ndarray, mask_id: int, last_content: int, sliding_window=None):
    """复刻 build_independent_option_masks + build_option_invariant_position_ids。

    返回 (full, sliding, position_ids, mask_positions)，形状 (1,1,S,S)/(1,S)/(K,)。
    """
    S = len(ids_row)
    pos = np.where(ids_row == mask_id)[0].tolist()
    opt = np.full(S, -1, np.int64)
    for k, st in enumerate(pos):
        en = pos[k + 1] if k + 1 < len(pos) else last_content
        opt[st:en] = k

    oi, oj = opt[:, None], opt[None, :]
    qp, kp, same = oi == -1, oj == -1, oi == oj
    allowed = (qp & kp) | (~qp & (kp | same))
    allowed = allowed | np.eye(S, dtype=bool)          # 全屏蔽行会让 softmax 出 NaN
    full = allowed[None, None]                          # (1,1,S,S)

    if sliding_window is None:
        sliding = full
    else:
        p = np.arange(S)
        local = np.abs(p[:, None] - p[None, :]) <= sliding_window
        sliding = ((allowed & local) | np.eye(S, dtype=bool))[None, None]

    position_ids = np.arange(S)[None, :]
    if pos:
        prefix_len = pos[0]
        for k, st in enumerate(pos):
            en = pos[k + 1] if k + 1 < len(pos) else last_content
            span = en - st
            position_ids[0, st:en] = np.arange(prefix_len, prefix_len + span)
    return full, sliding, position_ids.astype(np.int64), np.array(pos, np.int64)
