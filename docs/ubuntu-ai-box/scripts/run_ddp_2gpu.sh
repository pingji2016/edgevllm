#!/usr/bin/env bash
# minimind 双卡/多卡 DDP 启动器（配套：minimind-training.md 第 6 节）
#
# 用法：
#   bash run_ddp_2gpu.sh                  # 预训练，2 卡
#   NPROC=4 bash run_ddp_2gpu.sh          # 4 卡
#   TASK=sft bash run_ddp_2gpu.sh         # SFT（自动带 --from_weight pretrain）
#   BATCH=48 EPOCHS=2 bash run_ddp_2gpu.sh
#
# 前提：MINIMIND 指向本地 minimind 目录（默认 ~/apps/minimind），
#       且 train_pretrain.py 已按正文 6.3 节打过补丁（GPU 路径行为不变）。
#
# 重要提醒：
#   1. 总 batch = NPROC × BATCH × ACC，lr 按 sqrt(NPROC) 缩放（线性放大对小模型易发散）
#   2. world_size 变化后 iters 会变，**不要用旧 checkpoint 的 step 续训**
#      （SkipBatchSampler 的 skip 偏移会错位，表现为 loss 突然跳变）
#   3. 4090 / 5060 Ti 无 NVLink，卡间走 PCIe；30M 小模型实测加速比 ≈1.8x 而非 2x

set -euo pipefail

MINIMIND=${MINIMIND:-$HOME/apps/minimind}
NPROC=${NPROC:-2}
BATCH=${BATCH:-24}
ACC=${ACC:-8}
BASE_LR=${BASE_LR:-1e-4}
EPOCHS=${EPOCHS:-1}
SEQ=${SEQ:-512}
TASK=${TASK:-pretrain}
HS=${HS:-512}
LAYERS=${LAYERS:-8}

cd "$MINIMIND"
[ -x "$MINIMIND/.venv/bin/torchrun" ] || { echo "找不到 $MINIMIND/.venv/bin/torchrun" >&2; exit 1; }
cd "$MINIMIND/trainer"

LR=$(awk -v b="$BASE_LR" -v n="$NPROC" 'BEGIN{printf "%.3e", b*sqrt(n)}')
EFF=$((NPROC * BATCH * ACC))

case "$TASK" in
  pretrain)
    SCRIPT=train_pretrain.py
    DATA=${DATA:-../dataset/pretrain_t2t_mini.jsonl}
    EXTRA=(--from_weight none)
    ;;
  sft)
    SCRIPT=train_full_sft.py
    DATA=${DATA:-../dataset/sft_t2t_mini.jsonl}
    EXTRA=(--from_weight pretrain)
    ;;
  *)
    echo "未知 TASK=$TASK（pretrain | sft）" >&2
    exit 1
    ;;
esac

echo "------------------------------------------------------------"
echo " 任务        : $TASK   ($SCRIPT)"
echo " 卡数        : $NPROC"
echo " 每卡 batch  : $BATCH   梯度累积: $ACC"
echo " 等效 batch  : $EFF"
echo " learning rate: $BASE_LR (单卡) -> $LR (按 sqrt($NPROC) 缩放)"
echo "------------------------------------------------------------"

exec "$MINIMIND/.venv/bin/torchrun" \
  --nproc_per_node "$NPROC" \
  --master_port "${MASTER_PORT:-29500}" \
  "$SCRIPT" \
  --hidden_size "$HS" --num_hidden_layers "$LAYERS" \
  --batch_size "$BATCH" --accumulation_steps "$ACC" \
  --learning_rate "$LR" --epochs "$EPOCHS" --max_seq_len "$SEQ" \
  --data_path "$DATA" "${EXTRA[@]}" "$@"
