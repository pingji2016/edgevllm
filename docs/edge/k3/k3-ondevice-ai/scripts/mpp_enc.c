/*
 * mpp_enc.c —— SpaceMiT K3 MPP 原生 H.264/H.265 编码器（带码率控制）
 *
 * 为什么需要它（报告 §5.1 / §11 第 1 条）：
 *   ffmpeg 的 `*_stcodec` 封装**没有把 `-b:v` 映射到 MPP 的码率控制**，
 *   所以硬编码出来的码率不可调。这个程序直接调 MPP 的 VENC/VB API，
 *   显式设置 `VencChnAttr.eRcMode / u32Bitrate / u32MinQp / u32MaxQp`，
 *   从而拿到真正的 CBR / VBR / CVBR / FIXQP。
 *
 * 数据流（零拷贝）：
 *   读 NV12 裸帧 → VB 池取 dma-buf → 按 stride 写入平面 → VENC_SendFrame
 *   → VENC_GetStream（零拷贝拿到码流）→ 写文件 → VENC_ReleaseStream
 *   → VB_ReleaseBuffer
 *
 * 用法：
 *   ./mpp_enc -i in.nv12 -w 1920 -h 1080 -n 150 -o out.h264 \
 *             -c h264 -b 4000000 --rc cbr --fps 30 --gop 30
 *   ./mpp_enc ... --rc vbr
 *   ./mpp_enc ... --rc cvbr --max-bitrate 8000000
 *   ./mpp_enc ... --rc fixqp --iqp 26 --pqp 30
 *
 * 输入：裸 NV12（Y 平面 w*h 字节，紧跟 UV 交错平面 w*h/2 字节，紧密排列）。
 *   生成：ffmpeg -i in.mp4 -pix_fmt nv12 -f rawvideo in.nv12
 *
 * 编译：
 *   cc -O2 -Wall -o mpp_enc mpp_enc.c -lmpp
 *   （库名不确定时：ldconfig -p | grep -i mpp）
 *
 * ★ 首次跑请加 `--dump-info`，确认打印出的 planeNum/stride 符合预期。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <venc/venc_api.h>
#include <sys/vb_api.h>

static const char *prog = "mpp_enc";

static void usage(void)
{
    fprintf(stderr,
        "用法: %s -i <in.nv12> -w <W> -h <H> -n <帧数> -o <out.h264|h265> [选项]\n"
        "  -c h264|h265       编码格式（默认 h264）\n"
        "  -b <bps>           目标码率（默认 4000000 = 4Mbps）\n"
        "  --rc cbr|vbr|cvbr|fixqp   码率控制模式（默认 cbr）\n"
        "  --max-bitrate <bps>       CVBR 上限（默认 2×目标）\n"
        "  --min-qp <n> --max-qp <n> RC 的 QP 上下限（默认 10 / 45）\n"
        "  --iqp <n> --pqp <n>       FIXQP 的 I/P 帧 QP（默认 26 / 30）\n"
        "  --fps <n>          帧率（默认 30）\n"
        "  --gop <n>          GOP（默认 = fps）\n"
        "  --dump-info        打印一帧的 VB/平面信息\n",
        prog);
}

static const char *arg_val(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s 缺少参数: %s\n", prog, name);
        exit(2);
    }
    return argv[++(*i)];
}

/* 把编码器里已产出的码流全部抽干；返回抽出的包数 */
static int drain_streams(FILE *fout, unsigned long long *total, int timeout_ms)
{
    StreamBufferInfo sb;
    int n = 0;
    while (VENC_GetStream(0, &sb, (U32)timeout_ms) == 0) {
        if (sb.u32Size && sb.pu8Addr) {
            fwrite(sb.pu8Addr, 1, sb.u32Size, fout);
            *total += sb.u32Size;
        }
        n++;
        VENC_ReleaseStream(0, &sb);
        timeout_ms = 0;           /* 第一个之后改成非阻塞，抽干为止 */
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *in_path = NULL, *out_path = NULL;
    int width = 0, height = 0, frames = 0;
    unsigned bitrate = 4000000u, max_bitrate = 0;
    int fps = 30, gop = 0, min_qp = 10, max_qp = 45, iqp = 26, pqp = 30;
    int dump_info = 0;
    VencRcMode rc = VENC_RC_MODE_CBR;
    MppStreamCodecType codec = MPP_STREAM_CODEC_H264;

    FILE *fin = NULL, *fout = NULL;
    UL pool = 0;
    int chn_created = 0, chn_enabled = 0;
    unsigned char *ysrc = NULL, *uvsrc = NULL;
    int ret = 1;
    int sent = 0, got = 0;
    unsigned long long total_bytes = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-i")) in_path = arg_val(argc, argv, &i, a);
        else if (!strcmp(a, "-o")) out_path = arg_val(argc, argv, &i, a);
        else if (!strcmp(a, "-w")) width = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "-h")) height = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "-n")) frames = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "-b")) bitrate = (unsigned)strtoul(arg_val(argc, argv, &i, a), NULL, 10);
        else if (!strcmp(a, "-c")) {
            const char *c = arg_val(argc, argv, &i, a);
            codec = (!strcmp(c, "h265") || !strcmp(c, "hevc"))
                        ? MPP_STREAM_CODEC_H265 : MPP_STREAM_CODEC_H264;
        }
        else if (!strcmp(a, "--max-bitrate")) max_bitrate = (unsigned)strtoul(arg_val(argc, argv, &i, a), NULL, 10);
        else if (!strcmp(a, "--min-qp")) min_qp = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--max-qp")) max_qp = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--iqp")) iqp = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--pqp")) pqp = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--fps")) fps = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--gop")) gop = atoi(arg_val(argc, argv, &i, a));
        else if (!strcmp(a, "--dump-info")) dump_info = 1;
        else if (!strcmp(a, "--rc")) {
            const char *m = arg_val(argc, argv, &i, a);
            if (!strcmp(m, "cbr")) rc = VENC_RC_MODE_CBR;
            else if (!strcmp(m, "vbr")) rc = VENC_RC_MODE_VBR;
            else if (!strcmp(m, "cvbr")) rc = VENC_RC_MODE_CVBR;
            else if (!strcmp(m, "fixqp")) rc = VENC_RC_MODE_FIXQP;
            else { fprintf(stderr, "未知 --rc: %s\n", m); return 2; }
        }
        else if (!strcmp(a, "-H") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "未知参数: %s\n", a); usage(); return 2; }
    }

    if (!in_path || !out_path || width <= 0 || height <= 0) { usage(); return 2; }
    if (gop <= 0) gop = fps;
    if (max_bitrate == 0) max_bitrate = bitrate * 2;
    if ((width & 1) || (height & 1)) {
        fprintf(stderr, "宽高必须是偶数（NV12）：%dx%d\n", width, height);
        return 2;
    }

    const U32 align = 16;
    const U32 ah = (U32)((height + align - 1) / align * align);

    fin = fopen(in_path, "rb");
    if (!fin) { fprintf(stderr, "打不开输入 %s: %s\n", in_path, strerror(errno)); return 1; }
    fout = fopen(out_path, "wb");
    if (!fout) { fprintf(stderr, "打不开输出 %s: %s\n", out_path, strerror(errno)); fclose(fin); return 1; }

    if (VB_Init() != 0) { fprintf(stderr, "VB_Init 失败\n"); goto cleanup; }
    if (VENC_Init() != 0) { fprintf(stderr, "VENC_Init 失败\n"); goto cleanup; }

    /* ---------- 1) 描述帧格式，算 buffer 大小 ---------- */
    VideoFrameInfo fi;
    memset(&fi, 0, sizeof(fi));
    fi.eFrameType = FRAME_TYPE_VENC;
    fi.eModId = MPP_ID_VENC;
    fi.stCommFrameInfo.u32Width = (U32)width;
    fi.stCommFrameInfo.u32Height = (U32)height;
    fi.stCommFrameInfo.u32Align = align;
    fi.stCommFrameInfo.ePixelFormat = MPP_PIXEL_FORMAT_NV12;
    fi.stCommFrameInfo.eCompressMode = COMPRESS_MODE_NONE;
    fi.stCommFrameInfo.eColorSpace = COLOR_SPACE_BT709_LIMIT;

    S32 buf_size = VB_GetPicBufferSize(&fi);
    if (buf_size <= 0) {
        fprintf(stderr, "VB_GetPicBufferSize=%d，按宽高估算\n", buf_size);
        buf_size = (S32)((U32)width * ah * 3 / 2);
    }

    /* ---------- 2) VB 池（dma-buf 由 MPP 分配，交给编码器零拷贝） ---------- */
    VbPoolCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.u32BufSize = (U32)buf_size;
    cfg.u32BufCnt = 6;
    cfg.eModId = MPP_ID_VENC;
    cfg.eRemapMode = VBUF_REMAP_MODE_CACHED;   /* CPU 要 memcpy 进来 */

    pool = VB_CreatePool(&cfg);
    if (pool == 0) {
        fprintf(stderr, "VB_CreatePool 失败（size=%d cnt=%u）\n", buf_size, cfg.u32BufCnt);
        goto cleanup;
    }
    if (VB_SetFrameInfo(pool, &fi) != 0)
        fprintf(stderr, "警告: VB_SetFrameInfo 失败（继续）\n");

    /* ---------- 3) 编码通道：码率控制在这里 ---------- */
    VencChnAttr attr;
    memset(&attr, 0, sizeof(attr));
    attr.eCodecType = codec;
    attr.eInputPixelFormat = MPP_PIXEL_FORMAT_NV12;
    attr.u32Width = (U32)width;
    attr.u32Height = (U32)height;
    attr.u32Align = align;
    attr.u32Bitrate = bitrate;
    attr.u32FrameRate = (U32)fps;
    attr.u32Gop = (U32)gop;
    attr.u32Profile = 0;
    attr.u32RotateDegree = 0;
    attr.eFrameBufMode = VENC_FRAME_BUF_DMABUF_EXTERNAL;  /* 用 VB 池的 dma-buf */
    attr.eRcMode = rc;
    attr.u32MinQp = (U32)min_qp;
    attr.u32MaxQp = (U32)max_qp;
    attr.u32IQp = (U32)iqp;
    attr.u32PQp = (U32)pqp;
    attr.u32BQp = (U32)pqp;

    if (VENC_CreateChn(0, &attr) != 0) { fprintf(stderr, "VENC_CreateChn 失败\n"); goto cleanup; }
    chn_created = 1;
    if (VENC_EnableChn(0) != 0) { fprintf(stderr, "VENC_EnableChn 失败\n"); goto cleanup; }
    chn_enabled = 1;

    if (rc == VENC_RC_MODE_CVBR) {
        VencRcAttr rca;
        memset(&rca, 0, sizeof(rca));
        rca.enRcMode = VENC_RC_MODE_CVBR;
        rca.u32BitRate = bitrate;
        rca.u32MaxBitRate = max_bitrate;
        rca.u32MinQp = (U32)min_qp;
        rca.u32MaxQp = (U32)max_qp;
        if (VENC_SetRateControl(0, &rca) != 0)
            fprintf(stderr, "警告: VENC_SetRateControl(CVBR) 失败\n");
    }

    fprintf(stderr,
        "[mpp_enc] %s %dx%d fps=%d gop=%d rc=%d bitrate=%u bps "
        "minQP=%d maxQP=%d iQP=%d pQP=%d buf=%d\n",
        codec == MPP_STREAM_CODEC_H265 ? "h265" : "h264",
        width, height, fps, gop, (int)rc, bitrate, min_qp, max_qp, iqp, pqp, buf_size);

    /* ---------- 4) 逐帧编码 ---------- */
    {
        const size_t y_size = (size_t)width * (size_t)height;
        const size_t uv_size = (size_t)width * (size_t)(height / 2);
        ysrc = (unsigned char *)malloc(y_size);
        uvsrc = (unsigned char *)malloc(uv_size);
        if (!ysrc || !uvsrc) { fprintf(stderr, "内存不足\n"); goto cleanup; }

        int dumped = 0;
        for (int f = 0; f < frames; f++) {
            size_t ny = fread(ysrc, 1, y_size, fin);
            size_t nuv = fread(uvsrc, 1, uv_size, fin);
            if (ny < y_size || nuv < uv_size) {
                fprintf(stderr, "[mpp_enc] 输入只有 %d 帧完整数据，提前结束\n", f);
                break;
            }

            UL buf = VB_GetBuffer(pool, 1000);
            if (buf == 0) {
                got += drain_streams(fout, &total_bytes, 0);   /* 池满先抽码流 */
                buf = VB_GetBuffer(pool, 1000);
                if (buf == 0) {
                    fprintf(stderr, "[mpp_enc] VB_GetBuffer 超时，跳过帧 %d\n", f);
                    continue;
                }
            }

            VideoFrameInfo bfi;
            memset(&bfi, 0, sizeof(bfi));
            if (VB_GetFrameInfo(buf, &bfi) != 0) {
                fprintf(stderr, "[mpp_enc] VB_GetFrameInfo 失败\n");
                VB_ReleaseBuffer(buf);
                break;
            }
            if (dump_info && !dumped) {
                fprintf(stderr,
                    "[mpp_enc] frame: planeNum=%u stride0=%u stride1=%u size0=%u size1=%u fd0=%lu\n",
                    bfi.stVFrame.u32PlaneNum,
                    bfi.stVFrame.u32PlaneStride[0], bfi.stVFrame.u32PlaneStride[1],
                    bfi.stVFrame.u32PlaneSize[0], bfi.stVFrame.u32PlaneSize[1],
                    (unsigned long)bfi.stVFrame.u32Fd[0]);
            }
            dumped = 1;

            unsigned char *base = (unsigned char *)bfi.stVFrame.ulPlaneVirAddr[0];
            if (!base) {
                VOID *va = NULL;
                if (VB_GetVirAddr(buf, &va) != 0 || !va) {
                    fprintf(stderr, "[mpp_enc] 拿不到虚拟地址\n");
                    VB_ReleaseBuffer(buf);
                    break;
                }
                base = (unsigned char *)va;
            }

            U32 stride0 = bfi.stVFrame.u32PlaneStride[0];
            if (stride0 == 0) stride0 = (U32)width;
            for (int y = 0; y < height; y++)
                memcpy(base + (size_t)y * stride0, ysrc + (size_t)y * width, (size_t)width);

            unsigned char *uvdst = (unsigned char *)bfi.stVFrame.ulPlaneVirAddr[1];
            U32 stride1 = bfi.stVFrame.u32PlaneStride[1];
            if (!uvdst) uvdst = base + (size_t)stride0 * ah;   /* 紧跟 Y 平面 */
            if (stride1 == 0) stride1 = stride0;
            for (int y = 0; y < height / 2; y++)
                memcpy(uvdst + (size_t)y * stride1, uvsrc + (size_t)y * width, (size_t)width);

            if (bfi.stVFrame.u32PlaneNum == 0) bfi.stVFrame.u32PlaneNum = 2;
            bfi.stVFrame.u64PTS = (U64)f * 1000000ULL / (U64)fps;

            if (VENC_SendFrame(0, &bfi, 1000) != 0) {
                fprintf(stderr, "[mpp_enc] VENC_SendFrame 失败（帧 %d）\n", f);
                VB_ReleaseBuffer(buf);
                break;
            }
            sent++;
            VB_ReleaseBuffer(buf);

            got += drain_streams(fout, &total_bytes, 0);
        }

        /* ---------- 5) 显式 EOS + 刷尾 ---------- */
        if (chn_enabled) {
            VideoFrameInfo eos;
            memset(&eos, 0, sizeof(eos));
            eos.eFrameType = FRAME_TYPE_VENC;
            eos.eModId = MPP_ID_VENC;
            eos.stVFrame.u32FrameFlag = MPP_FRAME_FLAG_EOS;
            eos.stVFrame.u32PlaneNum = 0;
            if (VENC_SendFrame(0, &eos, 1000) != 0)
                fprintf(stderr, "警告: 发送 EOS 失败（继续抽流）\n");
            got += drain_streams(fout, &total_bytes, 1000);
            VENC_Flush(0);
            got += drain_streams(fout, &total_bytes, 1000);
        }
    }

    {
        double secs = (double)sent / (double)fps;
        fprintf(stderr,
            "[mpp_enc] 完成: 送 %d 帧 / 取 %d 包 / %llu 字节 = %.3f Mbps "
            "(目标 %.3f Mbps, %.1fs)\n",
            sent, got, total_bytes,
            secs > 0 ? (double)total_bytes * 8.0 / secs / 1e6 : 0.0,
            (double)bitrate / 1e6, secs);
        fprintf(stderr, "[mpp_enc] 输出: %s\n", out_path);
    }
    ret = 0;

cleanup:
    if (chn_enabled) VENC_DisableChn(0);
    if (chn_created) VENC_DestroyChn(0);
    if (pool) VB_DestroyPool(pool);
    VENC_Exit();
    VB_Exit();
    free(ysrc);
    free(uvsrc);
    if (fout) fclose(fout);
    if (fin) fclose(fin);
    return ret;
}
