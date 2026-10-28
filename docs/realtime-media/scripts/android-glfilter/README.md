# android-glfilter —— 真机 GLSL 滤镜链

配套正文：[06-android-glsl-filter-chain.md](../../06-android-glsl-filter-chain.md)

```
Camera2 ─┐
         ├─► SurfaceTexture ─► GL_TEXTURE_EXTERNAL_OES ─► fragment shader ─► EGL 窗口
MediaPlayer ─┘                     （零拷贝，不经过 CPU）
```

2 个 Java 文件、800 行，无资源文件、无 gradle。

## 编译

需要 Android SDK（build-tools 35 + platform android-35）和 JDK：

```bash
export ANDROID_SDK=$HOME/Library/Android/sdk   # 按自己的路径改
./build.sh
adb install -r build/glfilter.apk
```

`build.sh` 就是五条命令：`aapt2 link` → `javac` → `d8` → `zipalign` → `apksigner`。
首次运行会顺手生成一个 debug keystore（密码统一 `android`）。

## 运行

```bash
# 1) 实时相机（facing 0 = 后摄，1 = 前摄；rot 是纹理旋转，相机一般 1）
adb shell am start -n com.fengming.glfilter/.MainActivity --ei facing 0 --ei rot 1

# 2) 视频文件（MediaPlayer 解码 → 同一个 shader）
#    需要先把素材推到 assets/（见下），或改用 --es video /sdcard/... 走公共同路径
adb shell am start -n com.fengming.glfilter/.MainActivity \
  --ez land true --ei rot 0 --es asset demo.mp4

# 3) 自动演示：每 2.6 s 换一个滤镜 + 分屏对比（录屏用）
adb shell am start -n com.fengming.glfilter/.MainActivity \
  --ez demo true --ez land true --ei rot 0 --es asset demo.mp4 --ez split true
```

| intent extra | 类型 | 作用 |
| --- | --- | --- |
| `facing` | int | 0 后摄 / 1 前摄 |
| `rot` | int | 纹理旋转，0–3 个 90° |
| `land` | bool | 强制横屏 |
| `asset` | string | 从 APK `assets/` 读视频（推荐） |
| `video` | string | 从文件系统路径读视频 |
| `filter` | int | 启动时选中的效果（0–8） |
| `split` | bool | 左半屏原图 / 右半屏效果 |
| `demo` | bool | 自动轮播 + 隐藏提示文字 |

交互：**点一下切滤镜，长按切分屏对比**。

## 素材（可选）

`assets/demo.mp4` 不随仓库分发（4 MB）。想复现录屏效果，用任意 16:9 素材即可；
本次用的是 picsum（Unsplash 授权）的街景图做 50 s 慢推镜头：

```bash
ffmpeg -loop 1 -i street.jpg \
  -vf "zoompan=z='min(zoom+0.00055,1.30)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1500:s=1280x720:fps=30" \
  -t 50 -c:v libx264 -crf 21 -pix_fmt yuv420p assets/demo.mp4
```

## 结构

| 文件 | 内容 |
| --- | --- |
| `MainActivity.java` | Camera2 / MediaPlayer 双输入、EGL 上下文、触摸手势、intent 参数、演示轮播 |
| `FilterRenderer.java` | 着色器源码（9 个效果）、OES 纹理与变换矩阵、分屏、`glFinish` 计时打点 |

计时日志（`adb logcat -s GLFilter:*`）：

```
PERF filter=1 res=2272x1080 fps=30.0 gpu_ms=4.67
```

`gpu_ms` 是 `glDrawArrays` + `glFinish` 的墙钟时间，含呈现等待 —— 所以原图也有 4 ms。
详见正文第四节。
