package com.fengming.glfilter;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Camera2 (SurfaceTexture, OES 外部纹理) -> fragment shader 滤镜链 -> 屏幕。
 * 单 pass：一个 fragment shader 里按 uFilter 分支，够用来演示"GPU 逐像素加工"这件事。
 */
public class FilterRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    public interface Listener {
        void onGlReady(SurfaceTexture st, int texW, int texH);
    }

    private static final String TAG = "GLFilter";

    private static final String VERT = ""
            + "attribute vec4 aPos;\n"
            + "attribute vec2 aTex;\n"
            + "uniform mat4 uST;\n"
            + "uniform int uRot;\n"
            + "varying vec2 vTexCoord;\n"
            + "varying vec2 vQuad;\n"
            + "vec2 rot(vec2 t){\n"
            + "  if(uRot==1) return vec2(t.y, 1.0-t.x);\n"
            + "  if(uRot==2) return vec2(1.0-t.x, 1.0-t.y);\n"
            + "  if(uRot==3) return vec2(1.0-t.y, t.x);\n"
            + "  return t;\n"
            + "}\n"
            + "void main(){\n"
            + "  gl_Position = aPos;\n"
            + "  vec2 t = (uST * vec4(aTex, 0.0, 1.0)).xy;\n"
            + "  vTexCoord = rot(t);\n"
            + "  vQuad = aTex;\n"
            + "}\n";

    private static final String FRAG = ""
            + "#extension GL_OES_EGL_image_external : require\n"
            + "precision mediump float;\n"
            + "varying vec2 vTexCoord;\n"
            + "varying vec2 vQuad;\n"
            + "uniform samplerExternalOES uTex;\n"
            + "uniform int   uFilter;\n"
            + "uniform float uSplit;\n"
            + "uniform vec2  uTexel;\n"
            + "uniform vec2  uRes;\n"
            + "\n"
            + "vec3 S(vec2 uv){ return texture2D(uTex, uv).rgb; }\n"
            + "float lum(vec3 c){ return dot(c, vec3(0.299,0.587,0.114)); }\n"
            + "\n"
            + "float edgeMag(vec2 uv){\n"
            + "  vec2 d = uTexel * 1.2;\n"
            + "  float tl = lum(S(uv + vec2(-d.x, -d.y)));\n"
            + "  float tc = lum(S(uv + vec2( 0.0, -d.y)));\n"
            + "  float tr = lum(S(uv + vec2( d.x, -d.y)));\n"
            + "  float ml = lum(S(uv + vec2(-d.x,  0.0)));\n"
            + "  float mr = lum(S(uv + vec2( d.x,  0.0)));\n"
            + "  float bl = lum(S(uv + vec2(-d.x,  d.y)));\n"
            + "  float bc = lum(S(uv + vec2( 0.0,  d.y)));\n"
            + "  float br = lum(S(uv + vec2( d.x,  d.y)));\n"
            + "  float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);\n"
            + "  float gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);\n"
            + "  return sqrt(gx*gx + gy*gy);\n"
            + "}\n"
            + "\n"
            // 1 磨皮：双边近似（空间一圈 + 值域权重保边），再按肤色区间提亮
            + "vec3 fBeauty(vec2 uv, vec3 c){\n"
            + "  vec3 sum = c; float wsum = 1.0;\n"
            + "  float r = 3.5;\n"
            + "  for(int i=0;i<8;i++){\n"
            + "    float a = float(i) * 0.7853982;\n"
            + "    vec2 d = vec2(cos(a), sin(a)) * r * uTexel;\n"
            + "    vec3 s = S(uv + d);\n"
            + "    vec3 diff = s - c;\n"
            + "    float w = exp(-dot(diff, diff) * 14.0);\n"
            + "    sum += s * w; wsum += w;\n"
            + "  }\n"
            + "  vec3 sm = sum / wsum;\n"
            + "  float l = lum(sm);\n"
            + "  float skin = smoothstep(0.12, 0.34, l) * (1.0 - smoothstep(0.82, 1.0, l));\n"
            + "  vec3 o = mix(c, sm, 0.90);\n"
            + "  o *= (1.0 + 0.13 * skin);\n"
            + "  o.r = min(1.0, o.r + 0.035 * skin);\n"
            + "  o.g = min(1.0, o.g + 0.012 * skin);\n"
            + "  return clamp(o, 0.0, 1.0);\n"
            + "}\n"
            // 2 美白：提亮 + 降饱和 + 暖调
            + "vec3 fWhiten(vec3 c){\n"
            + "  float l = lum(c);\n"
            + "  vec3 t = mix(c, vec3(l), 0.30);\n"
            + "  t = t * 1.10 + 0.055;\n"
            + "  t.r = min(1.0, t.r * 1.03 + 0.012);\n"
            + "  t.b = min(1.0, t.b * 0.985);\n"
            + "  float m = smoothstep(0.02, 0.45, l);\n"
            + "  return clamp(mix(c, t, 0.55 + 0.45 * m), 0.0, 1.0);\n"
            + "}\n"
            // 3 卡通：色调分离 + 描边压暗
            + "vec3 fToon(vec2 uv, vec3 c){\n"
            + "  vec3 q = floor(c * 5.0 + 0.5) / 5.0;\n"
            + "  float e = edgeMag(uv);\n"
            + "  float line = smoothstep(0.10, 0.30, e);\n"
            + "  q *= (1.0 - 0.92 * line);\n"
            + "  return clamp(q, 0.0, 1.0);\n"
            + "}\n"
            // 4 素描
            + "vec3 fSketch(vec2 uv){\n"
            + "  float e = edgeMag(uv);\n"
            + "  float d = 1.0 - smoothstep(0.04, 0.40, e);\n"
            + "  float l = lum(S(uv));\n"
            + "  float v = d * (0.72 + 0.28 * l);\n"
            + "  v = pow(clamp(v, 0.0, 1.0), 0.85);\n"
            + "  return vec3(v * 0.98, v * 0.98, v * 0.96);\n"
            + "}\n"
            // 5 浮雕
            + "vec3 fEmboss(vec2 uv){\n"
            + "  vec2 d = uTexel * 1.6;\n"
            + "  float a = lum(S(uv + d));\n"
            + "  float b = lum(S(uv - d));\n"
            + "  float v = 0.5 + (a - b) * 2.2;\n"
            + "  vec3 c = vec3(clamp(v, 0.0, 1.0));\n"
            + "  return mix(c, c * vec3(1.02, 0.99, 0.95), 1.0);\n"
            + "}\n"
            // 6 老照片：sepia + 暗角 + 颗粒
            + "vec3 fSepia(vec2 uv, vec3 c){\n"
            + "  float l = lum(c);\n"
            + "  vec3 s = vec3(min(1.0, l * 1.08 + 0.05),\n"
            + "                min(1.0, l * 0.95 + 0.02),\n"
            + "                min(1.0, l * 0.74));\n"
            + "  vec2 p = gl_FragCoord.xy / uRes - 0.5;\n"
            + "  float vg = 1.0 - dot(p, p) * 1.05;\n"
            + "  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);\n"
            + "  s = s * vg + (n - 0.5) * 0.055;\n"
            + "  return clamp(s, 0.0, 1.0);\n"
            + "}\n"
            // 7 伪彩（热成像风格）：亮度映射到色带
            + "vec3 heat(float t){\n"
            + "  t = clamp(t, 0.0, 1.0);\n"
            + "  vec3 c1 = vec3(0.03, 0.03, 0.30);\n"
            + "  vec3 c2 = vec3(0.00, 0.55, 0.95);\n"
            + "  vec3 c3 = vec3(0.10, 0.90, 0.35);\n"
            + "  vec3 c4 = vec3(1.00, 0.85, 0.10);\n"
            + "  vec3 c5 = vec3(0.95, 0.15, 0.10);\n"
            + "  if(t < 0.25) return mix(c1, c2, t / 0.25);\n"
            + "  if(t < 0.50) return mix(c2, c3, (t - 0.25) / 0.25);\n"
            + "  if(t < 0.75) return mix(c3, c4, (t - 0.50) / 0.25);\n"
            + "  return mix(c4, c5, (t - 0.75) / 0.25);\n"
            + "}\n"
            + "vec3 fThermal(vec3 c){\n"
            + "  float l = lum(c);\n"
            + "  l = pow(clamp(l * 1.15, 0.0, 1.0), 1.15);\n"
            + "  return heat(l);\n"
            + "}\n"
            // 8 边缘检测（反色霓虹）
            + "vec3 fEdge(vec2 uv){\n"
            + "  float e = edgeMag(uv);\n"
            + "  float v = smoothstep(0.05, 0.55, e);\n"
            + "  return clamp(vec3(v * 0.35, v * 0.95, v * 0.75) + 0.03, 0.0, 1.0);\n"
            + "}\n"
            + "\n"
            + "void main(){\n"
            + "  vec3 c = S(vTexCoord);\n"
            + "  vec3 outColor;\n"
            + "  if(uFilter == 0)      outColor = c;\n"
            + "  else if(uFilter == 1) outColor = fBeauty(vTexCoord, c);\n"
            + "  else if(uFilter == 2) outColor = fWhiten(c);\n"
            + "  else if(uFilter == 3) outColor = fToon(vTexCoord, c);\n"
            + "  else if(uFilter == 4) outColor = fSketch(vTexCoord);\n"
            + "  else if(uFilter == 5) outColor = fEmboss(vTexCoord);\n"
            + "  else if(uFilter == 6) outColor = fSepia(vTexCoord, c);\n"
            + "  else if(uFilter == 7) outColor = fThermal(c);\n"
            + "  else                  outColor = fEdge(vTexCoord);\n"
            // 分屏对比：uSplit>0 时左半边(uSplit 比例)显示原图
            + "  float sx = gl_FragCoord.x / uRes.x;\n"
            + "  if(uSplit > 0.0 && sx < uSplit) {\n"
            + "    outColor = c;\n"
            + "  }\n"
            + "  gl_FragColor = vec4(outColor, 1.0);\n"
            + "}\n";

    private final Listener listener;
    private int texW = 1280, texH = 720;
    private int rotation = 1;

    private int program;
    private int texHandle;
    private SurfaceTexture surfaceTexture;
    private int posLoc, texLoc, stLoc, rotLoc, filterLoc, splitLoc, texelLoc, resLoc;

    private FloatBuffer posBuf;
    private FloatBuffer texBuf;
    private final float[] stMatrix = new float[16];

    private volatile int filter = 0;
    private volatile float split = 0f;
    private volatile boolean frameAvailable = false;

    private int viewW = 1, viewH = 1;
    private volatile boolean sizeDirty = false;
    private long frameCount = 0, statStart = 0;
    private long drawNanos = 0;

    public FilterRenderer(Listener l) {
        this.listener = l;
    }

    public void setFilter(int f) {
        this.filter = f;
    }

    public void setSplit(float s) {
        this.split = s;
    }

    public void setRotation(int r) {
        this.rotation = r & 3;
        this.sizeDirty = true;
    }

    public void setTexSize(int w, int h) {
        if (w > 0 && h > 0) {
            this.texW = w;
            this.texH = h;
            this.sizeDirty = true;
        }
    }

    public SurfaceTexture getSurfaceTexture() {
        return surfaceTexture;
    }

    private static int load(int type, String src) {
        int s = GLES20.glCreateShader(type);
        GLES20.glShaderSource(s, src);
        GLES20.glCompileShader(s);
        int[] ok = new int[1];
        GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0);
        if (ok[0] == 0) {
            Log.e(TAG, "shader compile failed: " + GLES20.glGetShaderInfoLog(s));
            throw new RuntimeException("shader: " + GLES20.glGetShaderInfoLog(s));
        }
        return s;
    }

    @Override
    public void onSurfaceCreated(GL10 unused, EGLConfig config) {
        int vs = load(GLES20.GL_VERTEX_SHADER, VERT);
        int fs = load(GLES20.GL_FRAGMENT_SHADER, FRAG);
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vs);
        GLES20.glAttachShader(program, fs);
        GLES20.glLinkProgram(program);
        int[] ok = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, ok, 0);
        if (ok[0] == 0) {
            Log.e(TAG, "link failed: " + GLES20.glGetProgramInfoLog(program));
        }
        posLoc = GLES20.glGetAttribLocation(program, "aPos");
        texLoc = GLES20.glGetAttribLocation(program, "aTex");
        stLoc = GLES20.glGetUniformLocation(program, "uST");
        rotLoc = GLES20.glGetUniformLocation(program, "uRot");
        filterLoc = GLES20.glGetUniformLocation(program, "uFilter");
        splitLoc = GLES20.glGetUniformLocation(program, "uSplit");
        texelLoc = GLES20.glGetUniformLocation(program, "uTexel");
        resLoc = GLES20.glGetUniformLocation(program, "uRes");

        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        texHandle = tex[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texHandle);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        surfaceTexture = new SurfaceTexture(texHandle);
        surfaceTexture.setOnFrameAvailableListener(this);

        float[] t = {0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f};
        texBuf = ByteBuffer.allocateDirect(t.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        texBuf.put(t).position(0);

        GLES20.glClearColor(0f, 0f, 0f, 1f);
        listener.onGlReady(surfaceTexture, texW, texH);
    }

    @Override
    public void onSurfaceChanged(GL10 unused, int w, int h) {
        viewW = Math.max(1, w);
        viewH = Math.max(1, h);
        GLES20.glViewport(0, 0, viewW, viewH);
        rebuildQuad();
    }

    private void rebuildQuad() {
        // 视频按 9:16 竖屏显示，做 letterbox；旋转 90° 后宽高互换
        float videoAspect = (rotation % 2 == 0)
                ? (float) texW / (float) texH
                : (float) texH / (float) texW;
        float viewAspect = (float) viewW / (float) viewH;
        float sx = 1f, sy = 1f;
        if (videoAspect > viewAspect) {
            sy = viewAspect / videoAspect;
        } else {
            sx = videoAspect / viewAspect;
        }
        float[] p = {-sx, -sy, sx, -sy, -sx, sy, sx, sy};
        posBuf = ByteBuffer.allocateDirect(p.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        posBuf.put(p).position(0);
    }

    @Override
    public void onDrawFrame(GL10 unused) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        if (sizeDirty) {
            sizeDirty = false;
            rebuildQuad();
        }

        if (frameAvailable) {
            frameAvailable = false;
            try {
                surfaceTexture.updateTexImage();
            } catch (Throwable t) {
                Log.w(TAG, "updateTexImage: " + t);
            }
        }
        surfaceTexture.getTransformMatrix(stMatrix);

        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texHandle);

        GLES20.glUniformMatrix4fv(stLoc, 1, false, stMatrix, 0);
        GLES20.glUniform1i(rotLoc, rotation);
        GLES20.glUniform1i(filterLoc, filter);
        GLES20.glUniform1f(splitLoc, split);
        GLES20.glUniform2f(texelLoc, 1f / texW, 1f / texH);
        GLES20.glUniform2f(resLoc, viewW, viewH);

        GLES20.glEnableVertexAttribArray(posLoc);
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 0, posBuf);
        GLES20.glEnableVertexAttribArray(texLoc);
        GLES20.glVertexAttribPointer(texLoc, 2, GLES20.GL_FLOAT, false, 0, texBuf);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(posLoc);
        GLES20.glDisableVertexAttribArray(texLoc);

        // 单帧 GPU 耗时（glFinish 之后才返回，等于等 GPU 真的画完）
        long t0 = System.nanoTime();
        GLES20.glFinish();
        drawNanos += System.nanoTime() - t0;

        frameCount++;
        if (statStart == 0) statStart = System.nanoTime();
        long dt = System.nanoTime() - statStart;
        if (frameCount % 60 == 0 && dt > 0) {
            Log.i(TAG, String.format("PERF filter=%d res=%dx%d fps=%.1f gpu_ms=%.2f",
                    filter, viewW, viewH, frameCount * 1e9 / dt, drawNanos / 60.0 / 1e6));
            drawNanos = 0;
            statStart = System.nanoTime();
            frameCount = 0;
        }
    }

    @Override
    public void onFrameAvailable(SurfaceTexture st) {
        frameAvailable = true;
        requestRender();
    }

    private GLSurfaceView view;

    public void attach(GLSurfaceView v) {
        this.view = v;
    }

    private void requestRender() {
        if (view != null) view.requestRender();
    }
}
