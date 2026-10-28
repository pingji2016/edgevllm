package com.fengming.glfilter;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.opengl.GLSurfaceView;
import android.media.MediaPlayer;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.TextView;

import java.util.Collections;

public class MainActivity extends Activity {

    private static final String TAG = "GLFilter";
    private static final int REQ_CAM = 1001;

    private static final String[] NAMES = {
            "原图（无处理）",
            "磨皮（双边近似，保边）",
            "美白（提亮 + 降饱和 + 暖调）",
            "卡通（色调分离 + 描边）",
            "素描（边缘 + 明暗）",
            "浮雕",
            "老照片（sepia + 暗角 + 颗粒）",
            "伪彩（热成像风格）",
            "边缘检测（霓虹）"
    };

    private GLSurfaceView glView;
    private FilterRenderer renderer;
    private TextView label;
    private TextView hint;

    private HandlerThread bgThread;
    private Handler bgHandler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private String cameraId;

    private int texW = 1280, texH = 720;
    private int filter = 0;
    private boolean demoMode = false;
    private boolean splitOn = false;
    private int rotation = 1;
    private int facing = 0;
    private String videoPath = null;
    private String assetName = null;
    private MediaPlayer player;

    private final Handler ui = new Handler(Looper.getMainLooper());

    private final Runnable demoStep = new Runnable() {
        @Override
        public void run() {
            if (!demoMode) return;
            filter = (filter + 1) % NAMES.length;
            renderer.setFilter(filter);
            updateLabel();
            ui.postDelayed(this, 2600);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);

        demoMode = getIntent().getBooleanExtra("demo", false);
        rotation = getIntent().getIntExtra("rot", 1);
        facing = getIntent().getIntExtra("facing", 0);
        videoPath = getIntent().getStringExtra("video");
        assetName = getIntent().getStringExtra("asset");
        if (getIntent().getBooleanExtra("land", false)) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        }
        filter = getIntent().getIntExtra("filter", 0);
        splitOn = getIntent().getBooleanExtra("split", demoMode);

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        renderer = new FilterRenderer(new FilterRenderer.Listener() {
            @Override
            public void onGlReady(SurfaceTexture st, int w, int h) {
                st.setDefaultBufferSize(w, h);
                ui.post(new Runnable() {
                    @Override
                    public void run() {
                        if (videoPath != null || assetName != null) startVideo();
                        else openCamera();
                    }
                });
            }
        });
        renderer.attach(glView);
        renderer.setRotation(rotation);
        renderer.setFilter(filter);
        renderer.setSplit(splitOn ? 0.5f : 0f);
        glView.setRenderer(renderer);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        root.addView(glView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        label = new TextView(this);
        label.setTextColor(Color.WHITE);
        label.setTextSize(18f);
        label.setGravity(Gravity.START);
        label.setPadding(36, 60, 36, 0);
        label.setShadowLayer(6f, 0f, 2f, Color.BLACK);
        root.addView(label, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        hint = new TextView(this);
        hint.setTextColor(0xFFB0B0B0);
        hint.setTextSize(13f);
        hint.setGravity(Gravity.CENTER_HORIZONTAL);
        hint.setPadding(0, 0, 0, 48);
        hint.setShadowLayer(6f, 0f, 2f, Color.BLACK);
        hint.setText("点一下切换滤镜 · 长按切换左半边原图对比");
        if (demoMode) hint.setVisibility(View.GONE);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.gravity = Gravity.BOTTOM;
        root.addView(hint, lp);

        glView.setOnTouchListener(new View.OnTouchListener() {
            private long downAt;

            @Override
            public boolean onTouch(View v, MotionEvent e) {
                if (e.getAction() == MotionEvent.ACTION_DOWN) {
                    downAt = System.currentTimeMillis();
                    return true;
                }
                if (e.getAction() == MotionEvent.ACTION_UP) {
                    long dt = System.currentTimeMillis() - downAt;
                    if (dt > 600) {
                        splitOn = !splitOn;
                        renderer.setSplit(splitOn ? 0.5f : 0f);
                    } else {
                        filter = (filter + 1) % NAMES.length;
                        renderer.setFilter(filter);
                    }
                    updateLabel();
                    return true;
                }
                return false;
            }
        });

        setContentView(root);

        bgThread = new HandlerThread("camera");
        bgThread.start();
        bgHandler = new Handler(bgThread.getLooper());

        updateLabel();

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAM);
        }

        if (demoMode) {
            ui.postDelayed(demoStep, 3000);
        }
    }

    private void updateLabel() {
        String s = NAMES[filter];
        if (demoMode) s = s + "   [demo]";
        if (splitOn) s = s + "\n左：原图 / 右：效果";
        label.setText(s);
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (code == REQ_CAM && res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        }
    }

    @SuppressWarnings("MissingPermission")
    private void openCamera() {
        if (camera != null) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        try {
            CameraManager cm = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            int want = (facing == 1) ? CameraCharacteristics.LENS_FACING_FRONT
                                     : CameraCharacteristics.LENS_FACING_BACK;
            String pick = null;
            for (String id : cm.getCameraIdList()) {
                CameraCharacteristics ch = cm.getCameraCharacteristics(id);
                Integer f = ch.get(CameraCharacteristics.LENS_FACING);
                if (f != null && f == want) {
                    pick = id;
                    break;
                }
            }
            if (pick == null) pick = cm.getCameraIdList()[0];
            cameraId = pick;

            CameraCharacteristics ch = cm.getCameraCharacteristics(pick);
            StreamConfigurationMap map =
                    ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map != null) {
                Size sz = chooseSize(map);
                if (sz != null) {
                    texW = sz.getWidth();
                    texH = sz.getHeight();
                    renderer.setTexSize(texW, texH);
                    SurfaceTexture st = renderer.getSurfaceTexture();
                    if (st != null) st.setDefaultBufferSize(texW, texH);
                }
            }
            Log.i(TAG, "open camera " + pick + " preview " + texW + "x" + texH);
            cm.openCamera(pick, deviceCallback, bgHandler);
        } catch (Throwable t) {
            Log.e(TAG, "openCamera failed: " + t);
            label.setText("打开相机失败: " + t);
        }
    }

    /** 视频文件模式：MediaPlayer 解码 → SurfaceTexture(OES) → 同一套 shader 链 */
    private void startVideo() {
        try {
            player = new MediaPlayer();
            if (assetName != null) {
                android.content.res.AssetFileDescriptor afd = getAssets().openFd(assetName);
                player.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
            } else {
                player.setDataSource(videoPath);
            }
            player.setSurface(new Surface(renderer.getSurfaceTexture()));
            player.setLooping(true);
            player.setVolume(0f, 0f);
            player.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                @Override
                public void onPrepared(MediaPlayer mp) {
                    renderer.setTexSize(mp.getVideoWidth(), mp.getVideoHeight());
                    mp.start();
                    Log.i(TAG, "video started " + mp.getVideoWidth() + "x" + mp.getVideoHeight());
                }
            });
            player.setOnErrorListener(new MediaPlayer.OnErrorListener() {
                @Override
                public boolean onError(MediaPlayer mp, int what, int extra) {
                    Log.e(TAG, "player error " + what + "/" + extra);
                    return false;
                }
            });
            player.prepareAsync();
        } catch (Throwable t) {
            Log.e(TAG, "startVideo: " + t);
            label.setText("视频失败: " + t);
        }
    }

    private Size chooseSize(StreamConfigurationMap map) {        Size[] sizes = map.getOutputSizes(SurfaceTexture.class);
        if (sizes == null || sizes.length == 0) return null;
        Size fallback = sizes[0];
        long bestArea = 0;
        for (Size s : sizes) {
            if (s.getWidth() == 1280 && s.getHeight() == 720) return s;
        }
        for (Size s : sizes) {
            if (s.getWidth() == 1920 && s.getHeight() == 1080) return s;
        }
        for (Size s : sizes) {
            long a = (long) s.getWidth() * s.getHeight();
            if (a > bestArea && a <= 1920L * 1080L) {
                bestArea = a;
                fallback = s;
            }
        }
        return fallback;
    }

    private final CameraDevice.StateCallback deviceCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice dev) {
            camera = dev;
            startPreview();
        }

        @Override
        public void onDisconnected(CameraDevice dev) {
            dev.close();
            camera = null;
        }

        @Override
        public void onError(CameraDevice dev, int error) {
            Log.e(TAG, "camera error " + error);
            dev.close();
            camera = null;
            ui.post(new Runnable() {
                @Override
                public void run() {
                    label.setText("相机错误");
                }
            });
        }
    };

    private void startPreview() {
        try {
            SurfaceTexture st = renderer.getSurfaceTexture();
            if (st == null || camera == null) return;
            final Surface surface = new Surface(st);
            camera.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession s) {
                            session = s;
                            try {
                                CaptureRequest.Builder b =
                                        camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                b.addTarget(surface);
                                b.set(CaptureRequest.CONTROL_MODE,
                                        CaptureRequest.CONTROL_MODE_AUTO);
                                b.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                b.set(CaptureRequest.CONTROL_AE_MODE,
                                        CaptureRequest.CONTROL_AE_MODE_ON);
                                session.setRepeatingRequest(b.build(), null, bgHandler);
                            } catch (Throwable t) {
                                Log.e(TAG, "setRepeatingRequest: " + t);
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession s) {
                            Log.e(TAG, "configure failed");
                        }
                    }, bgHandler);
        } catch (Throwable t) {
            Log.e(TAG, "startPreview: " + t);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        glView.onResume();
        glView.requestRender();
    }

    @Override
    protected void onPause() {
        super.onPause();
        ui.removeCallbacks(demoStep);
        if (session != null) {
            session.close();
            session = null;
        }
        if (camera != null) {
            camera.close();
            camera = null;
        }
        if (player != null) {
            try {
                player.release();
            } catch (Throwable ignored) {
            }
            player = null;
        }
        glView.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (bgThread != null) bgThread.quitSafely();
    }
}
