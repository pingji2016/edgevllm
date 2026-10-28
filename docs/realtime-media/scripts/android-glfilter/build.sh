#!/usr/bin/env bash
# 不用 gradle：aapt2 -> javac -> d8 -> zipalign -> apksigner
set -euo pipefail

SDK="${ANDROID_SDK:-$HOME/Library/Android/sdk}"
BT="$SDK/build-tools/35.0.0"
PLAT="$SDK/platforms/android-35/android.jar"
JDK_BIN="$(dirname "$(readlink -f "$(command -v javac)")")"
KEYSTORE="$(cd "$(dirname "$0")" && pwd)/debug.jks"

cd "$(dirname "$0")"
rm -rf build && mkdir -p build/obj build/dex

echo "== aapt2 link"
"$BT/aapt2" link -o build/base.apk -I "$PLAT" \
  --manifest AndroidManifest.xml -A assets \
  --min-sdk-version 21 --target-sdk-version 34 \
  --version-code 1 --version-name 1.0

echo "== javac"
"$JDK_BIN/javac" -source 8 -target 8 -Xlint:-options -nowarn \
  -classpath "$PLAT" -d build/obj \
  $(find src -name '*.java')

echo "== d8"
"$BT/d8" --min-api 21 --lib "$PLAT" --release \
  --output build/dex $(find build/obj -name '*.class')

echo "== package"
(cd build/dex && zip -q -j ../base.apk classes.dex)
"$BT/zipalign" -f -p 4 build/base.apk build/aligned.apk

if [ ! -f "$KEYSTORE" ]; then
  echo "== gen debug keystore"
  "$JDK_BIN/keytool" -genkeypair -keystore "$KEYSTORE" -storepass android \
    -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 \
    -validity 10000 -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1
fi

echo "== sign"
"$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
  --min-sdk-version 21 --out build/glfilter.apk build/aligned.apk

"$BT/apksigner" verify build/glfilter.apk && echo "OK -> build/glfilter.apk ($(du -h build/glfilter.apk | cut -f1))"
