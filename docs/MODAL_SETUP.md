# Modal 手動セットアップ手順

## 概要
公式HuggingFace は 6ヶ月以上放置状態で信用不可。ModelScope Studio で動作確認済みの NAS 実ファイルのみを使用し、Modal Volume へ手動UPする。

**前提**: NAS に ModelScope Studio のクローン済みファイルが配置済み
```
NAS: \\AS3202T\User Homes\unfix_biz\Z-Project\LAM\
  └── assets\
      ├── pretrained_models\   (~906 MB)
      ├── sample_motion\        (~264 MB)
      ├── external\             (~4.4 MB)
      ├── wheels\               (~101 MB)
      └── blender\blender-4.0.2-linux-x64.tar.xz  (~278 MB)
  └── exps\releases\lam\lam-20k\step_045500\
      ├── config.json
      ├── model.safetensors     (~2.19 GB) ← 最重要
      └── README.md
```

---

## Step 1: Modal CLI インストール・認証

Windows PowerShell:
```powershell
pip install modal
modal token new  # ブラウザが開いて認証
```

---

## Step 2: Modal Volume 作成

```powershell
modal volume create lam-data
```

---

## Step 3: NAS → Modal Volume へUP

**Zドライブマウント前提** (`Z:` = `\\AS3202T\User Homes\unfix_biz\Z-Project\LAM`)

```powershell
# 一度だけマウント
New-PSDrive -Name Z -PSProvider FileSystem -Root "\\AS3202T\User Homes\unfix_biz\Z-Project\LAM" -Persist
```

### 3-1. 重みファイル (最重要) ~2.19 GB
```powershell
modal volume put lam-data `
  "Z:\exps\releases\lam\lam-20k\step_045500\model.safetensors" `
  /exps/releases/lam/lam-20k/step_045500/model.safetensors

modal volume put lam-data `
  "Z:\exps\releases\lam\lam-20k\step_045500\config.json" `
  /exps/releases/lam/lam-20k/step_045500/config.json

modal volume put lam-data `
  "Z:\exps\releases\lam\lam-20k\step_045500\README.md" `
  /exps/releases/lam/lam-20k/step_045500/README.md
```

### 3-2. FLAME Tracking 事前学習モデル ~906 MB
```powershell
modal volume put lam-data "Z:\assets\pretrained_models" /pretrained_models
```

### 3-3. Sample Motion データ ~264 MB
```powershell
modal volume put lam-data "Z:\assets\sample_motion" /assets/sample_motion
```

### 3-4. External ソースコード (nvdiffrast含む) ~4.4 MB
```powershell
modal volume put lam-data "Z:\assets\external" /external
```

### 3-5. Wheels ~101 MB
```powershell
modal volume put lam-data "Z:\assets\wheels" /wheels
```

### 3-6. Blender バイナリ ~278 MB
```powershell
modal volume put lam-data `
  "Z:\assets\blender\blender-4.0.2-linux-x64.tar.xz" `
  /blender-4.0.2-linux-x64.tar.xz
```

**合計**: 約 3.5 GB (家の上り回線次第で 30分〜2時間)

---

## Step 4: UP確認

```powershell
modal volume ls lam-data /
modal volume ls lam-data /exps/releases/lam/lam-20k/step_045500/
modal volume ls lam-data /pretrained_models
```

期待される出力例:
```
/exps/releases/lam/lam-20k/step_045500/
  config.json            1,109 bytes
  model.safetensors      2,356,556,212 bytes  ← この数字が一致必須
  README.md              320 bytes
```

---

## Step 5: ハッシュ検証 (最重要)

`modal_app.py` を初回起動すると、自動でハッシュ検証されます:

| ファイル | 期待 SHA256 | 期待サイズ |
|---------|-------------|-----------|
| `exps/releases/lam/lam-20k/step_045500/model.safetensors` | `f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925` | `2,356,556,212` |

不一致の場合、**起動停止**し、即座に再UPを促すエラーを出力します。

---

## Step 6: デプロイ

```powershell
# 対象ディレクトリへ cd
cd C:\Users\hamad\LAM_mirai  # またはmodal_app.pyを置いた場所

# テスト起動 (一時的なURL発行、Ctrl+Cで停止)
modal serve modal_app.py

# 本番デプロイ (永続URL)
modal deploy modal_app.py
```

---

## トラブルシュート

### UP途中で止まった場合
```powershell
# 何度でも再実行OK。既存ファイルは上書きされる
modal volume put lam-data "Z:\..." /path --force
```

### Volume の中身を全消しして作り直す場合
```powershell
modal volume rm lam-data --confirm
modal volume create lam-data
# 再度 Step 3 から
```

### Volumeのファイル容量確認
```powershell
modal volume ls lam-data / --recursive
```
