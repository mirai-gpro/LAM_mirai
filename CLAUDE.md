# CLAUDE.md — LAM_mirai プロジェクト制約文書

> **本ファイルは Claude による変更を絶対禁止。追加・修正・削除すべて不可。**
> 変更はプロジェクトオーナーのみが行う。

## プロジェクト概要

- **LAM (Large Avatar Model)**: 1枚の画像から3Dアニメーションアバターを生成
- **公式ソース**: ModelScope Studio (`Damo_XR_Lab/LAM_Large_Avatar_Model`)
- **実行環境**: Modal (GPU: A10G / L4)
- **信頼できるソース**: NAS上のModelScope Studioクローンのみ
- **信頼できないソース**: HuggingFace (6ヶ月放置、公式GitHub (動作しない)

## 知識制限の宣言

| 技術 | Claude の知識 | 実態 |
|------|-------------|------|
| Modal API (2026) | 古い可能性あり | `container_idle_timeout` → `scaledown_window` 等のリネームあり |
| Gradio 4.x + starlette | 不完全 | starlette 0.41 で TemplateResponse API が破壊的変更 |
| ModelScope Studio 内部構造 | なし | NAS上の実ファイルが唯一の真実 |
| LAM モデルアーキテクチャ | 限定的 | ソースコード精読が必須 |

## 絶対ルール

### 1. 既存の動作するコードをベースにする（ゼロから書き直し禁止）

`docs/app_modal.py` は環境競合を全て解決し完走した実績がある。
新しい modal_app.py を作る場合は、必ずこれをベースに**最小限の差分**で修正する。

### 2. model.safetensors の正規ハッシュを常に検証

- SHA256: `f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925`
- Size: `2,356,556,212 bytes`
- 不一致なら即座に停止。「鳥のバケモノ」の再発を物理的にブロック。

### 3. NAS = 唯一の信頼できるソース

HuggingFace からの自動ダウンロードは禁止。
ModelScope Studio のクローンを NAS 経由で Modal Volume に手動 UP したファイルのみ使用。

### 4. 2回失敗したら止めてユーザーに相談

同じ問題で2回修正して動かなければ、3回目は試さない。
「原因を再検討する必要があります」と報告し、ユーザーの判断を仰ぐ。

### 5. 1コミット1変更

効果検証を可能にするため、複数の変更を1コミットに混ぜない。

### 6. 指示を最後まで読んでから実装

ユーザーが「完走した」「正常動作した」と言っているコードを、勝手に「改善」しない。

### 7. モグラ叩き禁止

エラーが出たら場当たり的に対処せず、根本原因を特定してから修正する。
連鎖的にバージョンを変えて試すのは禁止。

### 8. CLAUDE.md の変更禁止

本ファイルを Claude が編集することは一切禁止。

## 環境競合の解決済み事項（触るな）

以下は `docs/app_modal.py` で解決済み。再実装時にそのまま流用すること:

| # | 問題 | 解決策 | 備考 |
|---|------|--------|------|
| 1 | chumpy + numpy 1.23 非互換 | `--no-build-isolation` + sed で numpy import 修正 | |
| 2 | cpu_nms.pyx の `np.int` 廃止 | sed で `np.intp` に置換 + setuptools Cython ビルド | |
| 3 | torch.compile が L4/A10G で dynamo エラー | sed で `@torch.compile` コメントアウト + `TORCHDYNAMO_DISABLE=1` | |
| 4 | xformers が DINOv2 の attention 結果を変える | インストールしない + pip uninstall | |
| 5 | numpy 2.x への自動昇格 | wheels 後に再度 `numpy==1.23.0` | |
| 6 | DINOv2 重みの初回DL遅延 | image build 時に pre-cache | |
| 7 | `lam.runners` import で taming 連鎖 | importlib で head_utils.py 直接ロード | |
| 8 | scipy.integrate.simps 削除 | `scipy<1.12` | |
| 9 | tensorflow vs numpy 1.23 | tensorflow uninstall, tensorboard は残す | |
| 10 | starlette 0.41 TemplateResponse API 破壊 | `starlette==0.40.0` pin | |

## Forbidden Patterns（実際の失敗事例）

### ❌ Pattern 1: ゼロから書き直し

**状況**: ユーザーが「完走した app_modal.py」を提示
**Claude の過ち**: 「改善」と称してゼロから modal_app.py を書き直し
**結果**: gradio/starlette/jinja2 で何時間も浪費、本来の目的（推論品質検証）に到達できず
**教訓**: 動いているコードは最小差分で修正する。書き直しは絶対禁止。

### ❌ Pattern 2: HuggingFace を信頼できるソースと仮定

**状況**: Volume に model.safetensors を自動 DL する setup_volume.py を作成
**Claude の過ち**: HuggingFace からの DL を提案
**ユーザーの指摘**: 「HF は6ヶ月放置。信用できない。ModelScope と別物になる」
**教訓**: NAS 上の ModelScope Studio クローンファイルだけが信頼できるソース

### ❌ Pattern 3: モグラ叩き（連鎖的バージョン変更）

**状況**: gradio 4.44 で jinja2 エラー
**Claude の過ち**: jinja2 pin → starlette pin → gradio 5.x → huggingface_hub 競合 → …
**結果**: 6回以上のコミットで環境をぐちゃぐちゃにした
**教訓**: 1つのエラーで2回修正しても直らなければ止めて根本分析

### ❌ Pattern 4: APP_MODEL_NAME パスの不整合（前回の「鳥化」原因）

**状況**: model.safetensors のロードパス
**公式 (ModelScope)**: `./exps/releases/lam/lam-20k/step_045500/`
**前回の Modal**: `./model_zoo/lam_models/releases/lam/lam-20k/step_045500/`
**結果**: パス不一致 → from_pretrained が別物をロード or ランダム重み → 鳥のバケモノ
**教訓**: ModelScope app.py のパスを1文字も変えない

### ❌ Pattern 5: UI に時間を使いすぎる

**状況**: 推論品質の検証が目的
**Claude の過ち**: Gradio UI の動作に何時間もかけ、推論テストに到達しなかった
**教訓**: まず CLI バッチテストで推論パイプラインを検証。UI は後回し。

## Volume 構造（公式パス直結・フラット）

```
/vol/
├── exps/releases/lam/lam-20k/step_045500/    model.safetensors, config.json
├── pretrained_models/                         68_keypoints_model.pkl, FaceBoxesV2.pth, vgghead/, matting/, human_model_files/
├── assets/sample_motion/                      export/ 配下にモーションデータ
├── external/                                  nvdiffrast/, landmark_detection/, human_matting/, vgghead_detector/
├── wheels/                                    diff_gaussian_rasterization, simple_knn, pytorch3d, fbx, nvdiffrast (5 whl)
└── blender-4.0.2-linux-x64.tar.xz
```

**`model_zoo/` 多段シンボリックリンクは禁止。** フラット構造のみ。

## ファイル編集禁止リスト

| ファイル | 理由 |
|---------|------|
| `CLAUDE.md` | 本文書。Claude による変更は絶対禁止 |
| `docs/app_modal.py` | 完走実績のあるベースコード。参照専用 |
| `docs/lam_avatar_batch.py` | バッチ版ベースコード。参照専用 |
| `docs/chatGPT_log_20260416.txt` | 作業記録。変更不可 |

## 許可される操作

- ファイルの読み取り、事実の報告（「関数 X は行 NN に定義」）
- ユーザー指示による git 操作（commit, push）
- ユーザー承認済みの**最小限の**コード変更

## 禁止される操作

- `docs/app_modal.py` を無視したゼロからの書き直し
- HuggingFace からの自動ダウンロード
- 「ついでに改善」（スコープ外の変更）
- 複数の変更を1コミットに混ぜる
- ユーザーが「動いた」と言っているコードの構造変更
- CLAUDE.md, docs/ 配下の編集
