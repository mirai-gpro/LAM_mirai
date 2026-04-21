# 顔補正メソッド（LAM 日本人化頂点補正）

## 目的

LAM (Large Avatar Model) は FLAME (Faces Learned with an Articulated Model
and Expressions) を 3D 顔テンプレートに採用している。FLAME は約 3,800 人
の 3D スキャンで学習されているが、被験者の大半が欧米人であるため、日本人
画像を入力すると以下の系統的バイアスがかかる。

| 部位 | FLAME(欧米人ベース) | 日本人 | 結果 |
|------|---------------------|--------|------|
| 鼻   | 高い・狭い          | 低い・広い | 鼻が高く狭く変形 |
| 頬   | 縦長・彫り深い      | 平坦・丸い | 頬が縦に伸びる |
| 顎   | 突出・尖る          | 小さめ・丸い | 顎が前に出る |
| 目   | 大きく開く          | やや細い   | 切れ長に誇張 |
| 人中 | 長い                | 短い       | 鼻下が間延び |

本メソッドは、推論パイプライン後の OAC ZIP 生成時に **メッシュと Gaussian
の頂点座標を直接補正** することで、欧米人骨格バイアスを後段で打ち消す。
LAM 本体・FLAME 重み・推論コードは一切変更しない。

## OAC ZIP の構造と二重補正の必要性

OAC (Open Avatar Container) ZIP は 2 レイヤーで構成される。

| ファイル          | 役割                               | 補正対象 |
|-------------------|------------------------------------|----------|
| `skin.glb`        | FLAME メッシュ (subdivide 後)      | ✅ 必須 |
| `offset.ply`      | Gaussian Splatting 点群 (色・形状) | ✅ 必須 |
| `animation.glb`   | モーションデータ (固定テンプレート) | ✗ 補正不要 |

**メッシュだけを補正して PLY を放置すると、Gaussian が元位置に残り「鼻と
口元に二重表示 (ゴースト)」が発生する。** 必ず両方を同じ補正値で更新する。
両者は subdivide 後の 20,018 頂点で 1:1 対応している。

## 入力前提

| 項目 | 値 |
|------|----|
| FLAME 元頂点数 | 5,023 |
| subdivide 後頂点数 | 20,018 |
| 領域マスクファイル | `/vol/pretrained_models/human_model_files/flame_assets/flame/FLAME_masks.pkl` |
| マスクの領域名 | `face`, `neck`, `scalp`, `boundary`, `right_eyeball`, `left_eyeball`, `right_ear`, `left_ear`, `forehead`, `eye_region`, `nose`, `lips`, `right_eye_region`, `left_eye_region` |
| マスクの座標系 | 元の 5,023 頂点に対するインデックス |

## 手順

### Step 1: メッシュと PLY を生成

```python
saved_head = self.lam.renderer.flame_model.save_shaped_mesh(
    shape_param.unsqueeze(0).cuda(), fd=oac_dir,
)
_ply_path = os.path.join(oac_dir, "offset.ply")
res["cano_gs_lst"][0].save_ply(_ply_path, rgb2sh=False, offset2xyz=True)
```

`offset2xyz=True` により、`save_ply` が `self.offset` を xyz 座標として
書き出す。これにより `_ply['vertex']['x'/'y'/'z']` を直接書き換えるだけで
Gaussian 位置を補正できる。

### Step 2: FLAME_masks を読む（必ず `pickle.load`）

```python
import pickle
_masks_path = "/vol/pretrained_models/human_model_files/flame_assets/flame/FLAME_masks.pkl"
with open(_masks_path, "rb") as _mf:
    _part_masks = pickle.load(_mf, encoding="latin1")
```

❌ `np.load(allow_pickle=True)` は使わない。返り値が
   `0-d object ndarray` になるケースがあり、`.item()` を介さないと
   dict として参照できない。`.pkl` 拡張子なら `pickle.load` で意図を
   明確化する。

### Step 3: 元 5,023 インデックスから補正対象領域を作る

例（頬）: `face` から `nose` / `lips` / `eye_region` / `forehead` を除外。

```python
_n_orig = 5023
_face_idx = np.asarray(_part_masks["face"])
_face_idx = _face_idx[_face_idx < _n_orig]
_exclude = set()
for _region in ["nose", "lips", "eye_region", "forehead"]:
    _r = np.asarray(_part_masks[_region])
    _exclude.update(_r[_r < _n_orig].tolist())
_cheek_idx_orig = np.array(
    [i for i in _face_idx if i not in _exclude], dtype=np.int64
)
```

この時点で得られるのは **元 5,023 頂点中の頬** のみ（実測 430 頂点）。
subdivide 後の 5,024 〜 20,017 頂点はカバーされていない。

### Step 4: 空間バウンディングボックスで全 20,018 頂点に拡張

❌ インデックスをそのまま使うと 430 頂点しか動かず、視覚的に変化しない。
✅ 元頂点の bbox を基準に、subdivide 後を含む全頂点から該当領域を選ぶ。

```python
_mesh = trimesh.load_mesh(saved_head, process=False)  # ← Scene を返さない
_verts = _mesh.vertices.copy()

_cheek_ref = _verts[_cheek_idx_orig]
_x_min, _x_max = _cheek_ref[:, 0].min(), _cheek_ref[:, 0].max()
_y_min, _y_max = _cheek_ref[:, 1].min(), _cheek_ref[:, 1].max()
_z_min, _z_max = _cheek_ref[:, 2].min(), _cheek_ref[:, 2].max()
_margin = 0.002
_all_cheek = np.where(
    (_verts[:, 0] >= _x_min - _margin) & (_verts[:, 0] <= _x_max + _margin) &
    (_verts[:, 1] >= _y_min - _margin) & (_verts[:, 1] <= _y_max + _margin) &
    (_verts[:, 2] >= _z_min - _margin) & (_verts[:, 2] <= _z_max + _margin)
)[0]
```

❌ `trimesh.load(...)` は入力により Scene を返すことがあり、`.vertices`
   で AttributeError になる。**`load_mesh(..., process=False)` を使う。**

### Step 5: 中心からの相対座標で等比スケール補正

特定軸の中心からの相対距離をスケールする方式（中心は不動、外形だけ縮小／拡大）。

```python
_cheek_center_y = _verts[_all_cheek, 1].mean()
_verts[_all_cheek, 1] = _cheek_center_y + (
    _verts[_all_cheek, 1] - _cheek_center_y
) * 0.88
_mesh.vertices = _verts
_mesh.export(saved_head)
```

❌ 部分シフト（`_verts[idx, 1] += 0.002` のような単純加算）は
   **領域境界に段差を作り「イボ」状の突起を生む**。等比スケール方式に
   一本化すること。

### Step 6: 同じ bbox・同じスケールで PLY も補正

```python
_ply = PlyData.read(_ply_path, mmap=False)  # ← デバッグ容易化
_gy = np.array(_ply['vertex']['y'], copy=True)  # ← copy 必須 (alias回避)
_gx = np.array(_ply['vertex']['x'], copy=True)
_gz = np.array(_ply['vertex']['z'], copy=True)
_all_cheek_g = np.where(
    (_gx >= _x_min - _margin) & (_gx <= _x_max + _margin) &
    (_gy >= _y_min - _margin) & (_gy <= _y_max + _margin) &
    (_gz >= _z_min - _margin) & (_gz <= _z_max + _margin)
)[0]
_cheek_center_y_g = _gy[_all_cheek_g].mean()
_gy[_all_cheek_g] = _cheek_center_y_g + (
    _gy[_all_cheek_g] - _cheek_center_y_g
) * 0.88
_ply['vertex'].data['y'] = _gy
_ply.write(_ply_path)
```

PLY と メッシュは同じ座標系・同じ頂点数 (20,018) なので、**まったく同じ
bbox と同じスケール係数を適用**する。

### Step 7: 失敗を可視化する

```python
_log = lambda msg: open("/vol_out/oac_debug.txt", "a").write(msg + "\n")
_log("[OAC] start cheek correction")
...
_log(f"[OAC] orig cheek={len(_cheek_idx_orig)}, "
     f"spatial cheek={len(_all_cheek)}/{len(_verts)}")
```

ログを **Modal Volume のファイルに書き出す**。stdout は raster debug の
大量出力に埋もれるため使わない。エラー時は `oac_error.txt` も同様に
ファイル出力し、必ず `output_vol.commit()` で外部から見えるようにする。

## 現行補正パラメータ（cc81e92 ベース、2026-04-21 時点）

| 部位 | 軸 | 操作 | 係数 | 効果 |
|------|----|------|------|------|
| 頬   | Y  | 中心からの相対スケール | × 0.88 | 縦長を 12% 縮める |
| 鼻幅 | X  | 中心からの相対スケール | × 1.08 | 小鼻を 8% 広げる |
| 鼻高 | Z  | 中心からの相対スケール | × 0.85 | 突出を 15% 抑える |

## ユーザーフィードバックで未反映の追加補正候補

| 部位 | 軸 | 操作 | 係数 | 出典 |
|------|----|------|------|------|
| 鼻高 | Z  | スケール強化 | 0.85 → **0.52** 検討 | 「効き方が穏やか過ぎ。更に20%抑えて」 |
| 顎   | Y  | スケール | × 0.85 | 「唇と同じ高さ。15%抑えたい」 |
| 目   | X  | スケール | × 0.90 | 「目尻 10% 短く」 |

これらは過去 `20c3648` で一括追加され ZIP 生成成功実績あり。再適用時は
**1 コミット 1 部位**で慎重に進めること。

## やってはいけないこと（実証済みの失敗事例）

1. **鼻先 Y+0.002 のような部分シフト** → 鼻に「イボ」発生
2. **インデックスベース選択のまま subdivide 後を放置** → 430 頂点しか動かず無効果
3. **`np.load(allow_pickle=True)` で `.pkl` を読む** → dict として失敗するケース
4. **`trimesh.load(...)`** → Scene 返却で `.vertices` 失敗
5. **stdout のみでデバッグ** → raster debug に埋もれて見えない
6. **メッシュだけ補正・PLY 放置** → Gaussian がズレてゴースト二重表示
7. **コード削除時にログ参照を残す** → NameError で OAC 全体クラッシュ

## デプロイ手順（PowerShell, ユーザー側ワークフロー）

```powershell
cd C:\Users\hamad\LAM_mirai_modal
$BASE = "https://raw.githubusercontent.com/mirai-gpro/LAM_mirai/<COMMIT_HASH>/modal_app.py"
Invoke-WebRequest -Uri $BASE -OutFile modal_app.py
modal run modal_app.py
modal volume ls lam-output /
modal volume get lam-output chatting_avatar_<TIMESTAMP>.zip .\output\
modal volume get lam-output oac_debug.txt .\output\
type .\output\oac_debug.txt
```

⚠ **GitHub CDN がブランチ名を 5〜10 分キャッシュするため、最新版を確実に
取得するには `<COMMIT_HASH>` を直接指定する。**

## 参考: 根本解決の選択肢

| 方針 | 内容 | コスト | 現実性 |
|------|------|--------|--------|
| 頂点補正継続 (本メソッド) | OAC 出力後に補正 | 低 | ✅ 採用中 |
| FaceVerse の shape basis 借用 | FLAME トポロジ維持で PCA だけ差し替え | 中 | 検討余地 |
| FaceVerse で完全置換 | LAM 再学習が必要 | 極大 | 非現実的 |

FaceVerse (CVPR 2022) は東アジア人 60,000 RGB-D + 2,000 3D スキャンで
構築されており、日本人寄りの bbox を提供できる可能性がある。ただし
LAM/FLAME のメッシュトポロジ互換性に課題があるため、当面は本頂点補正
メソッドを継続する。
