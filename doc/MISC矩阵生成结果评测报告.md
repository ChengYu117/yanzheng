# MISC latent-label 鐭╅樀鐢熸垚缁撴灉璇勬祴鎶ュ憡

> 鐢熸垚鏃堕棿锛?026-04-27
> 杈撳嚭鐩綍锛歚outputs/structural_smoke_misc/functional/misc_label_mapping`
> 璇勬祴鎬ц川锛氬伐绋嬮摼璺?smoke 楠岃瘉锛屼笉浣滀负姝ｅ紡绉戠爺缁撹銆?
## 1. 鏈杩愯鐩殑

鏈杩愯鐢ㄤ簬楠岃瘉閲嶆瀯鍚庣殑 MISC 鍏ㄩ噺鏁版嵁鍙ｅ緞鑳藉惁瀹屾垚鈥渓atent 脳 MISC label鈥濈煩闃电敓鎴愶細

- 浣跨敤 MISC `unit_text` 浣滀负鏍锋湰鍗曚綅銆?- 澶嶇敤 SAE 鎺ㄧ悊闃舵淇濆瓨鐨?`utterance_features.pt`銆?- 鐢熸垚姣忎釜 latent 瀵规瘡涓?MISC 鏍囩鐨勫叧鑱旂粺璁°€?- 杈撳嚭鏍囩纰庣墖鍖栥€乴atent 閲嶅彔銆乀opK Jaccard 鍜岀ず渚嬪崱鐗囥€?
涓轰簡閬垮厤閲嶆柊鍔犺浇 8B 妯″瀷锛屾湰娆″鐢ㄤ簡鍓嶄竴杞粨鏋勬寚鏍?smoke 鐨?32 鏉℃牱鏈壒寰併€?
## 2. 杩愯鍛戒护

```powershell
conda run -n qwen-env-py311 python run_misc_label_mapping.py `
  --data-dir data/mi_quality_counseling_misc `
  --features-path outputs/structural_smoke_misc/feature_store/utterance_features.pt `
  --output-dir outputs/structural_smoke_misc/functional/misc_label_mapping `
  --limit-records 32 `
  --min-positive 2 `
  --min-negative 2 `
  --precision-k 3 5 10 `
  --chunk-size 512 `
  --top-k-per-label 20 `
  --top-example-latents 3 `
  --top-examples-per-latent 5
```

璇存槑锛?
- `--limit-records 32` 蹇呴』鍜屽墠涓€杞?feature store 鐨勬牱鏈暟涓€鑷淬€?- `--min-positive 2` 鏄负浜嗚灏忔牱鏈?smoke 鑳戒骇鍑虹煩闃碉紱姝ｅ紡瀹為獙搴旀仮澶嶅埌鏇寸ǔ鍋ラ槇鍊硷紝濡?`10` 鎴栨洿楂樸€?- 鏈娌℃湁閲嶆柊鎺ㄧ悊妯″瀷锛屽彧楠岃瘉鐭╅樀璁＄畻鍜岀粨鏋滆惤鐩樸€?
## 3. 杈撳嚭鏂囦欢

涓昏杈撳嚭濡備笅锛?
- `latent_label_matrix.csv`
- `label_summary.json`
- `label_indicator_matrix.csv`
- `label_fragmentation.json`
- `latent_overlap.json`
- `label_topk_jaccard.json`
- `behavior_asymmetry.md`
- `top_latents_by_label/`
- `top_examples_by_label/`
- `annotation_records.jsonl`
- `run_summary.json`

鐭╅樀瑙勬ā锛?
```text
n_records = 32
feature_shape = [32, 32768]
labels = RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER
latent_label_matrix shape = 294912 rows 脳 21 columns
```

鐭╅樀琛屾暟涓?`32768 脳 9`锛屽洜涓?`SU` 鍦?32 鏉℃牱鏈腑鍙湁 1 涓渚嬶紝浣庝簬鏈璁剧疆鐨?`min_positive=2`锛屽洜姝よ璺宠繃銆?
## 4. 鏍囩鍒嗗竷

| Label | Positive | Negative | Prevalence |
|---|---:|---:|---:|
| RE | 8 | 24 | 0.250 |
| RES | 5 | 27 | 0.156 |
| REC | 3 | 29 | 0.094 |
| QU | 10 | 22 | 0.312 |
| QUO | 6 | 26 | 0.188 |
| QUC | 4 | 28 | 0.125 |
| GI | 3 | 29 | 0.094 |
| SU | 1 | 31 | 0.031 |
| AF | 3 | 29 | 0.094 |
| OTHER | 7 | 25 | 0.219 |

璇勬祴锛?
- 鏍囩璇诲彇鍜屽眰绾ф爣绛惧睍寮€姝ｅ父銆?- `RE/RES/REC`銆乣QU/QUO/QUC` 鐨勭埗瀛愭爣绛惧悓鏃惰繘鍏ョ煩闃碉紝绗﹀悎瀵煎笀瑕佹眰鐨勨€滄爣绛剧┖闂翠笉鏄竴缁翠簩鍒嗙被鈥濈殑鍒嗘瀽鐩爣銆?- 灏忔牱鏈笅 `SU` 涓嶅彲鍒嗘瀽锛屾寮忓叏閲忚繍琛屾墠閫傚悎璇勪及鏀寔绫昏涓恒€?
## 5. 鏄捐憲鎬т笌纰庣墖鍖栫粨鏋?
鏈 FDR 鍚庡彧鏈?`AF` 鍑虹幇鏄捐憲 latent锛?
| Label | Significant Latents | Positive Effect | Negative Effect | 璇存槑 |
|---|---:|---:|---:|---|
| AF | 6 | 4 | 2 | 灏忔牱鏈笅鍞竴鍑虹幇 FDR 鏄捐憲鐨勬爣绛?|
| RE | 0 | 0 | 0 | 鏈夎緝寮哄€欓€変絾鏈繃 FDR |
| RES | 0 | 0 | 0 | 鏈夊€欓€変絾姝ｄ緥灏?|
| REC | 0 | 0 | 0 | 鏈夊€欓€変絾姝ｄ緥灏?|
| QU | 0 | 0 | 0 | AUC 杈冮珮浣嗘湭杩?FDR |
| QUO | 0 | 0 | 0 | 鏈繃 FDR |
| QUC | 0 | 0 | 0 | 鍊欓€?AUC 楂樹絾鏍锋湰澶皯 |
| GI | 0 | 0 | 0 | 鏈繃 FDR |
| OTHER | 0 | 0 | 0 | 鏈繃 FDR |

`AF` 鐨?FDR 鏄捐憲 latent锛?
| Latent | Cohen's d | AUC | Directional AUC | p-value |
|---:|---:|---:|---:|---:|
| 31133 | 3.896 | 1.000 | 1.000 | 5.03e-08 |
| 31867 | 2.121 | 0.925 | 0.925 | 1.07e-11 |
| 20743 | 1.452 | 0.885 | 0.885 | 3.22e-07 |
| 21800 | 1.334 | 0.874 | 0.874 | 1.21e-06 |
| 19663 | -1.232 | 0.172 | 0.828 | 6.15e-07 |
| 9993 | -1.176 | 0.190 | 0.810 | 1.34e-06 |

璇勬祴锛?
- 宸ョ▼涓婏紝纰庣墖鍖栫粺璁″凡缁忓彲鐢細姣忎釜鏍囩閮借兘浜у嚭鍊欓€?latent銆佹樉钁?latent 鏁伴噺銆佹璐熸晥搴旀暟閲忓拰 top examples銆?- 绉戠爺涓婏紝鏈 `AF` 鏄捐憲鎬т笉鑳界洿鎺ラ噰淇°€傚師鍥犳槸 `AF` 鍙湁 3 涓渚嬶紝涓旇繍琛岃繃绋嬩腑鍑虹幇浜?scipy precision loss 璀﹀憡锛岃鏄庨儴鍒?latent 鍦ㄥ皬鏍锋湰涓繎浼煎父鏁版垨缁勫唴鏂瑰樊鏋佷綆銆?- 姝ｅ紡缁撹蹇呴』渚濊禆鍏ㄩ噺 6194 鏉℃牱鏈殑鐭╅樀銆?
## 6. Top 鍊欓€夎瀵?
閮ㄥ垎鏍囩鐨?top 鍊欓€夊涓嬶細

| Label | Top latent | Cohen's d | AUC | Directional AUC | FDR |
|---|---:|---:|---:|---:|---|
| RE | 19435 | 1.772 | 0.786 | 0.786 | False |
| RES | 15747 | 2.193 | 0.785 | 0.785 | False |
| REC | 11656 | 4.047 | 0.833 | 0.833 | False |
| QU | 13430 | 2.817 | 0.914 | 0.914 | False |
| QUO | 9959 | 2.520 | 0.833 | 0.833 | False |
| QUC | 24695 | 11.561 | 1.000 | 1.000 | False |
| GI | 31220 | 3.772 | 0.966 | 0.966 | False |
| AF | 28087 | 43.231 | 1.000 | 1.000 | False |
| OTHER | 7117 | 1.779 | 0.714 | 0.714 | False |

璇勬祴锛?
- 杩欎簺 top 鍊欓€夎鏄庣煩闃靛凡缁忚兘鍖哄垎涓嶅悓 MISC 琛屼负鏍囩锛屽苟鑳界粰鍑烘爣绛惧搴旂殑 latent 鎺掑悕銆?- 浣嗘瀬楂樼殑 Cohen's d锛屼緥濡?`AF` 鐨?43.231銆乣QUC` 鐨?11.561锛屾洿鍍忓皬鏍锋湰鏂瑰樊涓嶇ǔ瀹氶€犳垚鐨?smoke artifact銆?- 姝ｅ紡鎶ュ憡涓簲浼樺厛鐪嬪叏閲忎笅鐨?`directional_auc`銆丗DR 鏄捐憲鏁伴噺銆乼op examples 鏄惁璇箟涓€鑷达紝鑰屼笉鏄崟鐙湅灏忔牱鏈?Cohen's d銆?
## 7. 鏍囩閲嶅彔涓?Jaccard

鏄捐憲 latent 閲嶅彔锛?
```text
significant latent-label edges = 6
latents with any significant label = 6
single-label latents = 6
multi-label latents = 0
```

TopK Jaccard 鏈€楂樼殑鏍囩瀵癸細

| Label A | Label B | Intersection | Union | Jaccard |
|---|---|---:|---:|---:|
| RE | RES | 8 | 32 | 0.250 |
| QU | QUC | 7 | 33 | 0.212 |
| QU | QUO | 7 | 33 | 0.212 |
| RE | REC | 2 | 38 | 0.053 |

璇勬祴锛?
- `RE` 涓?`RES` 鐨?overlap 鏈€楂橈紝绗﹀悎鐖舵爣绛句笌瀛愭爣绛惧叡浜?latent 鐨勯鏈熴€?- `QU` 涓?`QUO/QUC` 鐨?overlap 涔熺鍚?MISC 灞傜骇鏍囩鍏崇郴銆?- 灏忔牱鏈笅娌℃湁鏄捐憲 multi-label latent锛屼笉鑳借鏄庢寮忔暟鎹腑娌℃湁鍏变韩琛ㄥ緛锛屽彧鑳借鏄庡綋鍓?32 鏉?smoke 鏍锋湰涓嶈冻浠ョǔ瀹氫及璁°€?
## 8. 鎬讳綋缁撹

鏈鐭╅樀鐢熸垚閾捐矾閫氳繃锛?
- 鏁版嵁璇诲彇閫氳繃銆?- feature store 澶嶇敤閫氳繃銆?- `latent_label_matrix.csv` 鎴愬姛鐢熸垚銆?- 鏍囩鍒嗗竷銆佺鐗囧寲銆侀噸鍙犮€丣accard銆乼op examples 鍧囨垚鍔熻惤鐩樸€?- 缁撴灉缁撴瀯绗﹀悎瀵煎笀瑕佹眰鐨勨€滄爣绛剧┖闂翠笌 SAE 琛ㄥ緛绌洪棿涔嬮棿鐨勫瀵瑰鏄犲皠鈥濆垎鏋愭柟鍚戙€?
鏈缁撴灉鐨勭鐮斿彲淇″害鏈夐檺锛?
- 鏍锋湰鏁板彧鏈?32銆?- 澶氭暟鏍囩姝ｄ緥鏁拌繃灏戙€?- `SU` 琚烦杩囥€?- FDR 鏄捐憲鎬у彧鍑虹幇鍦?`AF`锛屼笖鍙兘鏄皬鏍锋湰鏂瑰樊瀵艰嚧銆?
鍥犳锛屾湰娆＄粨鏋滃簲浣滀负宸ョ▼楠屾敹缁撴灉锛屼笉浣滀负璁烘枃鎴栨€绘姤鍛婁腑鐨勬寮忕粨璁恒€?
## 9. 涓嬩竴姝ュ缓璁?
姝ｅ紡瀹為獙寤鸿鐩存帴璧版柊鐗堜富娴佺▼锛?
```powershell
python run_sae_evaluation.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc `
  --data-format auto `
  --label-mode misc_multilabel `
  --batch-size 4 `
  --max-seq-len 128 `
  --full-structural `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_full_sae_eval
```

姝ｅ紡璇勬祴閲嶇偣鐪嬶細

- `functional/misc_label_mapping/latent_label_matrix.csv`
- `functional/misc_label_mapping/label_fragmentation.json`
- `functional/misc_label_mapping/latent_overlap.json`
- `functional/misc_label_mapping/label_topk_jaccard.json`
- `functional/misc_label_mapping/top_examples_by_label/`

姝ｅ紡鎶ュ憡涓缓璁噰鐢細

- 姣忎釜 MISC 鏍囩鐨勬樉钁?latent 鏁伴噺琛ㄧず鈥滄爣绛剧鐗囧寲绋嬪害鈥濄€?- 姣忎釜 latent 鏄捐憲鍏宠仈鏍囩鏁拌〃绀衡€滆〃寰佸叡浜▼搴︹€濄€?- 鐖跺瓙鏍囩 TopK Jaccard 琛ㄧず鈥滃眰绾ф爣绛惧湪 SAE 绌洪棿涓殑閲嶅彔鈥濄€?- top examples 浣滀负璇箟瑙ｉ噴璇佹嵁锛岃€屼笉鏄彧渚濊禆缁熻閲忋€?