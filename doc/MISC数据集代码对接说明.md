# MISC 鏁版嵁闆嗗湪浠ｇ爜涓殑瀵规帴璇存槑

> 鐩殑锛氳鏄庡綋鍓?`data/mi_quality_counseling_misc` 鏁版嵁闆嗗湪浠ｇ爜閲屽浣曡鍙栥€佸浣曞垏鍒嗕负鏍锋湰銆佸浣曞垝鍒嗘爣绛撅紝浠ュ強濡備綍杈撳叆鍒?Llama + SAE 娴佺▼涓€?
## 1. 鏁版嵁婧愪綅缃?
褰撳墠姝ｅ紡涓诲疄楠岄粯璁ゆ暟鎹洰褰曟槸锛?
```text
data/mi_quality_counseling_misc
```

涓绘暟鎹潵鑷細

```text
data/mi_quality_counseling_misc/misc_annotations/high/*.jsonl
data/mi_quality_counseling_misc/misc_annotations/low/*.jsonl
```

浼氳瘽璐ㄩ噺鏍囩鏉ヨ嚜锛?
```text
data/mi_quality_counseling_misc/metadata/labels.csv
```

`labels.csv` 浣跨敤 `file_id -> high/low` 鐨勬槧灏勶紝渚嬪锛?
```csv
id,label
high_001,high
high_002,high
```

浠ｇ爜鍏ュ彛锛?
- `src/nlp_re_base/data.py`
- `load_experiment_dataset(...)`
- `load_misc_full_records(...)`
- `misc_label_set(...)`

## 2. 鍘熷杈撳叆鏍蜂緥

姣忎釜 `misc_annotations/*/*.jsonl` 鏂囦欢涓紝涓€琛屽氨鏄竴涓?MISC 琛屼负鍗曞厓銆傚吀鍨嬪師濮嬭褰曞涓嬶細

```json
{
  "file_id": "high_001",
  "unit_text": "so it sounds like you really want to go to Black Friday with your friends you're feeling a little anxious about it",
  "predicted_code": "RE",
  "predicted_subcode": "REC",
  "rationale": "Reflects both desire and anxiety, adding emotional nuance and linking them, which is more than a simple restatement.",
  "confidence": 0.93
}
```

鍙︿竴涓棶棰樼被琛屼负鏍蜂緥锛?
```json
{
  "file_id": "high_001",
  "unit_text": "hey Monica how are you doing today",
  "predicted_code": "QU",
  "predicted_subcode": "QUO",
  "rationale": "Greeter plus an open-ended inquiry about how the client is doing; overall functions as an open question.",
  "confidence": 0.86
}
```

## 3. 鏍锋湰鍗曚綅濡備綍鍒囧垎

褰撳墠浠ｇ爜涓嶅啀浠庡畬鏁磋浆褰曚腑閲嶆柊鍒囧彞锛屼篃涓嶆妸涓婁笅鏂囩獥鍙ｆ嫾杩涘幓銆?
姝ｅ紡鏍锋湰鍗曚綅鏄細

```text
涓€琛?JSONL = 涓€涓?unit_text = 涓€涓?SAE 鎺ㄧ悊鏍锋湰
```

涔熷氨鏄锛?
- 涓嶄娇鐢ㄥ畬鏁?session 浣滀负涓€涓牱鏈€?- 涓嶄娇鐢?client + counselor 瀵硅瘽绐楀彛浣滀负涓€涓牱鏈€?- 涓嶅湪涓绘祦绋嬩腑鍐嶆鎸夋爣鐐瑰垏鍙ャ€?- `unit_text` 宸茬粡琚涓?MISC 琛屼负鍗曞厓銆?
鍔犺浇鍚庢瘡鏉℃牱鏈細鐢熸垚绋冲畾 ID锛?
```text
sample_id = "{file_id}:{line_no:04d}"
```

渚嬪锛?
```text
high_001:0001
high_001:0002
```

杩欎繚璇?`records.jsonl`銆乣label_matrix.csv`銆乣utterance_features.pt` 鐨勮鍙峰彲浠ヤ竴涓€瀵归綈銆?
## 4. 璁板綍鏍囧噯鍖栧悗鐨勫瓧娈?
`load_misc_full_records(...)` 浼氭妸鍘熷 JSONL 缁熶竴鎴愬涓嬬粨鏋勶細

```json
{
  "sample_id": "high_001:0002",
  "record_id": "high_001:0002",
  "file_id": "high_001",
  "quality_label": "high",
  "text": "so it sounds like you really want to go to Black Friday with your friends you're feeling a little anxious about it",
  "unit_text": "so it sounds like you really want to go to Black Friday with your friends you're feeling a little anxious about it",
  "predicted_code": "RE",
  "predicted_subcode": "REC",
  "label_re": 1,
  "label_family": "RE",
  "labels": ["RE", "REC"],
  "confidence": 0.93,
  "rationale": "...",
  "source_split": "high",
  "source_file": "high_001.jsonl",
  "source_line": 2
}
```

鍏朵腑鏈€閲嶈鐨勬槸锛?
| 瀛楁 | 鍚箟 |
|---|---|
| `text` / `unit_text` | 鐪熸杈撳叆 Llama tokenizer 鐨勬枃鏈?|
| `labels` | 澶氭爣绛?MISC 鏍囩闆嗗悎 |
| `label_re` | RE vs NonRE 浜屽垎绫绘爣绛?|
| `quality_label` | 浼氳瘽璐ㄩ噺鏍囩锛屾潵鑷?`metadata/labels.csv` |
| `predicted_code` / `predicted_subcode` | 淇濈暀鍘熷 MISC code/subcode |
| `rationale` | 鏍囨敞瑙ｉ噴锛屼笉杈撳叆妯″瀷锛屽彧鐢ㄤ簬瀹℃煡 |

## 5. 鏍囩濡備綍鍒掑垎

鏍稿績鏍囩闆嗗悎鍦ㄤ唬鐮佷腑瀹氫箟涓猴細

```python
CORE_MISC_LABELS = {"RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"}
```

杈撳嚭鏍囩椤哄簭浼樺厛涓猴細

```text
RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER
```

### 5.1 澶氭爣绛捐鍒?
`misc_label_set(record)` 鐨勮鍒欐槸锛?
1. 濡傛灉 `predicted_code` 灞炰簬鏍稿績鏍囩锛屽垯鍔犲叆鏍囩闆嗗悎銆?2. 濡傛灉 `predicted_subcode` 灞炰簬鏍稿績鏍囩锛屽垯涔熷姞鍏ユ爣绛鹃泦鍚堛€?3. 濡傛灉 code/subcode 閮戒笉灞炰簬鏍稿績闆嗗悎锛屽垯褰掑叆 `OTHER`銆?4. 鍘熷 `predicted_code` 鍜?`predicted_subcode` 浠嶄繚鐣欙紝涓嶄細涓㈠純銆?
渚嬪瓙锛?
| 鍘熷 code | 鍘熷 subcode | 鏍囧噯 labels |
|---|---|---|
| `RE` | `RES` | `["RE", "RES"]` |
| `RE` | `REC` | `["RE", "REC"]` |
| `QU` | `QUO` | `["QU", "QUO"]` |
| `QU` | `QUC` | `["QU", "QUC"]` |
| `GI` | null | `["GI"]` |
| `SU` | null | `["SU"]` |
| `AF` | null | `["AF"]` |
| `ST` | null | `["OTHER"]` |
| `FA` | null | `["OTHER"]` |

### 5.2 RE vs NonRE 浜屽垎绫昏鍒?
`label_re=True` 鐨勮鍒欐槸锛?
```python
predicted_code == "RE" or predicted_subcode in {"RES", "REC"}
```

鍥犳锛?
```text
RE / RES / REC -> label_re = 1
鍏朵粬鏍囩 -> label_re = 0
```

杩欎釜浜屽垎绫诲彛寰勭敤浜庯細

- `functional/re_binary/candidate_latents.csv`
- RE/NonRE probe
- 鍚庣画 `causal/run_experiment.py` 鐨勯粯璁ゅ洜鏋滈獙璇佸叆鍙?
### 5.3 label matrix

涓绘祦绋嬩細鐢熸垚锛?
```text
outputs/misc_full_sae_eval/label_matrix.csv
```

璇ユ枃浠舵瘡琛屽搴斾竴鏉℃牱鏈紝姣忓垪瀵瑰簲涓€涓?MISC 鏍囩銆備緥濡傦細

| sample_id | RE | RES | REC | QU | QUO | QUC | GI | SU | AF | OTHER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `high_001:0001` | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `high_001:0002` | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 6. 褰撳墠鍏ㄩ噺鏁版嵁缁熻

姝ｅ紡鍏ㄩ噺 MISC 杩愯缁撴灉鏄剧ず锛?
| 椤圭洰 | 鏁板€?|
|---|---:|
| 鏍锋湰鏁?| 6194 |
| RE 鏍锋湰 | 1358 |
| NonRE 鏍锋湰 | 4836 |
| high quality 鏍锋湰 | 4169 |
| low quality 鏍锋湰 | 2025 |

鏍囩璁℃暟锛?
| 鏍囩 | 鏁伴噺 |
|---|---:|
| RE | 1358 |
| RES | 516 |
| REC | 842 |
| QU | 1974 |
| QUO | 1206 |
| QUC | 768 |
| GI | 681 |
| SU | 222 |
| AF | 349 |
| OTHER | 1610 |

鏂囨湰闀垮害鐗瑰緛锛?
| 鎸囨爣 | 鏁板€?|
|---|---:|
| 骞冲潎璇嶆暟 | 13.1 |
| 涓綅璇嶆暟 | 10 |
| 90 鍒嗕綅璇嶆暟 | 27.7 |
| 涓嶈秴杩?5 璇嶆牱鏈崰姣?| 23.5% |
| 涓嶈秴杩?10 璇嶆牱鏈崰姣?| 50.1% |

杩欎篃鏄?MISC 鍦?SAE 閲嶆瀯鎸囨爣涓婁綆浜?SlimPajama 瀹樻柟鍒嗗竷鐨勯噸瑕佸師鍥犱箣涓€锛氭牱鏈洿鐭€佹洿纰庯紝骞朵笖鍖呭惈澶ч噺 `okay`銆乣mm-hmm`銆乣yeah` 杩欑被鍜ㄨ浜掑姩鐭銆?
## 7. 濡備綍杈撳叆鍒?Llama 鍜?SAE

涓诲叆鍙ｏ細

```text
run_sae_evaluation.py
```

鏍稿績璋冪敤閾撅細

```text
run_sae_evaluation.py
  -> load_experiment_dataset(...)
  -> extract_and_process_streaming(...)
  -> sae.forward_with_details(...)
  -> run_structural_evaluation(...)
  -> run_functional_evaluation(...)
  -> run_misc_label_mapping(...)
```

### 7.1 鏂囨湰杈撳叆

杈撳叆缁?tokenizer 鐨勬枃鏈垪琛ㄦ槸锛?
```python
dataset.texts = [record["text"] for record in dataset.records]
```

涔熷氨鏄瘡鏉?`unit_text`銆?
tokenizer 鍙傛暟锛?
```python
padding=True
truncation=True
max_length=128
return_tensors="pt"
```

褰撳墠娌℃湁棰濆 prompt 妯℃澘锛屼篃娌℃湁鎻掑叆鏍囩銆乺ationale 鎴栦笂涓嬫枃銆?
### 7.2 hook 鐐?
褰撳墠 SAE 閰嶇疆锛?
```json
{
  "hook_point": "blocks.19.hook_resid_post",
  "d_model": 4096,
  "d_sae": 32768,
  "aggregation": "max"
}
```

鍦?HuggingFace Llama 涓紝璇?hook 琚槧灏勫埌锛?
```text
model.model.layers[19] 鐨勮緭鍑?hidden_states
```

姣忎釜 batch 鐨?hidden activation 褰㈢姸鏄細

```text
[B, T, 4096]
```

鍏朵腑锛?
- `B` 鏄?batch size
- `T <= 128`
- `4096` 鏄?Llama residual hidden size

### 7.3 SAE 缂栫爜

`extract_and_process_streaming(...)` 浼氭妸 `[B, T, 4096]` 杈撳叆 SAE锛?
```python
sae_outputs = sae.forward_with_details(sae_input)
```

寰楀埌锛?
| 杈撳嚭 | 褰㈢姸 | 鐢ㄩ€?|
|---|---|---|
| `latents` | `[B, T, 32768]` | token-level SAE latent |
| `reconstructed_raw` | `[B, T, 4096]` | raw activation 閲嶆瀯 |
| `reconstructed_normalized` | `[B, T, 4096]` | normalized 绌洪棿閲嶆瀯 |
| `input_normalized` | `[B, T, 4096]` | SAE 褰掍竴鍖栧悗鐨勮緭鍏?|

娉ㄦ剰锛氬叏閲?token-level latent 榛樿涓嶄繚瀛橈紝鍥犱负 `[6194, 128, 32768]` 鏂囦欢浼氶潪甯稿ぇ銆?
## 8. token 鍒?utterance 鐨勮仛鍚堟柟寮?
褰撳墠榛樿鑱氬悎鏂瑰紡鏄細

```text
aggregation = "max"
```

鍗冲姣忔潯 `unit_text` 鐨勬墍鏈夐潪 padding token锛屽湪 latent 缁村害涓婂彇鏈€澶ф縺娲伙細

```text
[B, T, 32768] -> [B, 32768]
```

padding token 閫氳繃 `attention_mask` 鎺掗櫎銆?
榛樿淇濆瓨锛?
```text
outputs/misc_full_sae_eval/feature_store/utterance_features.pt
```

鍏跺唴瀹规槸锛?
```python
{
  "utterance_features": Tensor[6194, 32768],
  "aggregation": "max",
  "hook_point": "blocks.19.hook_resid_post",
  "data_format": "misc_full"
}
```

鍚屾椂淇濆瓨妯″瀷鍙ュ悜閲忚仛鍚堬細

```text
outputs/misc_full_sae_eval/feature_store/utterance_activations.pt
```

鍏跺舰鐘朵负锛?
```text
[6194, 4096]
```

## 9. 缁撴瀯鎸囨爣濡備綍浣跨敤鏍锋湰

缁撴瀯鎸囨爣涓嶆槸鍙湅 utterance-level features銆?
褰撲娇鐢細

```text
--full-structural
```

鏃讹紝浠ｇ爜浼氬湪鎺ㄧ悊娴佷腑鎶婃瘡涓?batch 鐨?token-level activation銆乺econstruction銆乴atents 鍜?attention mask 杈撳叆 online accumulator銆?
涔熷氨鏄缁撴瀯鎸囨爣浣跨敤鐨勬槸锛?
```text
鎵€鏈夋牱鏈殑闈?padding token
```

褰撳墠鍏ㄩ噺 MISC 缁撴瀯鎸囨爣瑕嗙洊锛?
```text
n_tokens = 93636
```

杩欓儴鍒嗙敤浜庯細

- `mse`
- `cosine_similarity`
- centered EV
- OpenMOSS legacy EV
- L0
- dead feature ratio
- by-label structural metrics

## 10. 鍔熻兘鎸囨爣濡備綍浣跨敤鏍锋湰

鍔熻兘鎸囨爣鍒嗕袱灞傘€?
### 10.1 RE vs NonRE

浜屽垎绫诲姛鑳芥寚鏍囦娇鐢細

```python
label_re == 1 -> RE
label_re == 0 -> NonRE
```

杈撳叆鏄細

```text
utterance_features: [6194, 32768]
```

杈撳嚭鐩綍锛?
```text
outputs/misc_full_sae_eval/functional/re_binary
```

鍏抽敭鏂囦欢锛?
```text
candidate_latents.csv
metrics_functional.json
judge_bundle/
latent_cards/
```

鏍圭洰褰曚篃浼氶暅鍍忥細

```text
outputs/misc_full_sae_eval/candidate_latents.csv
```

杩欐槸涓轰簡鍏煎鏃х殑鍥犳灉楠岃瘉鑴氭湰銆?
### 10.2 MISC 澶氭爣绛剧煩闃?
MISC 澶氭爣绛惧姛鑳芥寚鏍囦娇鐢細

```text
records.jsonl + utterance_features.pt
```

閫愭爣绛捐绠楁瘡涓?latent 涓庤鏍囩鐨勫叧鑱旓細

```text
latent 脳 label
```

杈撳嚭鐩綍锛?
```text
outputs/misc_full_sae_eval/functional/misc_label_mapping
```

鍏抽敭鏂囦欢锛?
```text
latent_label_matrix.csv
label_summary.json
label_indicator_matrix.csv
label_fragmentation.json
latent_overlap.json
label_topk_jaccard.json
top_latents_by_label/
top_examples_by_label/
```

`latent_label_matrix.csv` 姣忚琛ㄧず涓€涓?`(label, latent_idx)` 鐨勭粺璁″叧绯伙紝鍖呮嫭锛?
- `cohens_d`
- `abs_cohens_d`
- `auc`
- `directional_auc`
- `p_value`
- `significant_fdr`
- `precision_at_10`
- `precision_at_50`

## 11. 姝ｅ紡杩愯鍛戒护

鏈湴姝ｅ紡鍏ㄩ噺 MISC 涓绘祦绋嬪懡浠ょず渚嬶細

```powershell
conda activate qwen-env-py311
$env:MODEL_DIR="D:\project\NLP_v3\NLP_data\Llama-3.1-8B"

python run_sae_evaluation.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc `
  --data-format misc_full `
  --label-mode misc_multilabel `
  --batch-size 1 `
  --max-seq-len 128 `
  --full-structural `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_full_sae_eval
```

蹇€?smoke 鍛戒护锛?
```powershell
python run_sae_evaluation.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc `
  --data-format misc_full `
  --max-samples 32 `
  --batch-size 1 `
  --max-seq-len 128 `
  --full-structural `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/structural_smoke_misc
```

## 12. 瀵规帴鏃舵渶瀹规槗璇В鐨勭偣

1. **`quality_label` 涓嶆槸琛屼负鏍囩銆?*
   瀹冨彧鏄?high/low 浼氳瘽璐ㄩ噺鏍囩锛岀敤浜庡垎灞傚垎鏋愩€?
2. **`rationale` 涓嶈緭鍏ユā鍨嬨€?*
   瀹冨彧鐢ㄤ簬浜哄伐瀹℃煡鍜岃В閲婏紝涓嶅弬涓?SAE 鎺ㄧ悊銆?
3. **`OTHER` 鏄紓璐ㄩ泦鍚堛€?*
   `ST`銆乣FA`銆乣FI`銆乣AD` 绛夐潪鏍稿績鏍囩閮戒細杩涘叆 `OTHER`锛屼笉鑳芥妸 `OTHER` 褰撴垚鍗曚竴蹇冪悊鍜ㄨ琛屼负銆?
4. **MISC 澶氭爣绛句笉鏄簰鏂ュ垎绫汇€?*
   `RE + REC`銆乣QU + QUO` 鏄埗瀛愭爣绛惧悓鏃跺瓨鍦ㄣ€?
5. **token-level latent 榛樿涓嶄繚瀛樸€?*
   褰撳墠榛樿鍙繚瀛?utterance-level `[N, 32768]` 鐗瑰緛鍜?`[N, 4096]` 妯″瀷鍙ュ悜閲忋€?
6. **缁撴瀯鎸囨爣鍜屽姛鑳芥寚鏍囦娇鐢ㄧ殑绮掑害涓嶅悓銆?*
   缁撴瀯鎸囨爣鍩轰簬 token-level activation/reconstruction 鐨勫湪绾跨粺璁★紱鍔熻兘鎸囨爣鍩轰簬 utterance-level 鑱氬悎鐗瑰緛銆?
7. **褰撳墠鏍锋湰娌℃湁涓婁笅鏂囩獥鍙ｃ€?*
   SAE 鐪嬪埌鐨勬槸鍗曟潯 `unit_text`锛屼笉鏄畬鏁村挩璇笂涓嬫枃銆傝繖鏈夊埄浜庡榻?MISC 琛屼负鍗曞厓锛屼絾浼氬墛寮遍暱鏂囨湰鍒嗗竷涓嬬殑 SAE 閲嶆瀯琛ㄧ幇銆?
## 13. 涓€鍙ヨ瘽鎬荤粨

褰撳墠 MISC 涓绘祦绋嬬殑鏍稿績璁捐鏄細

```text
姣忔潯 misc_annotations JSONL 琛?  -> 鍙?unit_text 浣滀负涓€涓牱鏈?  -> 鏍规嵁 predicted_code/subcode 鐢熸垚澶氭爣绛惧拰 RE 浜屽垎绫绘爣绛?  -> 杈撳叆 Llama tokenizer
  -> 鎶藉彇 blocks.19.hook_resid_post 鐨?[T, 4096] activation
  -> 杈撳叆 OpenMOSS SAE 寰楀埌 [T, 32768] latent
  -> 瀵归潪 padding token 鍋?max pooling
  -> 淇濆瓨 [N, 32768] utterance_features
  -> 璁＄畻缁撴瀯鎸囨爣銆丷E/NonRE 鍔熻兘鎸囨爣鍜?MISC latent-label 鐭╅樀
```
