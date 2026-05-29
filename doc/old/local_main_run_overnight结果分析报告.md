# local_main_run_overnight 缁撴灉鍒嗘瀽鎶ュ憡

> 鍒嗘瀽鏃堕棿锛?026-04-25
> 缁撴灉鐩綍锛歚outputs/local_main_run_overnight`
> 涓绘祦绋嬶細`run_sae_evaluation.py -> causal/run_experiment.py`
> 鏁版嵁闆嗭細`data/mi_re`
> SAE checkpoint 璇箟锛歚--checkpoint-topk-semantics hard`

## 1. 鎬讳綋缁撹

杩欐 `local_main_run_overnight` 宸茬粡瀹屾暣璺戦€氾紝鍥犳灉楠岃瘉闃舵鐘舵€佷负 `completed`锛屾€昏€楁椂绾?`12h25m55s`銆傜浉姣斾箣鍓嶁€滈暱鏃堕棿杩愯鍚庢棤閿欒淇℃伅涓柇鈥濈殑闂锛岃繖娆″凡缁忔垚鍔熷啓鍑?`run_status.json`銆乣causal_run.log`銆乣run_events.jsonl` 鍜屽悇绫荤粨鏋?JSON锛岃鏄庢柊澧炵殑杩涘害涓庨敊璇棩蹇楁満鍒舵湁鏁堛€?
浠庣鐮旂粨璁轰笂鐪嬶紝瀹為獙缁欏嚭浜嗚緝寮虹殑鈥淪AE latent 涓?RE/NonRE 鍖哄垎鐩稿叧鈥濈殑璇佹嵁锛氬崟鍙橀噺绛涢€夊緱鍒?`2187 / 32768` 涓?FDR 鏄捐憲 latent锛孴op-20 sparse probe 杈惧埌 `AUC=0.9158`锛宒ense probe 杈惧埌 `AUC=0.9677`銆傚洜鏋滈獙璇佷腑锛孴op 缁?ablation 浼氭槑鏄鹃檷浣?RE 鏍锋湰鐨?RE logit锛孏20 鐨?RE logit 涓嬮檷绾?`-1.258`锛岃€?Bottom20 鍜?Random20 鍩烘湰鎺ヨ繎闆讹紝鏀寔鍊欓€?latent 涓嶆槸闅忔満鍣０銆?
浣嗛渶瑕佽皑鎱庤В閲?steering/sufficiency锛氭鍚?steering 浼氭彁鍗?RE logit锛屼絾鍚屾椂 NonRE logit 涔熻鏄庢樉鎻愬崌锛屽挨鍏?G20 涓?NonRE 澧炲箙楂樹簬 RE 澧炲箙銆傝繖璇存槑褰撳墠 latent 缁勬洿鍍忊€滄縺娲讳竴绫荤浉鍏宠涔?琛ㄨ揪椋庢牸鐨勬柟鍚戔€濓紝杩樹笉鑳界洿鎺ュ绉颁负楂樺害閫夋嫨鎬х殑 RE 鍥犳灉寮€鍏炽€?
## 2. 杩愯瀹屾垚鎯呭喌

鍏抽敭鐘舵€佹枃浠讹細

- `outputs/local_main_run_overnight/causal_validation/run_status.json`
- `outputs/local_main_run_overnight/causal_validation/causal_run.log`
- `outputs/local_main_run_overnight/causal_validation/summary_tables.md`
- `outputs/local_main_run_overnight/causal_validation/results_necessity.json`
- `outputs/local_main_run_overnight/causal_validation/results_sufficiency.json`
- `outputs/local_main_run_overnight/causal_validation/results_selectivity.json`
- `outputs/local_main_run_overnight/causal_validation/results_group.json`

杩愯鐘舵€侊細

| 椤圭洰 | 缁撴灉 |
| --- | --- |
| 鍥犳灉闃舵鐘舵€?| `completed` |
| 鎬昏€楁椂 | `44754.903s`锛岀害 `12h25m55s` |
| batch size | `4` |
| max seq len | `128` |
| lambdas | `0.5, 1.0, 1.5, 2.0` |
| side effect samples | `16` |
| fatal traceback | 鏂囦欢瀛樺湪锛屼絾鏈瀹屾垚杩愯锛屾棤 fatal failure |

## 3. SAE 缁撴瀯鎸囨爣

缁撴瀯鎸囨爣鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/sae_eval/metrics_structural.json`
- `outputs/local_main_run_overnight/sae_eval/metrics_ce_kl.json`

鏍稿績鏁板€硷細

| 鎸囨爣 | 鏁板€?| 瑙ｈ |
| --- | ---: | --- |
| `n_tokens` | `28217` | 瀹屾暣 MI-RE 鏁版嵁闆嗕笂鐨?token 鏁?|
| `mse` | `6.4261` | SAE 閲嶆瀯璇樊鍋忛珮 |
| `cosine_similarity` | `0.8607` | 鏂瑰悜鐩镐技鎬у皻鍙?|
| `explained_variance` | `-0.9661` | raw EV 涓鸿礋锛屼粛鍙兘浣滀负杈呭姪璇婃柇 |
| `explained_variance_paper` | `-0.8363` | paper-style raw 鎸囨爣浠嶄负璐?|
| `l0_mean` | `43.3156` | 骞冲潎姣?token 婵€娲荤害 43 涓壒寰?|
| `dead_ratio` | `0.4543` | 绾?45.4% latent 鍦ㄨ鏁版嵁闆嗕笂鏈縺娲?|
| `ce_loss_orig` | `5.0569` | 鍘熸ā鍨?CE |
| `ce_loss_sae` | `6.5475` | SAE 閲嶆瀯鍚?CE |
| `delta_lm_loss` | `+1.4906` | 閲嶆瀯瀵艰嚧璇█妯″瀷鎹熷け鏄庢樉涓婂崌 |
| `kl_divergence` | `2.4183` | 杈撳嚭鍒嗗竷鎵板姩杈冨ぇ |

閲嶈娉ㄦ剰锛?
- `metric_primary` 浠嶆爣璁颁负 `ev_openmoss_legacy`锛屼絾鏈 `explained_variance_openmoss_legacy` 瀹為檯涓?`None`銆?- 杩欒鏄庡綋鍓?full structural 璺緞娌℃湁鎶?official legacy EV 姝ｇ‘钀界洏锛屾垨璇ヨ矾寰勪笅 legacy accumulator 娌℃湁琚惎鐢ㄣ€?- 鍥犳鏈鎶ュ憡涓嶈兘鐢?`ev_openmoss_legacy` 鍜岃鏂囨暟鍊煎仛鐩存帴杈炬爣姣旇緝锛屽彧鑳藉熀浜?raw/normalized EV銆丆E/KL 鍜屽姛鑳?鍥犳灉鎸囨爣鍋氳В閲娿€?
## 4. 鍔熻兘鎬ф寚鏍?
鍔熻兘鎸囨爣鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/sae_eval/metrics_functional.json`
- `outputs/local_main_run_overnight/sae_eval/candidate_latents.csv`

鍗曞彉閲忕瓫閫夛細

| 椤圭洰 | 鏁板€?|
| --- | ---: |
| 鎬?latent 鏁?| `32768` |
| FDR 鏄捐憲 latent | `2187` |
| FDR alpha | `0.05` |
| Top latent | `29759` |
| Top latent Cohen's d | `1.0163` |
| Top latent AUC | `0.7546` |

Top-10 鍗曞彉閲?latent锛?
| 鎺掑悕 | latent | Cohen's d | AUC | 鏂瑰悜 |
| ---: | ---: | ---: | ---: | --- |
| 1 | `29759` | `+1.016` | `0.755` | RE 鏇村己 |
| 2 | `19435` | `+0.920` | `0.711` | RE 鏇村己 |
| 3 | `31930` | `+0.869` | `0.696` | RE 鏇村己 |
| 4 | `5663` | `+0.861` | `0.698` | RE 鏇村己 |
| 5 | `9959` | `-0.838` | `0.341` | NonRE 鏇村己 |
| 6 | `14875` | `+0.837` | `0.689` | RE 鏇村己 |
| 7 | `1516` | `+0.830` | `0.713` | RE 鏇村己 |
| 8 | `1211` | `+0.787` | `0.704` | RE 鏇村己 |
| 9 | `11660` | `-0.760` | `0.361` | NonRE 鏇村己 |
| 10 | `11405` | `+0.755` | `0.685` | RE 鏇村己 |

Probe 缁撴灉锛?
| Probe | Accuracy | F1 | AUC |
| --- | ---: | ---: | ---: |
| sparse k=1 | `0.7384` | `0.6995` | `0.7545` |
| sparse k=5 | `0.7897` | `0.7814` | `0.8629` |
| sparse k=20 | `0.8260` | `0.8268` | `0.9158` |
| dense probe | `0.9280` | `0.9290` | `0.9677` |
| diffmean | `0.8166` | `0.8135` | `0.8967` |

瑙ｈ锛?
- Top-20 sparse probe 宸茬粡鎻愪緵浜嗚緝寮虹殑鍙В閲婁綆缁村垽鍒俊鍙枫€?- dense probe 鏄捐憲鏇村己锛岃鏄?RE/NonRE 宸紓鍒嗗竷鍦ㄦ洿骞跨殑 SAE latent 绌洪棿涓€?- 浣?sparse k=20 涓?dense probe 涔嬮棿浠嶆湁鏄庢樉宸窛锛屽悗缁鏋滆拷姹傗€滃皯閲忓彲瑙ｉ噴 latent 鍗冲彲瑙ｉ噴浠诲姟宸紓鈥濓紝闇€瑕佽繘涓€姝ョ瓫閫夊拰浜哄伐璇箟璇勪及銆?
## 5. 鍥犳灉楠岃瘉缁撴灉

### 5.1 Necessity锛氭秷铻嶆槸鍚︿細闄嶄綆 RE 淇″彿

缁撴灉鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/causal_validation/results_necessity.json`

鏍稿績鏁板€奸噰鐢?zero ablation锛?
| 缁勫埆 | RE logit 鍙樺寲 | NonRE logit 鍙樺寲 | 瑙ｈ |
| --- | ---: | ---: | --- |
| G1 | `-0.112` | `-0.015` | 鍗?latent 宸叉湁寮卞繀瑕佹€?|
| G5 | `-0.547` | `+0.148` | RE 淇″彿鏄庢樉涓嬮檷 |
| G10 | `-1.001` | `+0.686` | RE 淇″彿寮轰笅闄?|
| G20 | `-1.258` | `+1.816` | RE 淇″彿寮轰笅闄嶏紝浣?NonRE 鍙嶅悜鍙樺寲涔熷緢澶?|
| Bottom20 | `-0.000` | `-0.001` | 璐熸帶鎺ヨ繎闆?|
| Random20 | `-0.001` | `+0.010` | 闅忔満鎺ф帴杩戦浂 |

缁撹锛?
- Top latent 缁勫 RE 鍒ゅ埆淇″彿鍏锋湁鏄庢樉蹇呰鎬с€?- Bottom20 鍜?Random20 杩戦浂锛岃鏄庢晥鏋滀笉鏄换鎰?latent 娑堣瀺閮戒細鍑虹幇銆?- G20 鐨?NonRE 鍙樺寲寰堝ぇ锛屾彁绀鸿繖浜?latent 褰卞搷鐨勬槸鍒ゅ埆杈圭晫鎴栧箍涔夎涔夋柟鍚戯紝鑰屼笉鏄彧浣滅敤浜?RE 鏍锋湰銆?
### 5.2 Sufficiency锛歴teering 鏄惁鑳芥彁鍗?RE 淇″彿

缁撴灉鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/causal_validation/results_sufficiency.json`

浠?`lambda=2.0` 涓轰緥锛?
| 缁勫埆 | 妯″紡 | RE logit 鍙樺寲 | NonRE logit 鍙樺寲 | 鏀瑰杽姣斾緥 |
| --- | --- | ---: | ---: | ---: |
| G1 | constant | `+0.495` | `+0.550` | `0.961` |
| G5 | constant | `+0.589` | `+0.546` | `0.854` |
| G10 | constant | `+0.352` | `+0.453` | `0.921` |
| G20 | constant | `+0.348` | `+0.428` | `0.947` |
| Orthogonal | direction | `-0.002` | `+0.020` | `0.519` |
| Random_dir | direction | `+0.041` | `+0.036` | `0.632` |

缁撹锛?
- 姝ｅ悜 steering 鏈夋晥锛孴op 缁勬瘮 Orthogonal/Random_dir 鎺у埗鏂瑰悜鏇村己銆?- 浣?NonRE 涔熻鎻愬崌锛屽挨鍏?G1/G10/G20 涓?NonRE 澧炲箙涓嶄綆浜?RE銆?- 鍥犳 sufficiency 璇佹嵁鏀寔鈥滆繖浜涙柟鍚戣兘鎺ㄥ姩妯″瀷杩涘叆鐩稿叧琛ㄨ揪鍖哄煙鈥濓紝浣嗛€夋嫨鎬т笉瓒炽€?
### 5.3 Selectivity锛氱敓鎴愬壇浣滅敤

缁撴灉鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/causal_validation/results_selectivity.json`

褰撳墠 selectivity 浣跨敤灏忔牱鏈敓鎴愪唬鐞嗘寚鏍囷細

| 缁勫埆 | content retention | delta TTR | delta repetition | generated RE logit delta |
| --- | ---: | ---: | ---: | ---: |
| G1/G5/G10/G20 | `1.000` | `0.000` | `0.000` | `0.000` |
| Orthogonal | `0.985` | `-0.001` | `+0.003` | `-0.001` |
| Random_dir | `0.942` | `-0.005` | `+0.005` | `+0.139` |

瑙ｉ噴鏃惰淇濆畧锛?
- Top 缁勭敓鎴愪唬鐞嗘寚鏍囨病鏈夊彉鍖栵紝鍙兘璇存槑褰撳墠 `cond_token` 骞查鍦ㄧ敓鎴愯缃笅娌℃湁浜х敓鍙娴嬭緭鍑哄彉鍖栥€?- 杩欎笉鑳界瓑鍚屼簬鈥滄棤鍓綔鐢ㄢ€濓紝鏇村彲鑳芥槸鐢熸垚渚ц瘎浼版牱鏈皯銆佸共棰勫急鎴栨寚鏍囪繃绮椼€?- 鍚庣画闇€瑕佹墿澶ф牱鏈€佸鍔犱汉宸ヨ瘎鍒嗘垨 AI judge锛屾墠鑳芥妸 selectivity 浣滀负璁烘枃绾х粨璁恒€?
## 6. Group Structure锛歭atent 缁勪笉鏄畝鍗曠嚎鎬у彔鍔?
缁撴灉鏉ヨ嚜锛?
- `outputs/local_main_run_overnight/causal_validation/results_group.json`

鍏抽敭鍙戠幇锛?
- G10 cumulative effect 鍦ㄧ 10 涓?latent 鍚庤揪鍒?`+1.930`銆?- 鍓嶅嚑涓?latent 鐨勭疮璁℃晥鏋滃苟涓嶅崟璋冿細渚嬪 k=2 涓?`-0.920`锛宬=3 涓?`-1.669`锛宬=4 鍙嶅脊鍒?`+0.338`銆?- synergy score 涓?`+1.640`锛岃В閲婂瓧娈典负 `positive (super-additive)`銆?
杩欒鏄庡€欓€?latent 涔嬮棿瀛樺湪鏄庢樉缁勫悎鏁堝簲锛屼笉瀹滄妸姣忎釜 latent 褰撴垚鐙珛鍙姞鐨勨€滆涔夋寜閽€濄€傚悗缁В閲?latent 缁勬椂锛屽簲浼樺厛瑙ｉ噴鈥滃瓙绌洪棿/缁勫悎鏂瑰悜鈥濓紝鍐嶈В閲婂崟涓?latent銆?
## 7. 褰撳墠鏈€閲嶈鐨勯闄╃偣

1. 缁撴瀯鎸囨爣鐨?`ev_openmoss_legacy` 鏈惤鐩?
   鏂囨。鍜屼唬鐮佸彛寰勪粛璇翠富鎸囨爣鏄?`ev_openmoss_legacy`锛屼絾鏈缁撴灉閲岃瀛楁涓?`None`銆傝繖浼氬奖鍝嶅拰璁烘枃鎸囨爣鐨勭洿鎺ュ榻愶紝闇€瑕佷慨澶?full structural 璺緞銆?
2. SAE 閲嶆瀯璐ㄩ噺涓嶅鐞嗘兂
   `delta_lm_loss=+1.4906`銆乣KL=2.4183`銆乺aw EV 涓鸿礋锛岃鏄?SAE 閲嶆瀯瀵瑰師妯″瀷杈撳嚭鎵板姩鏄庢樉銆傚姛鑳?鍥犳灉缁撴灉鍙互缁х画鍒嗘瀽锛屼絾涓嶈兘鎶?SAE 褰撲綔浣庢崯閲嶆瀯銆?
3. 鍥犳灉 steering 鐨勯€夋嫨鎬т笉瓒?
   Top 缁?steering 鎻愬崌 RE logit 鐨勫悓鏃朵篃鎻愬崌 NonRE logit锛岃鏄庡畠鎹曟崏鐨勫彲鑳芥槸鏇村娉涚殑璇箟/椋庢牸鏂瑰悜銆?
4. 鐢熸垚渚?selectivity 璇佹嵁鍋忓急
   褰撳墠灏忔牱鏈敓鎴愪唬鐞嗘寚鏍囧熀鏈笉鍙橈紝鏃犳硶鏀拺寮虹粨璁恒€傚缓璁鍔?AI judge 鎴栦汉宸?rubric銆?
淇璁板綍锛?
- 2026-04-25 宸蹭慨澶嶄富娴佺▼浠ｇ爜锛歚run_sae_evaluation.py` 鐨?`--full-structural` 璺緞鐜板湪浼氬拰璇婃柇鑴氭湰涓€鏍峰惎鐢?`OfficialMetricsAccumulator`锛屽苟閫氳繃 `apply_full_structural_metrics(...)` 鍐欏叆 `ev_openmoss_legacy`銆?- 娉ㄦ剰锛氭湰鎶ュ憡鍒嗘瀽鐨勬槸鏃х粨鏋滅洰褰?`outputs/local_main_run_overnight`锛屾棫鐨?`metrics_structural.json` 涓嶄細鑷姩鍥炲～銆傞渶瑕侀噸璺?`run_sae_evaluation.py --full-structural` 鍚庯紝鏂扮粨鏋滄墠浼氬寘鍚慨澶嶅悗鐨?official legacy EV銆?
## 8. 寤鸿涓嬩竴姝?
浼樺厛绾т粠楂樺埌浣庯細

1. 淇 `metrics_structural.json` 涓?`ev_openmoss_legacy=None` 鐨勯棶棰橈紝纭繚 official legacy EV 鍦?full dataset 涓婂彲澶嶇幇钀界洏銆?2. 瀵?Top-20 latent 鍋?AI judge 鎴栦汉宸ヨ涔夋爣娉紝楠岃瘉瀹冧滑鏄惁鐪熺殑瀵瑰簲鍙嶆槧寮忓€惧惉銆佹儏缁壙鎺ャ€佸杩般€佹敮鎸佹€у洖搴旂瓑 RE 姒傚康銆?3. 瀵瑰洜鏋滈獙璇佸鍔犳洿寮虹殑閫夋嫨鎬ф寚鏍囷紝渚嬪鍙湅 RE-specific tokens銆佹儏缁瘝銆佸杩版ā鏉匡紝閬垮厤鍙敤鏁翠綋 RE logit銆?4. 灏?G1/G5/G10/G20 鐨?latent 缁勫拰鍗曞彉閲?Top-20 鍒嗗紑瑙ｉ噴锛氬墠鑰呮潵鑷洜鏋?缁勯€夋嫨锛屽悗鑰呮潵鑷崟鍙橀噺缁熻锛屼袱鑰呬笉鏄悓涓€鎺掑簭銆?5. 濡傛灉瑕佸啓璁烘枃寮忕粨璁猴紝寤鸿琛ㄨ堪涓猴細鈥滃彂鐜颁簡涓€缁勪笌 RE/NonRE 鍒ゅ埆鏄捐憲鐩稿叧锛屽苟鍦?ablation 涓叿鏈夊繀瑕佹€х殑 SAE latent锛泂teering 缁撴灉鏄剧ず鏂瑰悜鏈夋晥浣嗛€夋嫨鎬т粛闇€杩涗竴姝ラ獙璇併€傗€?
## 9. 鍙紩鐢ㄧ殑缁撴灉鏂囦欢

- SAE 缁撴瀯鎸囨爣锛歚outputs/local_main_run_overnight/sae_eval/metrics_structural.json`
- SAE CE/KL锛歚outputs/local_main_run_overnight/sae_eval/metrics_ce_kl.json`
- SAE 鍔熻兘鎸囨爣锛歚outputs/local_main_run_overnight/sae_eval/metrics_functional.json`
- 鍊欓€?latent锛歚outputs/local_main_run_overnight/sae_eval/candidate_latents.csv`
- 鍥犳灉杩愯鐘舵€侊細`outputs/local_main_run_overnight/causal_validation/run_status.json`
- 鍥犳灉姹囨€昏〃锛歚outputs/local_main_run_overnight/causal_validation/summary_tables.md`
- Necessity锛歚outputs/local_main_run_overnight/causal_validation/results_necessity.json`
- Sufficiency锛歚outputs/local_main_run_overnight/causal_validation/results_sufficiency.json`
- Selectivity锛歚outputs/local_main_run_overnight/causal_validation/results_selectivity.json`
- Group structure锛歚outputs/local_main_run_overnight/causal_validation/results_group.json`
