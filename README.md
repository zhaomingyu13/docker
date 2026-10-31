Qwen3.8-27B-W8A8 DFlash2 与 DSpark 投机部署性能对照（含 DFlash2 叠加 PR #19512）
测试日期：2026-10-30；机器：41_20002（8×Ascend 910B4，仅用 NPU 4-7）；各部署同镜像同参数顺序执行。

1. 测试环境
项	值
目标模型	Qwen3.8-27B-W8A8（Eco-Tech 官方昇腾量化，28.3GB）
vLLM / vllm-ascend	0.32.0+empty / 0.19.1rc2.dev3176+g1eda0ddad
torch / torch-npu	2.10.0+cpu / 2.10.0.post7
镜像	vllm-ascend:20261030
并行 / 精度	TP4 / BF16（W8A8 量化目标）
服务参数	max_num_seqs=8, max_model_len=75808, batch_tokens=32768, prefix caching, chunked prefill, FULL_DECODE_ONLY 图
投机	K7（num_speculative_tokens=7，+1 bonus）；DFlash2 基线草稿 enforce_eager，DSpark 草稿整步图捕获
DFlash2+PR19512	在 DFlash2 基线上叠加 vllm-ascend PR #19512（未合入，head 1caf18ec）：草稿 FULL ACL 图捕获，移除 enforce_eager，VLLM_USE_V2_MODEL_RUNNER=1；5 个源文件 overlay（patches/pr19512/，env 门控）
草稿权重	DFlash2=incoai 官方；DSpark=AICC-Spec-Models Base + 目标 lm_head 按 t2d 掩码切片组装（见 ASSEMBLY.md）
2. 用例构造与步骤
五场景×2 轮共 384 条正式请求，冻结输入（token IDs 与 3.6 27_test 相同）；temperature=0/seed=42 流式请求； 每轮含不计入统计的预热请求；指标为两个有效轮次均值的算术平均。请求经 /v1/completions 直传 token ID 数组， TTFT/TPOT 以携带 token 的 SSE chunk 计时。各部署的服务启动、预热、场景顺序完全一致。

3. 五场景结果对照
同场景相邻行。

场景	并发/请求数	部署	TTFT ms	TPOT ms	请求 E2E s	场景 E2E s	输出吞吐 tokens/s	平均接受长度 含 bonus
8K 输入 + 8K 输出	8 / 64	DFlash2 K7	2368.800	21.739	180.433	1493.819	351.62	4.4574
8K 输入 + 8K 输出	8 / 64	DFlash2 K7 +PR19512	2321.817	11.158	93.716	809.745	648.30	7.3048
8K 输入 + 8K 输出	8 / 64	DSpark K7	2233.692	10.503	88.266	728.878	719.33	6.4668
20K 输入 + 300 输出	4 / 32	DFlash2 K7	4247.507	65.722	23.899	193.322	49.66	2.0405
20K 输入 + 300 输出	4 / 32	DFlash2 K7 +PR19512	3854.687	40.546	15.978	129.661	74.04	4.2396
20K 输入 + 300 输出	4 / 32	DSpark K7	4393.458	37.579	15.630	125.961	76.22	3.8382
GSM8K 自然结束	8 / 64	DFlash2 K7	567.303	16.890	6.841	74.991	318.69	6.2847
GSM8K 自然结束	8 / 64	DFlash2 K7 +PR19512	513.304	14.675	6.307	63.520	398.53	6.1375
GSM8K 自然结束	8 / 64	DSpark K7	487.924	14.045	6.160	61.842	420.13	5.2738
LongBench 上下文	8 / 24	DFlash2 K7	5654.692	300.669	13.476	48.863	49.30	2.7662
LongBench 上下文	8 / 24	DFlash2 K7 +PR19512	5989.914	251.762	12.495	42.541	57.55	3.5287
LongBench 上下文	8 / 24	DSpark K7	5776.737	226.605	12.352	41.483	61.25	3.3391
20K 输入 + 300 输出	1 / 8	DFlash2 K7	2741.429	32.602	12.490	99.925	24.02	2.1611
20K 输入 + 300 输出	1 / 8	DFlash2 K7 +PR19512	2727.243	11.426	6.144	49.156	48.83	4.6442
20K 输入 + 300 输出	1 / 8	DSpark K7	2694.273	9.557	5.552	44.425	54.02	4.2281
注：自然结束场景（S3/S4）输出量因部署而异时，E2E 与吞吐同时受输出长度影响；固定长度场景（S1/S2/S5）输出严格一致。

4. 投机解码与缓存指标
场景	部署	接受率	平均接受长度(不含bonus)	Prefix 命中率	KV 峰值占用	抢占数	两轮输出一致
8K 输入 + 8K 输出	DFlash2 K7	49.39%	3.4574	0.00%	39.44%	0	16/64
8K 输入 + 8K 输出	DFlash2 K7 +PR19512	90.07%	6.3048	0.00%	40.32%	0	4/64
8K 输入 + 8K 输出	DSpark K7	78.10%	5.4668	0.00%	45.27%	0	1/64
20K 输入 + 300 输出	DFlash2 K7	14.86%	1.0405	0.00%	24.68%	0	12/32
20K 输入 + 300 输出	DFlash2 K7 +PR19512	46.28%	3.2396	0.00%	24.82%	0	1/32
20K 输入 + 300 输出	DSpark K7	40.55%	2.8382	0.00%	26.62%	0	22/32
GSM8K 自然结束	DFlash2 K7	75.50%	5.2847	0.00%	21.77%	0	64/64
GSM8K 自然结束	DFlash2 K7 +PR19512	73.39%	5.1375	0.00%	22.25%	0	4/64
GSM8K 自然结束	DSpark K7	61.05%	4.2738	0.00%	27.16%	0	6/64
LongBench 上下文	DFlash2 K7	25.23%	1.7662	0.00%	35.02%	0	17/24
LongBench 上下文	DFlash2 K7 +PR19512	36.12%	2.5287	0.00%	36.04%	0	24/24
LongBench 上下文	DSpark K7	33.42%	2.3391	0.00%	43.11%	0	13/24
20K 输入 + 300 输出	DFlash2 K7	16.59%	1.1611	0.00%	6.40%	0	8/8
20K 输入 + 300 输出	DFlash2 K7 +PR19512	52.06%	3.6442	0.00%	6.56%	0	8/8
20K 输入 + 300 输出	DSpark K7	46.12%	3.2281	0.00%	6.69%	0	8/8
5. PR #19512 叠加说明与正确性验证
PR：vllm-ascend #19512 [Feature] Enable DFlash2 full ACL graph drafting on V2（分支 sunny-rain-63:feat/dflash2-graph-main，head 1caf18ec，未合入）；改动 5 个源文件（dflash2/dflash speculator、dflash aclgraph、attention_v1 FIA band mode、qwen3_dflash2 selector 编译命名空间），git apply --check 对镜像内 1eda0ddad 干净适用。
落地方式：bundle patches/pr19512/ overlay（bootstrap 顶部 env BENCH_APPLY_PR19512=1 门控，在任何 vllm import 前拷贝，fail-fast）；正式 run formal-dflash2pr-001 日志 overlay 标记 6 次、草稿图捕获证据 Capturing dflash2 CUDA graphs (FULL) ✓。
正确性：冒烟（smoke-dflash2pr-002）与 DFlash2 基线（smoke-dflash2-009）三个请求 token 级逐位一致（warmup/gsm8k/synthetic-8k，贪心无损）；gsm8k 答案 #### 2180 一致。
冒烟单请求 TPOT：gsm8k 10.88→7.65ms（-30%）、synthetic-8k 25.65→11.50ms（-55%）；接受率 47.7%→65.4%（+17.7pp，与 PR 修复 selector 编译缓存命名空间有关）。
6. KV 容量与物理占用
部署	KV 池 blocks	引擎口径容量 tokens	75808 长度并发倍数	每卡 KV 物理 GiB
DFlash2 K7	3125	467,258	6.16x	13.35
DFlash2 K7 +PR19512	3125	467,258	6.16x	13.35
DSpark K7	741	501,551	6.62x	13.30
注：各部署每卡 KV 物理背衬相同（同一 14.34GB 缓冲）；blocks 与容量差异来自 vLLM 0.32 分组打包方式（DFlash2 按层小组分页、DSpark 打包成 uniform 组），页大小不同导致页数与引擎容量口径不同；DFlash2 基线与 PR19512 叠加版 blocks 相同（草稿 ACL 图占用未挤占 KV 池）。

7. 统计口径说明
TTFT：请求发出到首个携带 token 的 SSE chunk；TPOT：(末 chunk − 首 chunk)/(输出 token 数−1)
请求 E2E：从请求发出到 [DONE]；场景 E2E（wall）：该轮全部请求（含调度等待）完成耗时
输出吞吐：场景输出 token 总数 / 场景 E2E
接受率 = accepted/proposed；平均接受长度含 bonus = 1 + accepted/drafts；自洽校验 accepted ≤ proposed ≤ 8×drafts
Prefix 命中率 = prefix_cache_hits/queries（block_size=896，S3 的 98-token 输入不足一个 block 故为 0；每轮 warmup 后 reset，冷缓存口径）
KV 峰值占用 = vllm:kv_cache_usage_perc 采样峰值（1s 间隔）
每卡 KV 物理 = untyped_storage().data_ptr() 去重后的 backing buffer 字节数（埋点口径，与 worker 日志吻合；描述符求和为别名口径，勿用）
每轮均含 warmup（不计入）；指标 = 两个有效轮次各自均值的算术平均
8. 版本与配置对照（与 3.6 27_test 差异）
项	3.6 27_test	本测试（3.8）
vLLM	0.30.0rc2	0.32.0
vllm-ascend	0.30.0rc2 时代补丁版	0.19.1rc2.dev3176（原生 dspark/dflash，无需 Free9 五文件补丁）
DSpark batch	8320	32768（与 DFlash2 统一）
DSpark 草稿 lm_head	Free9 运行时推导	checkpoint 自带（按 t2d 掩码从目标 BF16 lm_head 切片组装）
Model Runner	V1（强制）	V2（默认；DFlash2+PR19512 因 dflash2-graph 黑名单需 env 强制 V2）
KV 埋点	类级补丁	函数级钩 attn_utils.init_kv_cache（vllm-ascend patch 层会顶掉类级补丁）
DFlash2 K7 审计：384 请求全部通过（run formal-dflash2-001）
DFlash2 K7 +PR19512 审计：384 请求全部通过（run formal-dflash2pr-001）
DSpark K7 审计：384 请求全部通过（run formal-dspark-001）
