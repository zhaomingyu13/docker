# Qwen3.8-27B-W8A8 DFlash2 与 DSpark 投机部署性能对照

测试日期：2026-10-30；机器：41_20002（8×Ascend 910B4，仅用 NPU 4-7）；两后端同镜像同参数顺序执行。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 目标模型 | Qwen3.8-27B-W8A8（Eco-Tech 官方昇腾量化，28.3GB） |
| vLLM / vllm-ascend | 0.32.0+empty / 0.19.1rc2.dev3176+g1eda0ddad |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post7 |
| 镜像 | vllm-ascend:20261030 |
| 并行 / 精度 | TP4 / BF16（W8A8 量化目标） |
| 服务参数 | max_num_seqs=8, max_model_len=75808, batch_tokens=32768, prefix caching, chunked prefill, FULL_DECODE_ONLY 图 |
| 投机 | K7（num_speculative_tokens=7，+1 bonus）；DFlash2 草稿 enforce_eager，DSpark 草稿整步图捕获 |
| 草稿权重 | DFlash2=incoai 官方；DSpark=AICC-Spec-Models Base + 目标 lm_head 按 t2d 掩码切片组装（见 ASSEMBLY.md） |

## 2. 用例构造与步骤

五场景×2 轮共 384 条正式请求，冻结输入（token IDs 与 3.6 27_test 相同）；`temperature=0/seed=42` 流式请求；
每轮含不计入统计的预热请求；指标为两个有效轮次均值的算术平均。请求经 `/v1/completions` 直传 token ID 数组，
TTFT/TPOT 以携带 token 的 SSE chunk 计时。两后端的服务启动、预热、场景顺序完全一致。

## 3. 五场景结果对照

同场景相邻两行。

| 场景 | 并发/请求数 | 部署 | TTFT ms | TPOT ms | 请求 E2E s | 场景 E2E s | 输出吞吐 tokens/s | 平均接受长度 含 bonus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | 8 / 64 | DFlash2 K7 | 2368.800 | 21.739 | 180.433 | 1493.819 | 351.62 | 4.4574 |
| 8K 输入 + 8K 输出 | 8 / 64 | DSpark K7 | 2233.692 | 10.503 | 88.266 | 728.878 | 719.33 | 6.4668 |
| 20K 输入 + 300 输出 | 4 / 32 | DFlash2 K7 | 4247.507 | 65.722 | 23.899 | 193.322 | 49.66 | 2.0405 |
| 20K 输入 + 300 输出 | 4 / 32 | DSpark K7 | 4393.458 | 37.579 | 15.630 | 125.961 | 76.22 | 3.8382 |
| GSM8K 自然结束 | 8 / 64 | DFlash2 K7 | 567.303 | 16.890 | 6.841 | 74.991 | 318.69 | 6.2847 |
| GSM8K 自然结束 | 8 / 64 | DSpark K7 | 487.924 | 14.045 | 6.160 | 61.842 | 420.13 | 5.2738 |
| LongBench 上下文 | 8 / 24 | DFlash2 K7 | 5654.692 | 300.669 | 13.476 | 48.863 | 49.30 | 2.7662 |
| LongBench 上下文 | 8 / 24 | DSpark K7 | 5776.737 | 226.605 | 12.352 | 41.483 | 61.25 | 3.3391 |
| 20K 输入 + 300 输出 | 1 / 8 | DFlash2 K7 | 2741.429 | 32.602 | 12.490 | 99.925 | 24.02 | 2.1611 |
| 20K 输入 + 300 输出 | 1 / 8 | DSpark K7 | 2694.273 | 9.557 | 5.552 | 44.425 | 54.02 | 4.2281 |

注：自然结束场景（S3/S4）输出量因部署而异时，E2E 与吞吐同时受输出长度影响；固定长度场景（S1/S2/S5）输出严格一致。

## 4. 投机解码与缓存指标

| 场景 | 部署 | 接受率 | 平均接受长度(不含bonus) | Prefix 命中率 | KV 峰值占用 | 抢占数 | 两轮输出一致 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | DFlash2 K7 | 49.39% | 3.4574 | 0.00% | 39.44% | 0 | 16/64 |
| 8K 输入 + 8K 输出 | DSpark K7 | 78.10% | 5.4668 | 0.00% | 45.27% | 0 | 1/64 |
| 20K 输入 + 300 输出 | DFlash2 K7 | 14.86% | 1.0405 | 0.00% | 24.68% | 0 | 12/32 |
| 20K 输入 + 300 输出 | DSpark K7 | 40.55% | 2.8382 | 0.00% | 26.62% | 0 | 22/32 |
| GSM8K 自然结束 | DFlash2 K7 | 75.50% | 5.2847 | 0.00% | 21.77% | 0 | 64/64 |
| GSM8K 自然结束 | DSpark K7 | 61.05% | 4.2738 | 0.00% | 27.16% | 0 | 6/64 |
| LongBench 上下文 | DFlash2 K7 | 25.23% | 1.7662 | 0.00% | 35.02% | 0 | 17/24 |
| LongBench 上下文 | DSpark K7 | 33.42% | 2.3391 | 0.00% | 43.11% | 0 | 13/24 |
| 20K 输入 + 300 输出 | DFlash2 K7 | 16.59% | 1.1611 | 0.00% | 6.40% | 0 | 8/8 |
| 20K 输入 + 300 输出 | DSpark K7 | 46.12% | 3.2281 | 0.00% | 6.69% | 0 | 8/8 |

## 5. KV 容量与物理占用

| 部署 | KV 池 blocks | 引擎口径容量 tokens | 75808 长度并发倍数 | 每卡 KV 物理 GiB |
| --- | --- | --- | --- | --- |
| DFlash2 K7 | 3125 | 467,258 | 6.16x | 13.35 |
| DSpark K7 | 741 | 501,551 | 6.62x | 13.30 |

注：两后端每卡 KV 物理背衬相同（同一 14.34GB 缓冲）；blocks 与容量差异来自 vLLM 0.32 分组打包方式（DFlash2 按层小组分页、DSpark 打包成 uniform 组），页大小不同导致页数与引擎容量口径不同。

## 6. 统计口径说明

- TTFT：请求发出到首个携带 token 的 SSE chunk；TPOT：(末 chunk − 首 chunk)/(输出 token 数−1)
- 请求 E2E：从请求发出到 [DONE]；场景 E2E（wall）：该轮全部请求（含调度等待）完成耗时
- 输出吞吐：场景输出 token 总数 / 场景 E2E
- 接受率 = accepted/proposed；平均接受长度含 bonus = 1 + accepted/drafts；自洽校验 accepted ≤ proposed ≤ 8×drafts
- Prefix 命中率 = prefix_cache_hits/queries（block_size=896，S3 的 98-token 输入不足一个 block 故为 0）
- KV 峰值占用 = `vllm:kv_cache_usage_perc` 采样峰值（1s 间隔）
- 每卡 KV 物理 = `untyped_storage().data_ptr()` 去重后的 backing buffer 字节数（埋点口径，与 worker 日志吻合；描述符求和为别名口径，勿用）
- 每轮均含 warmup（不计入）；指标 = 两个有效轮次各自均值的算术平均

## 7. 版本与配置对照（与 3.6 27_test 差异）

| 项 | 3.6 27_test | 本测试（3.8） |
|---|---|---|
| vLLM | 0.30.0rc2 | 0.32.0 |
| vllm-ascend | 0.30.0rc2 时代补丁版 | 0.19.1rc2.dev3176（原生 dspark/dflash，无需 Free9 五文件补丁） |
| DSpark batch | 8320 | 32768（与 DFlash2 统一） |
| DSpark 草稿 lm_head | Free9 运行时推导 | checkpoint 自带（按 t2d 掩码从目标 BF16 lm_head 切片组装） |
| Model Runner | V1（强制） | V2（默认） |
| KV 埋点 | 类级补丁 | 函数级钩 `attn_utils.init_kv_cache`（vllm-ascend patch 层会顶掉类级补丁） |

- DFlash2 K7 审计：384 请求全部通过（run `formal-dflash2-001`）
- DSpark K7 审计：384 请求全部通过（run `formal-dspark-001`）
