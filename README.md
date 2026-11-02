# Qwen3.8-27B-W8A8 DFlash2 投机部署性能测试（叠加 PR #19512）

测试日期：2026-10-31；机器：41_20002（8×Ascend 910B4，仅用 NPU 4-7）；run id：`formal-dflash2cfg-002`。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 目标模型 | Qwen3.8-27B-W8A8（Eco-Tech 官方昇腾量化，28.3GB） |
| vLLM / vllm-ascend | 0.32.0+empty / 0.19.1rc2.dev3176+g1eda0ddad |
| 镜像 | vllm-ascend:20261030 |
| 并行 / 精度 | TP4 / BF16（W8A8 量化目标） |
| 服务参数 | max_num_seqs=8, max_model_len=75808, batch_tokens=32768, prefix caching, chunked prefill |
| 投机 | dflash2 K7（num_speculative_tokens=7，+1 bonus），官方 incoai 草稿权重（config 为官方原版：非因果 + 滑窗 2048） |
| 图模式 | 目标 + 草稿 FULL ACL 图（草稿非 eager，PR #19512 解锁） |
| PR #19512 | vllm-ascend PR #19512（未合入，2026-10-31 最新 head）：5 文件 overlay（patches/pr19512/，env `BENCH_APPLY_PR19512=1` 门控）；`VLLM_USE_V2_MODEL_RUNNER=1` |

## 2. 用例构造与步骤

五场景×2 轮共 384 条正式请求，冻结输入（token IDs 与 3.6 27_test 相同）；`temperature=0/seed=42` 流式请求；
每轮含不计入统计的预热请求；指标为两个有效轮次均值的算术平均。请求经 `/v1/completions` 直传 token ID 数组，
TTFT/TPOT 以携带 token 的 SSE chunk 计时。

## 3. 五场景结果

| 场景 | 并发/请求数 | TTFT ms | TPOT ms | 请求 E2E s | 场景 E2E s | 输出吞吐 tokens/s | 平均接受长度 含 bonus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | 8 / 64 | 2406.804 | 10.838 | 91.183 | 743.315 | 705.38 | 7.5827 |
| 20K 输入 + 300 输出 | 4 / 32 | 4193.380 | 39.507 | 16.006 | 129.339 | 74.22 | 4.2625 |
| GSM8K 自然结束 | 8 / 64 | 514.680 | 14.671 | 6.389 | 65.304 | 394.64 | 6.0424 |
| LongBench 上下文 | 8 / 24 | 5954.766 | 249.700 | 12.377 | 42.059 | 58.20 | 3.5287 |
| 20K 输入 + 300 输出 | 1 / 8 | 2714.242 | 11.265 | 6.083 | 48.668 | 49.31 | 4.6442 |

注：自然结束场景（GSM8K/LongBench）输出量随内容而异，E2E 与吞吐同时受输出长度影响；固定长度场景输出严格一致。

## 4. 投机解码与缓存指标

| 场景 | 接受率 | 平均接受长度(不含bonus) | Prefix 命中率 | KV 峰值占用 | 抢占数 | 两轮输出一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | 94.04% | 6.5827 | 0.00% | 40.24% | 0 | 16/64 |
| 20K 输入 + 300 输出 | 46.61% | 3.2625 | 0.00% | 24.68% | 0 | 7/32 |
| GSM8K 自然结束 | 72.03% | 5.0424 | 0.00% | 22.25% | 0 | 4/64 |
| LongBench 上下文 | 36.12% | 2.5287 | 0.00% | 36.04% | 0 | 24/24 |
| 20K 输入 + 300 输出 | 52.06% | 3.6442 | 0.00% | 6.56% | 0 | 8/8 |

## 5. PR #19512 叠加说明与正确性验证

- PR：vllm-ascend #19512 `[Feature] Enable DFlash2 full ACL graph drafting on V2`（未合入，2026-10-31 最新 head，diff 存档 `patches/pr19512_update2.diff`）。核心内容：DFlash2 草稿 FULL ACL 图捕获（移除 enforce_eager 限制）、selector 编译上下文、非因果滑窗 FIA band-mode 修正（band mask 一次性构建缓存）。
- 落地方式：bundle `patches/pr19512/` overlay（bootstrap 顶部 env `BENCH_APPLY_PR19512=1` 门控，在任何 vllm import 前拷贝 5 个文件，fail-fast）。
- 生效证据（本次 run `formal-dflash2cfg-002` 服务日志）：`BENCH_PR19512_OVERLAY_APPLIED 5 files` × 6 进程；草稿图捕获 `Capturing dflash2 CUDA graphs (FULL): 8/8` 档。
- 正确性：冒烟（`smoke-dflash2cfg-002`）8 项 checks 全过、零抢占、gsm8k 答案 `#### 2180` 正确；投机数值与同配置历史冒烟逐位一致（接受率 0.6536 / 含 bonus 5.575 / drafts 120 / accepted 549），贪心确定性可复现。
- 草稿 config 为官方原版（sha256 与 ModelScope master 一致）：`is_causal: false`、`layer_types: 5×sliding_attention`、`sliding_window: 2048`、`use_sliding_window: true`——非因果滑窗草稿，PR 的 FIA band-mode 修正分支在此配置下生效。

## 6. KV 容量与物理占用

| 项 | 值 |
| --- | --- |
| KV 池 blocks | 3125 |
| 引擎口径容量 tokens | 467,258 |
| 75808 长度并发倍数 | 6.16x |
| 每卡 KV 物理 GiB | 13.35 |

注：草稿层为 SlidingWindowSpec(2048) 单独分组（5 层）；每卡 KV 物理背衬 14.34GB 缓冲（`untyped_storage().data_ptr()` 去重口径）。

## 7. 统计口径说明

- TTFT：请求发出到首个携带 token 的 SSE chunk；TPOT：(末 chunk − 首 chunk)/(输出 token 数−1)
- 请求 E2E：从请求发出到 [DONE]；场景 E2E（wall）：该轮全部请求（含调度等待）完成耗时
- 输出吞吐：场景输出 token 总数 / 场景 E2E
- 接受率 = accepted/proposed；平均接受长度含 bonus = 1 + accepted/drafts；自洽校验 accepted ≤ proposed ≤ 8×drafts
- Prefix 命中率：每轮 warmup 后 reset prefix cache（冷缓存口径，恒 0 为设计行为）
- KV 峰值占用 = `vllm:kv_cache_usage_perc` 采样峰值（1s 间隔）
- 每轮均含 warmup（不计入）；指标 = 两个有效轮次各自均值的算术平均

## 8. 有效性

- 全部场景 `valid=true`，checks 全过（请求完整、服务端成功数一致、零抢占、cache 明细有效、前缀查询覆盖输入、投机计数自洽）
- raw-audit：384/384 请求逐 token 审计 PASS（`runs/formal-dflash2cfg-002/raw-audit.json`）
- 零抢占（全部场景）；冻结输入 SHA256 校验通过

## 9. 数据路径

- 汇总与场景明细：`runs/formal-dflash2cfg-002/`（各场景 round1/round2：summary.json / responses.jsonl / metrics / streams）
- 审计：`runs/formal-dflash2cfg-002/raw-audit.json`；驱动日志：`runs/formal-dflash2cfg-002-driver.log`
- 冒烟：`runs/smoke-dflash2cfg-002/`（KV 分组证据 `server/kv-allocation.json`）
- PR overlay：`patches/pr19512/`（旧版备份 `patches/pr19512_old_v1/`）；PR diff 存档：`patches/pr19512_update2.diff`
- 草稿 config：`/home/models/Qwen3.8-27B-DFlash2/config.json`（官方原版）；历史草稿 causal 化对照数据：`runs/formal-dflash2cfg-001/`
