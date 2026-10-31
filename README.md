# Qwen3.8-27B-W8A8 DFlash2（causal 草稿）与 DSpark 投机部署性能对照

测试日期：2026-10-31；机器：41_20002（8×Ascend 910B4，仅用 NPU 4-7）；两部署同镜像同参数顺序执行。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 目标模型 | Qwen3.8-27B-W8A8（Eco-Tech 官方昇腾量化，28.3GB） |
| vLLM / vllm-ascend | 0.32.0+empty / 0.19.1rc2.dev3176+g1eda0ddad |
| 镜像 | vllm-ascend:20261030 |
| 并行 / 精度 | TP4 / BF16（W8A8 量化目标） |
| 服务参数 | max_num_seqs=8, max_model_len=75808, batch_tokens=32768, prefix caching, chunked prefill, FULL 图 |
| 投机 | K7（num_speculative_tokens=7，+1 bonus）；两部署草稿均为 FULL ACL 图捕获 |
| DFlash2 | 官方 incoai 草稿权重 + **草稿 config 本地修改（causal 化）** + vllm-ascend PR #19512 overlay（5 文件，update2 版） |
| 草稿 config 修改 | `is_causal: false→true`、`layer_types: 5×sliding_attention→5×full_attention`、`sliding_window: 2048→null`、`use_sliding_window: true→false`（非因果滑窗 → 因果全注意力） |
| DSpark | AICC-Spec-Models Base + 目标 lm_head 按 t2d 掩码切片组装（原生，无叠加） |

## 2. 用例构造与步骤

五场景×2 轮共 384 条正式请求，冻结输入（token IDs 与 3.6 27_test 相同）；`temperature=0/seed=42` 流式请求；
每轮含不计入统计的预热请求；指标为两个有效轮次均值的算术平均。请求经 `/v1/completions` 直传 token ID 数组，
TTFT/TPOT 以携带 token 的 SSE chunk 计时。两部署的服务启动、预热、场景顺序完全一致。

## 3. 五场景结果对照

同场景相邻行。

| 场景 | 并发/请求数 | 部署 | TTFT ms | TPOT ms | 请求 E2E s | 场景 E2E s | 输出吞吐 tokens/s | 平均接受长度 含 bonus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | 8 / 64 | DFlash2 K7 causal 草稿 | 2237.952 | 14.136 | 118.029 | 965.907 | 542.82 | 4.8235 |
| 8K 输入 + 8K 输出 | 8 / 64 | DSpark K7 | 2233.692 | 10.503 | 88.266 | 728.878 | 719.33 | 6.4668 |
| 20K 输入 + 300 输出 | 4 / 32 | DFlash2 K7 causal 草稿 | 4280.025 | 47.812 | 18.576 | 149.482 | 64.22 | 2.2315 |
| 20K 输入 + 300 输出 | 4 / 32 | DSpark K7 | 4393.458 | 37.579 | 15.630 | 125.961 | 76.22 | 3.8382 |
| GSM8K 自然结束 | 8 / 64 | DFlash2 K7 causal 草稿 | 492.025 | 12.672 | 5.469 | 52.716 | 482.60 | 6.0721 |
| GSM8K 自然结束 | 8 / 64 | DSpark K7 | 487.924 | 14.045 | 6.160 | 61.842 | 420.13 | 5.2738 |
| LongBench 上下文 | 8 / 24 | DFlash2 K7 causal 草稿 | 5589.240 | 285.545 | 12.159 | 41.348 | 58.47 | 2.8499 |
| LongBench 上下文 | 8 / 24 | DSpark K7 | 5776.737 | 226.605 | 12.352 | 41.483 | 61.25 | 3.3391 |
| 20K 输入 + 300 输出 | 1 / 8 | DFlash2 K7 causal 草稿 | 2675.283 | 17.649 | 7.953 | 63.626 | 37.72 | 2.2811 |
| 20K 输入 + 300 输出 | 1 / 8 | DSpark K7 | 2694.273 | 9.557 | 5.552 | 44.425 | 54.02 | 4.2281 |

注：自然结束场景（GSM8K/LongBench）输出量因部署而异时，E2E 与吞吐同时受输出长度影响；固定长度场景输出严格一致。

## 4. 投机解码与缓存指标

| 场景 | 部署 | 接受率 | 平均接受长度(不含bonus) | Prefix 命中率 | KV 峰值占用 | 抢占数 | 两轮输出一致 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8K 输入 + 8K 输出 | DFlash2 K7 causal 草稿 | 54.62% | 3.8235 | 0.00% | 45.90% | 0 | 11/64 |
| 8K 输入 + 8K 输出 | DSpark K7 | 78.10% | 5.4668 | 0.00% | 45.27% | 0 | 1/64 |
| 20K 输入 + 300 输出 | DFlash2 K7 causal 草稿 | 17.59% | 1.2315 | 0.00% | 26.65% | 0 | 32/32 |
| 20K 输入 + 300 输出 | DSpark K7 | 40.55% | 2.8382 | 0.00% | 26.62% | 0 | 22/32 |
| GSM8K 自然结束 | DFlash2 K7 causal 草稿 | 72.46% | 5.0721 | 0.00% | 26.99% | 0 | 48/64 |
| GSM8K 自然结束 | DSpark K7 | 61.05% | 4.2738 | 0.00% | 27.16% | 0 | 6/64 |
| LongBench 上下文 | DFlash2 K7 causal 草稿 | 26.43% | 1.8499 | 0.00% | 39.84% | 0 | 14/24 |
| LongBench 上下文 | DSpark K7 | 33.42% | 2.3391 | 0.00% | 43.11% | 0 | 13/24 |
| 20K 输入 + 300 输出 | DFlash2 K7 causal 草稿 | 18.30% | 1.2811 | 0.00% | 6.86% | 0 | 8/8 |
| 20K 输入 + 300 输出 | DSpark K7 | 46.12% | 3.2281 | 0.00% | 6.69% | 0 | 8/8 |

## 5. 草稿 config 修改说明与生效验证

- 修改内容：草稿 config 四字段 causal 化（见第 1 节表）；官方原版备份于 `config_bak.json` 与 `/home/z30075380/ms_master_config.json`。
- 运行时生效证据（冒烟 `smoke-dflash2cfg-001` + 正式 run `formal-dflash2cfg-001`）：
  - 草稿层 KV spec 由官方 config 的 `SlidingWindowSpec(2048)` 单独分组，变为与目标 16 个 full-attention 层合并的 21 层 uniform 组（`server/kv-allocation.json`）；
  - KV 池由 467,258 tokens / 6.16x 升至 503,581 tokens / 6.64x（与 DSpark 501,551 / 6.62x 趋同）；
  - 草稿 FULL 图 8/8 档捕获成功（`Capturing dflash2 CUDA graphs (FULL)`）；
  - 冒烟投机数值与官方 config 版分叉（接受率 65.36%→50.24%），确认行为变化。
- 注意：PR #19512 的非因果 FIA band-mode 修正分支在此配置下不触发（草稿层已 causal）。
- 冒烟 8 项 checks 全过、零抢占；gsm8k 答案 `#### 2180` 正确。

## 6. KV 容量与物理占用

| 部署 | KV 池 blocks | 引擎口径容量 tokens | 75808 长度并发倍数 |
| --- | --- | --- | --- |
| DFlash2 K7 causal 草稿 | 744 | 503,581 | 6.64x |
| DSpark K7 | 741 | 501,551 | 6.62x |

注：causal 化后草稿层与目标 full-attention 层合并分组，两部署分组结构趋同，容量口径几乎一致；每卡 KV 物理背衬相同（14.34GB 缓冲）。

## 7. 统计口径说明

- TTFT：请求发出到首个携带 token 的 SSE chunk；TPOT：(末 chunk − 首 chunk)/(输出 token 数−1)
- 请求 E2E：从请求发出到 [DONE]；场景 E2E（wall）：该轮全部请求（含调度等待）完成耗时
- 输出吞吐：场景输出 token 总数 / 场景 E2E
- 接受率 = accepted/proposed；平均接受长度含 bonus = 1 + accepted/drafts；自洽校验 accepted ≤ proposed ≤ 8×drafts
- Prefix 命中率：每轮 warmup 后 reset prefix cache（冷缓存口径，恒 0 为设计行为）
- KV 峰值占用 = `vllm:kv_cache_usage_perc` 采样峰值（1s 间隔）
- 每轮均含 warmup（不计入）；指标 = 两个有效轮次各自均值的算术平均

## 8. 结论

1. **TPOT/吞吐：DSpark 五场景中的四个更快**（8k8k 快 34.7%、20k300_c1 快 84.6%、20k300_c4 快 27.2%、LongBench 快 26.0%）；**唯一例外 GSM8K——DFlash2 causal 草稿 TPOT 快 9.8%**（12.67 vs 14.04ms，吞吐 482.6 vs 420.1 tokens/s）。TTFT 两者持平。
2. **接受率：DFlash2 causal 草稿仅 GSM8K 占优（72.5% vs 61.1%）**，四个合成/长文本场景全面落后（8k8k 54.6% vs 78.1%、20k300 两场景约腰斩）。
3. **草稿 causal 化的影响**（对照同 PR 官方草稿 config 的 formal-dflash2pr-001）：官方 DFlash2 草稿权重按非因果滑窗（2048）设定训练，causal+全注意力化后草稿预测质量显著下降——合成场景接受率大幅回落（8k8k 90.1%→54.6%、20k300_c1 52.1%→18.3%、20k300_c4 46.3%→17.6%），自然语言短上下文几乎无损（GSM8K 73.4%→72.5%）且 TPOT 反而更快（14.67→12.67ms）；长上下文 decode 场景承受接受率下降与全注意力开销的双重代价。
4. **确定性改善**：causal 化后低并发场景两轮输出完全一致（20k300_c4 32/32、20k300_c1 8/8；官方 config 版为 1/32、8/8；DSpark 为 22/32）；高并发（C8）仍存在图模式 padding 引发的非确定性。
5. 零抢占（两部署全部场景）、384 请求逐 token 审计全过（`raw-audit.json`：两部署均 all_passed=True）。

## 9. 数据路径

- DFlash2（causal 草稿）：`runs/formal-dflash2cfg-001/`（冒烟 `runs/smoke-dflash2cfg-001/`；KV 分组证据 `server/kv-allocation.json`）
- DSpark：`runs/formal-dspark-001/`（2026-10-30）
- 草稿 config 存档：`/home/models/Qwen3.8-27B-DFlash2/config.json`（修改版）、`config_bak.json`（官方原版）、`/home/z30075380/ms_master_config.json`（ModelScope 官方对照）
- PR overlay：`patches/pr19512/`；PR diff 存档：`patches/pr19512_update2.diff`
- 参照（同 PR 官方草稿 config 版）：`runs/formal-dflash2pr-001/`、`reports/Qwen38-27B_DFlash2_vs_DSpark.md`
