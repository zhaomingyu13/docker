# Qwen3.8-27B-W8A8 DFlash2 性能测试 —— PR #19512 最新版（前两场景）

**测试时间**：2026-10-31
**机器**：41_20002（NPU 4-7，TP4）
**run id**：`formal-dflash2pr2-002`（冒烟前置验证：`smoke-dflash2pr2-002`）
**范围**：DFlash2 部署叠加 vllm-ascend PR #19512 最新版（2026-10-31 head）后，重测前两个场景——`20k300_c4_n32` 与 `8k8k_c8_n64`，各 1 轮。

## 测试对象

vllm-ascend PR #19512（DFlash2 草稿 FULL ACL 图捕获，draft 状态）。本次为该 PR 2026-10-31 最新 head 的重测，相对上次叠加版本的实质生产变更为：

- `attention_v1.py`：非因果注意力按滑窗分流——无滑窗走原始非因果 FIA（`sparse_mode=0`）；带滑窗（非因果 DFlash 层）改用 FIA band mode（`sparse_mode=4`）+ 压缩三角模板 mask + 对称 `W-1` 可见性，且模板只构建一次（`_non_causal_swa_attn_mask` 缓存，eager/图执行共享）
- `dflash2/speculator.py`：候选打分编译上下文改用 `self.vllm_config`（不再穿透 model 取配置）

## 环境与配置（冻结）

| 项 | 值 |
|---|---|
| 模型 | Qwen3.8-27B-W8A8（BF16） |
| 部署 | TP4 / Model Runner V2 / 端口 19340 |
| 投机 | dflash2（DFlash2 草稿），K=7 |
| 图模式 | 目标 FULL ACL 图 + 草稿 FULL ACL 图（PR 解锁，非 eager） |
| 调度 | FDO / batch_tokens 32768 / max_num_seqs 8 / max_model_len 75808 |
| 软件 | vllm 0.32.0+empty / vllm-ascend 0.19.1rc2.dev3176+g1eda0ddad（镜像 vllm-ascend:20261030）+ PR #19512 overlay（5 文件，`BENCH_APPLY_PR19512=1` 门控） |
| 输入 | 冻结数据集（SHA256 校验通过，审计 96/96 请求逐 token 验证 PASS） |

冒烟门槛（`smoke-dflash2pr2-002`）：8 项 checks 全过（valid=true）、overlay 5 文件生效、目标与草稿 FULL 图均 8/8 捕获成功、token 级输出一致性验证通过、零抢占。

## 场景结果（round1）

### 20k300_c4_n32（输入 20480 / 输出 300 tokens，并发 4，32 请求）

| 指标 | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| TTFT (ms) | 4110.80 | 3057.74 | 9660.23 | 10525.86 | 10615.81 |
| TPOT (ms) | 40.03 | 41.78 | 66.23 | 68.37 | 69.12 |
| E2E (s) | 16.08 | 15.80 | 24.27 | 25.35 | 25.54 |

- 墙钟 130.09 s；吞吐 73.80 output tokens/s（0.246 requests/s）
- 投机：接受率 45.08%，平均接受长度（含 bonus）4.156，drafts 2322 / accepted 7328（proposed 16254）
- KV 峰值 24.7%，运行并发峰值 4，零抢占

### 8k8k_c8_n64（输入 8192 / 输出 8192 tokens，并发 8，64 请求）

| 指标 | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| TTFT (ms) | 2274.90 | 1508.73 | 8108.74 | 8431.24 | 8431.35 |
| TPOT (ms) | 11.08 | 10.89 | 12.45 | 15.42 | 20.17 |
| E2E (s) | 93.06 | 91.03 | 103.55 | 127.80 | 166.69 |

- 墙钟 785.64 s；吞吐 667.34 output tokens/s（0.0815 requests/s）
- 投机：接受率 91.63%，平均接受长度（含 bonus）7.414，drafts 70737 / accepted 453729（proposed 495159）
- KV 峰值 40.2%，运行并发峰值 8 / 等待峰值 2，零抢占

## 有效性

- 两场景 `valid=true`，checks 全过（请求完整、服务端成功数一致、零抢占、cache 明细有效、前缀查询覆盖输入、投机计数自洽）
- raw-audit：96/96 请求 token 级验证 PASS（20k300 32 请求 + 8k8k 64 请求）
- finish_reason 全部为 length（符合合成数据 cap 预期），无空文本

## 数据路径

- 汇总：`runs/formal-dflash2pr2-002/aggregate.json`（scope 说明：本轮为子集 1 轮，逐场景权威数据在 round1/summary.json）
- 场景明细：`runs/formal-dflash2pr2-002/{20k300_c4_n32,8k8k_c8_n64}/round1/`（summary.json / responses.jsonl / metrics / streams）
- 审计：`runs/formal-dflash2pr2-002/raw-audit.json`
- 驱动日志：`runs/formal-dflash2pr2-002-driver.log`；服务日志：`runs/formal-dflash2pr2-002/server/service.log`
- PR diff 存档：`patches/pr19512_update2.diff`（本包镜像）/ `/home/z30075380/pr19512_update2.diff`（服务器）；overlay bundle：`patches/pr19512/`（旧版备份 `patches/pr19512_old_v1/`）
