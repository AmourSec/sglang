# GLM-5.2 C8 / Ascend 950 sparse KV 算子迁移设计

日期：2026-10-24。状态：代码、测试脚本和 Mac 静态复核已完成；950 实机验收待执行。

## 已确认的范围

- 将 `vllmascend_sep_fif` 的 `994236084^..cd84b83c` 中 SparseKvPlan / SparseKvTransfer 及所需 LRU 管理迁入本仓库的 Ascend decode 流程。
- 用户确认源仓库 `clean_simt` 能运行。当前 HEAD `2804765f8e016574dcbda205c8751e6281be24ef` 是 `cd84b83c566e62ac26934758cc563b152292f41e` 的下一提交，仅规范化行尾；关键 sparse KV 源文件无逻辑差异。
- SGLang 阅读基线：`904691e0b5dc4516c5bc9771874d03dd0cadb470`。
- 用户明确本轮只做本地 KV offload 和 decode/MTP 接入，不做 P/D 分离或 URMA 接收链路。
- 硬件为 950 系列，CANN 9.1.0，torch / torch_npu 2.10.0。
- 必须覆盖 MTP=0 和 MTP=3、batch size 至少到 32、多个序列长度且包含约 140k。
- 用户最终明确：MTP=3 预测 3 个 token，每序列同时 decode 4 个 token，分别有 4 组 TopK2048；MTP=0 每序列 1 个 token/1 组 TopK2048。
- Mac 上做静态检查；950 编译和真实数值测试由用户 push/pull 后运行。本机不宣称 NPU 测试已通过。

## 模型规格及证据边界

已读取 Z.ai 发布的 [GLM-5.2-FP8 config.json](https://huggingface.co/zai-org/GLM-5.2-FP8/raw/main/config.json)，用于核实模型结构维度，不能把它的 FP8 权重量化配置当作 C8 配置。

| 字段 | 值 |
| --- | --- |
| architecture | GlmMoeDsaForCausalLM |
| hidden size / backbone 层数 | 6144 / 78 |
| attention heads | 64，TP 后本地头数由实际配置决定 |
| kv_lora_rank / qk_rope_head_dim | 512 / 64 |
| qk_nope_head_dim / v_head_dim | 192 / 256 |
| index_n_heads / index_head_dim / index_topk | 32 / 128 / 2048 |
| index_topk_freq / index_skip_topk_offset | 4 / 3 |
| index_share_for_mtp_iteration | true |
| num_nextn_predict_layers | 1；不等于推测解码步数 |

源 manager `register_kv_caches` 明确要求主 SFA cache 为 BF16，允许 C8 的是留在 device 的 indexer cache。迁移的主 payload 是一个 latent KV head 的 K512 + rope64，每 token 合计 1152 bytes。不能使用 config 的 `num_key_value_heads=64` 去给压缩 MLA cache 多分配 64 个 head。

源 `attention/indexer.py:173-180` 根据硬件能力选择 C8 的物理格式：950/A5 有 `FP8_ATTENTION` 能力，因此使用 **FP8 E4M3FN + FP32 scale**；非 FP8 分支才使用 INT8 + FP16 scale。量化前是标准归一化 Hadamard128。目标 `NPUMLATokenToKVPool` 现有量化 indexer 分支与 `dsa_kv_cache_store_fp8` 绑定；应复用现有量化 helper，同时把 indexer dtype/scale 与主 BF16 cache 独立描述，不能强制把950 C8改成INT8。

源 `ascend_config.py:1390` 的 resident capacity 默认4096，与 TopK2048不同；本次保持默认4096，允许显式配置不小于TopK的容量。

## 方案选择

建议将两个 kernel、host tiling、Torch binding 和构建脚本作为独立算子扩展保存在 SGLang 仓库，并接入现有 `sparsity_driven_kv_offload`。这样用户只需推送本仓库；运行不依赖 vLLM 安装。

另一条路线是修改外部 `sgl-kernel-npu` 仓库，长期维护更集中，但本地没有该仓库，且需额外交付和安装一份 fork。直接导入 vLLM extension 会耦合 vLLM ABI、初始化及其大量无关算子，不建议。

保留 CANN custom-op 的 host/tiling/kernel 结构、动态 UB 和 SIMT 编译选项。不能把带 tiling 的 kernel 文件直接放进普通 C++ extension 并假定可以编译。独立构建需要明确 CANN 工程生成/打包入口，须提供源码与脚本并做静态依赖核查，950 实机验证构建。

## 数据布局与生命周期

1. **独立连续平面。** 源 Transfer 按连续 K 和 rope 平面寻址；SGLang 当前 host buffer 是交错 `[request, seq, 1, 576]`。不能直接把 split 出来的非连续 view 传给源 kernel。建议新路径使用独立 `[host_slots, 1, 512]` / `[host_slots, 1, 64]` BF16 平面，resident 使用 `[row, capacity, 1, dim]`。
2. **地址语义。** kernel 接收 NPU 可访问的 host DVA；CPU `.data_ptr()` 是 HVA，不能替代 DVA。复用 SGLang 已有 host memory provider 前须核实 950 映射能力；必要时沿用源实现的 MemFabric giant-page/get_dva 方案。测试必须包含真实 DDR→HBM，不能仅以 HBM→HBM 代替。
3. **存储所有权。** 每个 target/draft KV pool 各持有唯一 DDR store 和请求 generation；同一 draft runner 的多个 step backend 共享。当前 `AscendAttnMultiStepDraftBackend` 每步新建 backend，若沿用每 backend 创建 manager 的行为，会重复分配、覆盖 alloc hook，并可能让 draft extend 与 decode 读写不同 store。
4. **请求身份。** 使用 pool slot + generation 判断请求重用；单纯 slot ID 不足。alloc/reset 必须作用于所有相关状态，target 和 draft 的 host 存储独立。
5. **行语义。** Plan 的 persistent state 和 resident payload 按 query row 对齐。MTP=3 为 `[B,4,2048]` 展平为 `[4B,2048]`，BS32 时最多 128 行；MTP=0 为 B 行。按线性预测链处理，不能把 request 数当 query token 数或允许树形兄弟共享同一位置的 KV。
6. **可见范围与回滚。** 每个 query 使用自己的 causal visible length；stable prefix 对应本次可能重写的后缀起点。缓存行若在若干轮未激活，恢复时仍需失效期间可能被重写的后缀；设计需记录上次 stable frontier 或等价 mutation generation，不能只看当前 forward 的起点。
7. **共享 TopK。** 模型允许跨层和跨 MTP iteration 共享 top-k；这不代表可以共享不同层的 KV payload。仅在 resident token/LRU 状态保持一致时复用 plan，否则每层重新规划。
8. **并发与 graph。** D2H、Plan、Transfer、attention 必须有明确 stream/event 顺序；持久状态和 workspace 预分配，capture/replay 不做设备 `.item()` 或动态分配持久地址。
9. **缓存边界。** 当前 per-request DDR store 没有自动接入 prefix/radix cache、HiCache 或 P/D 接收。新路径须明确配置约束，不能在只拥有 indexer cache 的情况下误报历史 main KV 可复用；chunked prefill 和同请求续写仍需正确保存历史。

## 算子契约

- Plan：`req_ids:int64[R]`；`topk_indices:int32[T,K]`；逐 row stable/visible/token_to_req；`block_table:int32[R,P]`；`active_rows:int32[1]`。
- 原地状态：last request IDs、slot-to-token、LRU slot permutation、current slots、miss counts/tokens/slots、workspace。
- `H=max(32,next_power_of_two(2*K))`；每行 workspace 为 `3*H+2*C+K` 个 int32。112 KiB 内走 UB，否则分配完整 GM workspace。
- Transfer：两个 host DVA、block table、miss plan，以及连续 BF16 resident K/rope；保持按 miss 搬运及未触及槽内容不变。
- 保留源重复 token 语义：已有 resident hit 只回填 TopK 第一次位置，后续重复可产生独立 miss；LRU 从旧到新，miss 使用最老可淘汰槽。
- binding 需补齐 dtype、shape、contiguity、workspace 容量和 resident byte-size 校验，防止错误调用造成越界；不改变合法输入的 kernel 语义。

## 正确性测试与交付

测试不加载完整模型权重。固定模型维度并以独立 CPU reference 校验，索引、LRU 状态和 BF16 搬运要求精确一致；attention 结果使用 FP32 reference 和明确容差。

| 范围 | 必须覆盖 |
| --- | --- |
| Batch | 1、2、4、8、16、31、32；不同活动行数与 graph padding |
| Sequence | 1、页边界前后、2047/2048/2049、4096、32768、140000、143360；混合长度 batch |
| 模型 payload | K512/rope64 BF16、indexer head dim128、top-k2048；TP 本地 attention heads |
| Plan | 冷缓存、全命中、全 miss、混合命中、LRU 淘汰、重复/非法 top-k、无效 block、换请求、重排 batch、inactive rows |
| Workspace | UB 与 GM 分支、容量边界、非整 tile、跨 64 AIV 的行数 |
| Transfer | 非连续物理 block table、跨页、不同请求和层、真实 DDR DVA、未写槽 sentinel 保持、guard 区不变 |
| MTP | 宽度1/4，覆盖 draft/verify/extend、逐 token 因果、accept/reject 后缀覆盖、停止/恢复行、共享 TopK、多 step 共享 store |
| Graph | capture/replay 两组不同 metadata/数据；padding 与真实长度变化；不能只测首次 capture |
| 集成 | chunked prefill→decode；连续 decode；主模型与 draft 存储隔离；配置不支持组合明确失败 |

140k/BS32 的单层主 host KV 约 4.81 GiB；78 层约 374.91 GiB/进程，另有 padding/draft。capacity=4096 的78层resident在MTP=3时约43.88 GiB/进程，capacity=2048时约21.94 GiB。固定 resident/metadata 必须在 indexer pool 容量计算前预留。单算子测试按层运行、分块生成 reference，避免一次分配78层全模型数据。

计划交付：独立扩展源码和构建脚本、Python runtime 接入、CPU reference、950 单算子/集成测试、环境探测及分组运行脚本、执行说明与静态检查结果。不能把未执行的 NPU 测试标记为通过。

## 软件与技能依据

- 本仓 `docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx` 当前推荐 CANN9.1.0、Python3.12、torch2.10.0、torch_npu2.10.0.post6、sgl-kernel-npu2026.9.0.post9；950 必须使用对应 950 wheel 或 `Ascend950PR_9599` 源码 target。用户的 torch_npu2.10.0 是否带 post 后缀由测试脚本记录，不擅自升级环境。
- 已读取 [aikg npu-arch-capability-check](https://github.com/AmourSec/aikg/blob/a420cb1233b283391c5d9d90cb28e0eb9485ebeb/skills/npu-arch-capability-check/SKILL.md)：区分 SKU/SoC/NpuArch，核实 950 / DAV_3510，使用平台查询核数，先正确性再性能；记录缺失的硬件与运行证据。没有执行或安装该仓库脚本。

## 工作状态

- 已完成：指定提交、源 runtime/kernel/binding/build、目标 decode/indexer/MTP/store 生命周期的并行阅读；真实模型结构和软件文档核对。
- 已确认：本地 offload 范围、MTP=3 的四 query 语义、公开模型维度和目标软件版本。用户要求模型其余信息自行查询。
- 已实现：独立算子扩展、runtime/MTP、C8/indexer 格式适配、测试及脚本。四个独立审查范围均无剩余阻塞项；CPU runner 51 项通过，40 个新增/修改 Python 文件静态检查通过。NPU 编译和运行按用户安排留给 950 环境。

## 交付入口

- 构建：`sgl-kernel-npu-sparse-kv/README.md`。
- 测试和启动参数：`test/manual/ascend/sparse_kv/README.md`。
- 统一脚本：`scripts/ascend/run_sparse_kv_950.sh`；`full` 包含独立进程运行的真实 manager 生命周期测试。
- Mac 核对：CPU runner 51 passed；manual 目录 8 passed、8 个 NPU 模块跳过（这 8 项已包含在 51 项内）；Python AST/Ruff/format/isort、Shell 语法和自编代码 diff 检查通过。完整暂存检查另外显示 46 个 CANN 第三方支撑/许可证文件存在原始空白格式告警，保留上游原文。
- 已覆盖的实机测试场景不等于实机测试通过；当前没有 CANN 编译、DDR DVA、ACL graph、attention 数值或整模型服务的设备执行证据。
- 实现阶段交付时未执行 commit/push；用户随后授权提交并推送到 `AmourSec/sglang` 的 `wzj_kv_offload` 分支。
