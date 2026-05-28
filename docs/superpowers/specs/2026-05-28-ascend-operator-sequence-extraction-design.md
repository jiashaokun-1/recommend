# 推荐模型 Kernel 级算子序抽取 — 设计文档

- **日期**: 2026-05-28
- **状态**: 已评审，待实现
- **范围**: 本次 spec 仅覆盖 P1 + P2（见分期）

## 1. 背景与目标

本项目用于推荐模型建模，最终目标是评估**算子的性能上限**（roofline）。

第一步要打通的功能是「抓算子序」：

- **输入**：一个 PyTorch 推荐模型（含 TorchRec）。
- **输出**：kernel 级的融合算子序，每个算子带输入/输出的 shape 和 dtype。
- **用途**：供后续 roofline 性能上限评估消费。

### 关键决策（来自评审）

| 决策点 | 结论 |
|---|---|
| 模型框架 | PyTorch（含 TorchRec） |
| 算子粒度 | 编译器融合后的 kernel（Ascend = torchair/GE 融合图；GPU = Inductor） |
| 推理/训练 | 仅推理 forward，单图 |
| shape 处理 | 方案1：specialize 固定代表性 shape；IR 字段预留符号维度容纳能力 |
| 目标设备 | 主：Ascend NPU；次：GPU 做性能对比 |
| Ascend 抽取路径 | GE 静态图（算子序+shape）+ profiler（实测耗时），但**两者解耦** |
| 是否实跑 | **算子序抽取与 roofline 上限全程不实跑**；profiler 是唯一需实跑、且独立可选 |

### 一个核心澄清

roofline 性能上限是**纯解析计算**：拿算子 shape + dtype + 类型，套 FLOPs/访存字节公式，除以硬件峰值，全程不跑模型。profiler 实测仅用于**偶尔校准**这个上限离真实有多远，不是每次都跑。

抓算子序本身是**编译期**动作（一次），优先用 FakeTensor/export 模式只做图构建与 shape 传播、不在 NPU 真执行 kernel，且结果缓存复用。

## 2. 整体架构与分期

```
                   ┌─────────────────────────┐
推荐模型(PyTorch) → │  公共 IR（后端中立）      │ → 下游：roofline 性能上限 + 跨设备对比
                   │  融合算子序+shape/dtype   │
                   └─────────────────────────┘
                      ▲           ▲
        ┌─────────────┘           └──────────────┐
   Ascend 抽取器                              GPU 抽取器
   ① torchair GE 图 dump → 算子序+静态shape    ① Inductor → 算子序+shape
   ② torch_npu profiler → 实测耗时回填(可选)    ② torch.profiler → 实测耗时(可选)
```

**设计原则**：公共 IR 是核心契约，所有抽取器产出它、所有下游消费它。两个后端融合策略不同（GE vs Inductor），故**不做 kernel-to-kernel 硬对比**，而是在 IR 里保留逻辑算子类型与源码映射，让下游按层/按逻辑算子做归一化对比。

### 分期

| 阶段 | 内容 | 是否实跑 | 本次 spec |
|---|---|---|---|
| **P1** | 公共 IR + Ascend 静态算子序抽取（GE 图，编译一次、可缓存） | 否 | ✅ 核心 |
| **P2** | profiler 耗时回填，作为独立可选工具，默认不调用 | 是 | ✅ 可选交付 |
| P3 | GPU/Inductor 抽取器，复用同一 IR | 否 | 下个 spec |
| P4 | roofline 性能上限评估器 + 跨设备对比 | 否 | 下个 spec |

P3、P4 各自独立成 spec，消费同一 IR，互不阻塞。

## 3. 公共 IR 结构

```jsonc
// 顶层：一次抓取 = 一个模型 + 一组 specialize shape 配置
ModelOpSeqCapture {
  schema_version: "0.1",
  model_id, model_name,
  backend: "ascend",              // 抽取器来源；GPU 时为 "gpu"
  capture_mode: "specialize",     // 当前静态；字段预留，未来 "symbolic"
  device_info: { chip: "Ascend910B" },   // 下游按此查峰值算 roofline
  input_spec: {
    inputs: [ { name, shape, dtype, format } ],
    symbol_bindings: {}           // 现在空；未来 {s0: 1024} ← 符号维度容纳点
  },
  guards: [],                     // 现在空；未来存有效区间约束
  ops: [ OpNode... ]              // 按执行序拓扑排列的融合算子序
}

OpNode {
  id,                             // 稳定 id = 执行序号
  op_type,                        // 归一化类型：MatMul/LayerNorm/Softmax/TBE_Lookup/FusedElementwise...
  backend_op_name,                // GE 原始节点名 ← profiler 回填的 join key
  fusion_group_id,                // 属于哪个融合 kernel
  inputs:  [ TensorDesc... ],
  outputs: [ TensorDesc... ],
  attrs: { ... },                 // 算子特有：transpose_a/b、eps、axis...
  source_map: { module_path, aten_op },  // 逻辑映射 → 支撑跨后端归一化对比
  measured: null                  // 仅 profiler 可选工具回填 { latency_us, ... }
}

TensorDesc {
  shape: [ Dim ],                 // Dim = int（现在）| "s0*256"（未来符号表达式）★符号容纳点
  dtype,                          // fp16/bf16/fp32/int8...
  format,                         // ★Ascend 关键：ND/NZ/NC1HWC0...（影响访存字节）；GPU 退化为 stride 语义
  stride: [int] | null
}
```

### 设计要点

1. **`shape` 的 Dim 是联合类型**：现在存 int，未来塞符号表达式字符串，不动结构即可升级到方案2（符号 shape）。
2. **`format` 字段**：Ascend 的 NZ/NC1HWC0 等内部格式改变实际访存字节数，roofline 算访存量必须知道格式；GPU 上退化为 stride 语义。
3. **`backend_op_name` + `source_map`**：前者是 profiler 实测耗时回填的连接键；后者让融合策略不同、kernel 不 1:1 的跨后端场景下，下游仍能按逻辑算子/按层归一化对比。

## 4. Ascend 静态抽取流程（P1，本次核心）

```
① 加载模型(eval) → ② 构造代表性输入 → ③ torchair 编译+dump GE 图 → ④ 解析图 → ⑤ 归一化 → ⑥ 产出IR+缓存
```

- **① 加载模型**：`nn.Module`，eval 模式，`torch_npu` 环境。
- **② 构造代表性输入**：推荐模型需单独造 `KeyedJaggedTensor`（稀疏）+ dense 张量，固定 batch 与各特征长度。需要一个「输入构造器」按模型签名生成 specialize 输入。
- **③ 编译 + dump**：`torch.compile(backend=torchair)` 配置 dump GE 融合图。优先 **FakeTensor / export 模式**，只做图构建 + shape 传播，**不在 NPU 真执行 kernel**；编译一次即得。
- **④ 解析**：读 GE 图（pbtxt/proto），节点 → `OpNode`，`output_desc` → shape/dtype/**format**，边 → 拓扑序。
- **⑤ 归一化**：维护一张 `GE 算子名 → 归一化 op_type` 映射表。先覆盖推荐模型主力算子（MatMul/Addmm、TBE_Lookup、LayerNorm、Softmax、elementwise、concat/gather、reduction），其余落 `Unknown` 不阻塞。TBE 查表当 opaque 节点，记录输入 desc，FLOPs 模型留给 P4。
- **⑥ 缓存**：IR 落 JSON，按 `(model_id, shape配置hash, backend)` 缓存，同配置下游零运行复用。

## 5. profiler 可选工具（P2）

独立工具，默认不调用，仅偶尔校准时手动运行：

1. 跑 N 次模型，采集 msprof / `torch_npu.profiler` 输出。
2. 解析输出，得到按 kernel 名索引的实测耗时。
3. 按 `backend_op_name` join 回 IR，填充 `measured` 字段。
4. **另存**一份带实测的 IR，不污染静态 IR。

## 6. 模块边界

| 模块 | 职责 | 依赖 | 接口 |
|---|---|---|---|
| `ir` | 定义 IR 数据结构 + JSON 序列化/校验 | 无 | `ModelOpSeqCapture` 等 dataclass / schema |
| `input_builder` | 按模型签名生成 specialize 输入（含 KJT） | torch, torchrec | `build_inputs(model, shape_cfg) -> inputs` |
| `ascend_extractor` | torchair 编译 + dump + 解析 GE 图 → IR | torch_npu, torchair | `extract(model, inputs, device_info) -> ModelOpSeqCapture` |
| `op_normalizer` | GE 算子名 → 归一化 op_type 映射 | 无 | `normalize(ge_op) -> op_type` |
| `cache` | 按 key 存取 IR JSON | ir | `get/put(key) -> IR` |
| `profiler_attach`（可选） | 跑模型 + 解析 profiler + join 回 IR | torch_npu.profiler | `attach(ir, model, inputs) -> IR` |

每个模块单一职责、通过 IR 或明确签名交互，可独立测试。

## 7. 风险与实现期待确认项

| 风险 | 说明 | 缓解 |
|---|---|---|
| FakeTensor 能否完全免真实执行 | torchair 把 FakeTensor 图降到 GE 不真跑 kernel——实现期确认 | 兜底：触发一次真实 forward 编译（仍是一次性、非 profiling 循环） |
| GE 融合改名/合并致 join 不准 | profiler 的 kernel 名与 GE 节点名可能对不齐 | 名字 + 执行序 + shape 多键匹配 |
| 归一化映射表覆盖不全 | 新算子落 Unknown | 增量补表，不阻塞主链路 |
| 符号维度/动态 shape | 本次只做 specialize | IR 结构已预留，升级方案2 不推翻 |
| torchair 版本/GE dump 格式差异 | 不同 CANN/torch_npu 版本 dump 格式可能不同 | 解析层做版本适配，集中在 `ascend_extractor` |

## 8. 非目标（本次明确不做）

- 符号 shape / 动态 shape（方案2）——仅预留结构。
- GPU/Inductor 抽取器（P3）。
- roofline 性能上限评估器与跨设备对比（P4）。
- 训练（backward）算子序。
- TBE/embedding 的 FLOPs/访存代价模型（P4）。
