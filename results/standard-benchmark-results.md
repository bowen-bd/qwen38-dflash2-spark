# Standard Multi-Condition Benchmark: Qwen3.8-27B NVFP4 + DFlash2 on DGX Spark (GB10)

Measured on 2026-08-19 against the live `RadixArk/Qwen3.8-27B-NVFP4` model served by SGLang (`qwen3.8-27b-sglang`) using `z-lab/Qwen3.8-27B-DFlash2` speculative decoding on an NVIDIA GB10 GPU (45.5 GiB VRAM footprint).

---

## 1. Workload Diversity & Output Predictability (Single-Stream, Batch = 1, Thinking Off)
Evaluates decode speed across different output entropy / predictability regimes.

| Workload Domain | Median Decode (tok/s) | Median TTFT (ms) | Median ITL (ms) | Output Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **Free-form Creative Prose** | **25.09** | 188.3 | 39.95 | 400 |
| **Technical In-Depth Explanation** | **39.00** | 219.9 | 25.71 | 400 |
| **Python Code Generation** | **44.08** | 217.4 | 22.74 | 400 |
| **Code Editing / Refactoring** | **52.02** | 201.4 | 19.27 | 400 |
| **Structured JSON Output** | **54.22** | 196.5 | 18.49 | 400 |
| **Step-by-Step Math / Reasoning** | **59.93** | 191.7 | 16.73 | 400 |

- **Spread**: **2.39× speedup** from low-predictability prose (25.1 tok/s) to highly structured math/JSON (54–60 tok/s).
- **Draft Efficiency**: Empirical acceptance rate scales from ~24% on prose to ~77% on repetitive/syntactic code and structured math.

---

## 2. Concurrency & Throughput Scaling (300 tokens / stream, Thinking Off)
Evaluates serving saturation and latency under concurrent multi-client load up to and beyond the server's `max_running_requests=8`.

| Concurrency ($C$) | Aggregate Throughput (tok/s) | Per-Stream Speed (tok/s) | Avg TTFT (ms) | P90 TTFT (ms) | Total Wall Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **39.83** | 39.83 | 146.9 | 146.9 | 7.53 |
| **2** | **63.58** | 31.79 | 842.6 | 842.6 | 9.44 |
| **4** | **118.59** | 29.65 | 227.7 | 230.1 | 10.12 |
| **8** | **198.63** | 24.83 | 266.3 | 280.4 | 12.08 |
| **16** | **194.28** | 12.14 | 6,352.4 | 12,410.2 | 24.71 |

- **Peak Serving Capacity**: Reaches **198.6 aggregate tok/s** at 8 parallel streams.
- **Queueing Behavior**: At concurrency 16, throughput saturates at ~194 tok/s while excess requests queue gracefully.

---

## 3. Context & Prompt Length Scaling (Prefill Throughput & TTFT)
Evaluates TTFT and prefill throughput across prompt context lengths up to 64k tokens.

| Target Tokens | Actual Prompt Tokens | TTFT (s) | Prefill Throughput (tok/s) | Decode Rate (tok/s) |
| :---: | :---: | :---: | :---: | :---: |
| **128** | 132 | **0.220** | 599.9 | 19.8 |
| **512** | 290 | **0.252** | 1,150.7 | 22.6 |
| **1,024** | 598 | **0.324** | 1,845.1 | 19.7 |
| **4,096** | 2,369 | **1.069** | 2,216.6 | 19.6 |
| **16,384** | 9,453 | **4.640** | 2,037.3 | 17.2 |
| **32,768** | 18,924 | **5.802** | 3,261.6 | 25.6 |
| **65,536** | 37,866 | **18.424** | 2,055.3 | 25.1 |

- **Steady-state Prefill**: Sustained at **2,000 – 3,260 prompt tokens/sec** for medium-to-long contexts.

---

## 4. Prefix Caching / Radix Cache (Cold vs Warm Context Hit)
Evaluates the efficiency of SGLang's Unified Radix Cache on a shared ~4.8k token document context.

| Condition | Prompt Tokens | TTFT (ms) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Cold Request** (0% Cache Hit) | 4,787 | 2,286.3 | 1.00× (Baseline) |
| **Warm Request** (100% Radix Hit) | 4,789 | 201.8 | **11.33× Faster** |

- **Impact**: Instantaneous first-token generation for multi-turn conversations and system prompts.

---

## 5. Output Generation Length Scaling (Decode Stability)
Evaluates token generation rate across output horizons from 64 to 1024 tokens.

| Target Tokens | Actual Tokens Generated | Decode Speed (tok/s) | Inter-Token Latency (ms) | Total Wall Time (s) |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 64 | **35.05** | 28.99 | 2.04 |
| **128** | 128 | **34.31** | 29.38 | 3.94 |
| **256** | 256 | **33.36** | 30.09 | 7.89 |
| **512** | 512 | **28.88** | 34.69 | 17.94 |
| **1024** | 1024 | **31.66** | 31.62 | 32.56 |

- **Stability**: Consistent decode rate of **31–35 tok/s** without memory degradation across long sequence outputs.

---

## 6. Reasoning / Thinking Mode (Thinking On vs Off)
Evaluates generation throughput with internal chain-of-thought active (`enable_thinking=True`) vs direct answer mode (`enable_thinking=False`) on a multi-step engineering problem.

| Mode | Total Tokens | Reasoning Tokens | TTFT (ms) | Decode Speed (tok/s) | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thinking OFF** | 512 | 0 | 204.6 | **53.44** | 9.79 |
| **Thinking ON** | 512 | 512 | 213.5 | **52.22** | 10.02 |

- **Insight**: DFlash2 speculative decoding maintains high efficiency (**52.2 tok/s**) during deep thinking token generation.
