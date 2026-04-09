# Project SENTINEL
## Engineering Specification v2.0 — Elite Autonomous Vulnerability Auditor

> *A Gemma 4 31B Dense model, natively trained in NVFP4 on 8×NVIDIA B200 GPUs, engineered to autonomously discover, triage, and remediate software vulnerabilities at scale — with empirical safety alignment via automated DPO synthesis and dual-objective evaluation.*

---

## Executive Summary

Project SENTINEL constructs a **tier-one autonomous security asset** by combining three breakthroughs:

1. **Native 4-bit training** — leveraging NVIDIA's published NVFP4 methodology to train a 31B dense model in a fraction of the time and compute of standard FP8 pipelines
2. **Agentic vulnerability hunting** — the model does not passively read code; it actively navigates repositories using LSP tooling, terminal access, and iterative call-chain assembly
3. **Mathematically constrained alignment** — DPO training on automatically synthesized red-team data ensures the model operates exclusively as a network defender

The system deploys as a **heterogeneous dual-model architecture** governed by a deterministic state machine, with human defenders supervising execution telemetry in real-time.

---

## 1. Hardware & Compute Architecture

### 1.1 Infrastructure

| Component | Specification |
|---|---|
| **Node** | 1× DGX B200 (or equivalent) |
| **GPUs** | 8× NVIDIA B200 |
| **VRAM per GPU** | 192 GB HBM3e |
| **Aggregate VRAM** | 1,536 GB (~1.5 TB) |
| **Interconnect** | NVLink 5.0 (1.8 TB/s bidirectional) |
| **Base Model** | Gemma 4 31B Dense |
| **Training Precision** | NVFP4 (native 4-bit) |

### 1.2 Why Gemma 4 31B Dense

The dense architecture is a deliberate choice over Mixture-of-Experts (MoE):

- **Predictable fine-tuning** — no risk of "expert collapse" where specialized experts degrade during domain-specific SFT
- **Deterministic inference** — every token activates every parameter, eliminating routing variance that could cause inconsistent security assessments
- **Native reasoning** — Gemma 4's `<|think|>` block enables internal attack-graph mapping before output generation

### 1.3 NVFP4 Training Methodology

Native NVFP4 training is validated by NVIDIA's published research: a 12B model trained on 10T tokens achieved 62.58% MMLU-pro accuracy, matching the FP8 baseline (62.62%). Reference: arXiv:2509.25149

The training pipeline implements NVIDIA's four-component methodology:

1. **Random Hadamard Transforms (RHT)** — 16×16 matrices on Wgrad inputs. Redistributes block-level outliers into Gaussian distribution. Single fixed random sign vector across all layers.

2. **Two-Dimensional Block Scaling** — Weights: 16×16 2D blocks (forward/backward parity). Activations & Gradients: 1×16 1D blocks. Two-level: FP32 tensor-scale + E4M3 block-scale.

3. **Stochastic Rounding (Gradients Only)** — Eliminates quantization bias in backward pass. Round-to-nearest-even for weights and activations. Critical for convergence at multi-trillion tokens.

4. **Selective High-Precision Layers (15%)** — First 2 blocks + last 8 blocks remain in BF16. Final layers require more dynamic range than FP4. Remaining 85% of linear layers in NVFP4.

**Software:** NVIDIA Transformer Engine with native NVFP4 support on Blackwell.

### 1.4 Parallelism Strategy — ZeRO-2 Data Parallelism

The 31B model in NVFP4 is only ~17 GB. Tensor Parallelism across 8 GPUs would slice a 17 GB model into 2 GB shards — the GPUs would spend more time on NVLink activation transfers than on actual compute. Pure Data Parallelism (DDP) also fails because Adam optimizer states alone require ~372 GB, exceeding a single GPU's 192 GB capacity.

**Solution: ZeRO Stage 2** — shard optimizer states and gradients across GPUs while replicating model weights.

#### Per-GPU VRAM Budget (ZeRO-2)

| Component | Per-GPU Memory | Calculation |
|---|---|---|
| Model weights (NVFP4) | 17 GB | 31B × 0.5 bytes + scale overhead |
| Master weights (FP32, replicated) | — | Held by ZeRO shard |
| Optimizer shard (FP32 master + Adam m + Adam v) | 46.5 GB | (124 + 124 + 124) GB / 8 GPUs |
| Gradient shard (FP32) | 15.5 GB | 124 GB / 8 GPUs |
| **Fixed overhead** | **~79 GB** | |
| **Available for activations** | **~113 GB** | 192 - 79 |

Each GPU processes independent batches and synchronizes gradients over NVLink 5.0 (1.8 TB/s). This achieves near-linear 8× throughput scaling.

#### Optimizer Selection

| Optimizer | Per-GPU Fixed | Activation Budget | Status |
|---|---|---|---|
| **Adam** (β₁=0.9, β₂=0.95) | ~79 GB | ~113 GB | **Primary — validated with NVFP4** |
| **Muon** (no second moment) | ~63 GB | ~129 GB | Candidate — requires ablation study |

Muon eliminates Adam's second moment state, recovering ~16 GB/GPU. This enables ~14% larger batch sizes. However, Muon's orthogonalization step (Newton-Schulz iterations) creates different gradient distributions that are untested with NVFP4's stochastic rounding and RHT. A 1.2B-scale ablation (NVFP4 + Muon vs. NVFP4 + Adam, 100B tokens, ~2 days) must validate compatibility before committing at 31B scale.

---

## 2. Phase 1: High-Velocity Data Curation

**Timeline: Weeks 1–3 (21 days)**

### 2.1 The "Commit Delta" Strategy

We strictly avoid training on raw open-source code repositories. Empirical analysis of corpora like The Stack v2 reveals that ~58% of code blobs are unmaintained and contain thousands of known CVEs. Training on this data teaches the model to write vulnerable code, not to fix it.

**Core Principle:** Train exclusively on the mathematical delta between a vulnerable commit and its secure, patched counterpart.

**Source Repositories:**
- CVEFixes dataset (5,495 vulnerability-fixing commits)
- OSV.dev advisories → linked Git commits
- GitHub Security Advisories (GHSA)
- NVD → mapped to upstream patches

**Extraction Logic:** For each CVE:
1. Isolate the fixing commit SHA
2. Extract parent commit (vulnerable state)
3. Compute unified diff
4. Extract ±3 function-level context around changes
5. Map to CWE taxonomy (CWE-20, CWE-79, CWE-119...)
6. Generate structured training sample

**Quality filters:**
- Reject commits touching >20 files (noisy refactors)
- Reject diffs >4,096 tokens (out of scope)
- Require linked CVE or security advisory

**Target: 50,000–100,000 high-quality delta pairs**

### 2.2 Agentic Augmentation (Multi-Turn Synthesis)

Raw commit deltas teach patch mechanics. To teach investigative reasoning — the ability to navigate a codebase, trace data flows, and identify attack surfaces — we generate multi-turn conversational samples.

**Generation method:** Use an existing strong model (e.g., Gemini 2.5 Pro) to synthesize multi-turn traces against known-vulnerable open-source projects, then validate the identified vulnerabilities against ground-truth CVE databases.

**Target:** 20,000–30,000 multi-turn agentic traces.

### 2.3 Supply Chain Integrity

All ingested datasets are integrity-verified:

- **Cryptographic hashing** — SHA-256 checksums for every sample, stored in a tamper-evident Merkle tree
- **Provenance tracking** — each sample links to its source commit SHA, CVE ID, and advisory URL
- **Contamination scanning** — automated detection of known benchmark samples to prevent data leakage
- **Human audit sampling** — 5% random audit of training samples by security engineers

---

## 3. Phase 2: Supervised Fine-Tuning (SFT)

**Timeline: Weeks 4–6 (21 days)**

### 3.1 Phased Context Extension

At 256K context, activation memory dominates VRAM. The correct strategy is phased training: learn vulnerability mechanics at short context, then extend to full-codebase analysis at long context.

| Phase | Context Length | Batch Size (per GPU) | Duration | Purpose |
|---|---|---|---|---|
| **A: Core SFT** | 8,192 | 8–16 | 14 days | Learn commit-delta patch mechanics, CWE taxonomy, reasoning patterns |
| **B: Context Extension** | 32,768 | 2–4 | 5 days | Multi-file vulnerability tracing, cross-module data flow analysis |
| **C: Long-Range** | 131,072 | 1 | 2 days | Full repository-scale call-chain assembly, architecture-level threat modeling |

**Context extension method:** YaRN (Yet another RoPE extensioN) with NTK-aware interpolation.

### 3.2 Internal Reasoning Integration

The model is explicitly trained to use Gemma 4's native `<|think|>` block for internal deliberation:

1. Map the attack graph (sources → sinks → sanitization checkpoints)
2. Evaluate theoretical severity (CVSS 4.0 vector)
3. Formulate remediation strategy (minimal fix, defense-in-depth)

### 3.3 Training Configuration

- Model: gemma-4-31b-dense
- Precision: NVFP4
- Optimizer: Adam (or Muon, pending ablation)
- Learning rate: 2e-5 with cosine warmup schedule
- Weight decay: 0.1
- Effective batch size: 256 (8 GPUs × 8 batch × 4 gradient accumulation)
- Epochs: 3
- Framework: NVIDIA Transformer Engine + DeepSpeed ZeRO-2

---

## 4. Phase 3: Safety & Utility Alignment

**Timeline: Weeks 7–9 (21 days)**

### 4.1 Automated Red-Team DPO Synthesis Pipeline

Manual curation of chosen/rejected pairs for zero-day exploits is infeasible at scale. We implement a fully automated closed-loop pipeline inspired by DARPA AIxCC architectures.

**Pipeline:**
1. Feed Gemma a known vulnerable code snippet and ask it to write a patch
2. Compile the output in an isolated Docker sandbox
3. Run a fuzzer and unit tests against the compiled code
4. Label: fails to compile or fails security test → Rejected. Passes all → Chosen.
5. Feed into DPO training dataset continuously.

**Differential Fuzzing Enhancement:** Both original vulnerable code AND patched code are fuzzed. Valid "chosen" requires: crashes_on_original == True AND crashes_on_patch == False.

**Target:** 50,000+ automatically generated preference pairs.

### 4.2 Direct Preference Optimization (DPO)

| Parameter | Value | Rationale |
|---|---|---|
| β (temperature) | 0.1 | Standard for code generation tasks |
| Learning rate | 5e-7 | 40× lower than SFT to preserve learned capabilities |
| Epochs | 1 | Single pass to prevent overfitting on synthetic data |
| Reference model | Frozen SFT checkpoint | From Phase 2 |

### 4.3 The "PurpCode" CWE Rule Learning

Every model output grounds findings in CWE taxonomy with CVSS 4.0 scoring, specific file locations, vulnerable code snippets, and secure alternatives with explanations.

### 4.4 Over-Refusal Prevention (Dual-Objective Evaluation)

| Evaluator | Signal | Weight |
|---|---|---|
| **Safety Oracle** | Penalizes generation of weaponized exploits | 0.6 |
| **Utility Oracle** | Verifies the model can analyze real-world vulnerable code and produce working patches | 0.4 |

---

## 5. Phase 4: Production Deployment Architecture

**Timeline: Weeks 10–12 (21 days)**

### 5.1 The Agentic Retrieval Harness

The model does NOT receive entire codebases as raw text. It operates as an autonomous agent with tool access:

**Tools:**
- LSP Client (go-to-definition, find-references, hover-info, call-hierarchy, diagnostics)
- Terminal (grep, find, cat, git log, git diff)
- Semantic Search (embedding similarity, AST-aware chunking)
- Compiler (gcc/clang, rustc, python)
- Fuzzer (AFL++, libFuzzer, Jazzer)
- SAST Scanner (Semgrep, CodeQL, Bandit)

The context window is a working memory buffer (32K–128K active), not a data dump.

### 5.2 Heterogeneous "Alloy" — Dual-Model State Machine

Gemma 4 31B (Architect) handles strategic reasoning and vulnerability analysis. A fast code-centric model (DeepSeek-Coder or Codestral) acts as the Executor. They are governed by a deterministic state machine:

1. **Intake** → Target repository received
2. **ArchitectReasoning** → Gemma issues tool calls, explores codebase
3. **VulnerabilityReport** → Structured JSON report (CWE, CVSS, location)
4. **PatchGeneration** → API Bridge formats prompt for Executor
5. **ExecutorCodeGen** → Code-centric model generates patch
6. **SandboxValidation** → Compile + test + fuzz
7. **ErrorFeedback** (on failure) → stderr routed back to Architect
8. **HumanReview** → HOTL approval gate
9. **AutoRemediate** → Patch deployed

### 5.3 Human-On-The-Loop (HOTL) Integration

- **Tier 1 (Full Autonomy):** Read-only scanning, report generation, sandbox execution
- **Tier 2 (Auto-Execute + Notify):** Endpoint isolation, WAF rule injection, cert rotation
- **Tier 3 (Human Approval Required):** Production code deployment, infra changes, access control

---

## 6. Evaluation & Benchmarks

| Benchmark | Target Score |
|---|---|
| CyberSecEval 3 (Meta) | >75% |
| CVEFixes Test Split | >80% fix rate |
| SWE-bench Verified | >45% |
| PurpleLlama Insecure Code | <5% generation rate |
| Custom Red-Team Suite | <2% jailbreak rate |
| MITRE ATT&CK Coverage | >60% of techniques recognized |

---

## 7. Project Timeline

**Total: 12 weeks (84 days) from data curation to deployment-ready.**

Compressible to ~10 weeks with aggressive parallelization.

---

## 8. Risk Registry

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| NVFP4 training diverges at 31B | HIGH | LOW | Validated at 12B. Fallback: BF16 for final 20% |
| Insufficient training data | HIGH | MEDIUM | Supplement with synthetic vulnerability injection |
| Muon incompatible with NVFP4 | MEDIUM | MEDIUM | Ablation study. Fallback: Adam |
| Model over-refuses | HIGH | MEDIUM | Dual-objective evaluation |
| Agentic harness loops | MEDIUM | LOW | 30-min timeout, 200 tool call limit |
| Prompt injection bypass | CRITICAL | LOW | Multi-layer defense: input filtering + output scanning + behavioral monitoring |

---

*Project SENTINEL — v2.0 — April 2026*
