# SAEP-SVD

**Sign Alignment and Energy Preservation for Stable Distillation of Extreme-Rank LoRA Adapters in Diffusion Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

High-rank LoRA adapters ($r = 512–2048$) trained on large diffusion models (Z-Image, Flux.2) suffer from severe degradation after standard Eckart–Young SVD distillation. The resulting adapters only remain stable in an extremely narrow weight range **$w \in [0.01, 0.05]$**. Higher weights cause immediate diffusion collapse, high-frequency noise, and geometric breakdown.

We identify two root causes:
1. **Arbitrary sign flips** of singular vectors $\rightarrow$ phase misalignment across layers.
2. **Substantial loss of Frobenius energy** during rank truncation.

**SAEP-SVD** introduces two fundamental corrections — **Sign Alignment** and **Energy Preservation** - restoring a practical operating weight range of approximately **$w \in [0.05, 1.20]$** while preserving fine artistic style and structural details.

---

## The Problem

After classic Eckart–Young SVD distillation of extreme-rank LoRAs:

| Weight Range | Result |
|---|---|
| `0.01 – 0.05` | Stable generation (weak effect) |
| `≥ 0.10` | Collapse, noise, anatomical destruction |

Negative weights can partially work around the issue, but they destroy meaningful strength control. It becomes impossible to push an artistic style toward full expression ($w \approx 0.8–1.0$) while keeping the original lighting and atmosphere.

---

## Method: SAEP-SVD

1. **Reconstruct full operator:**
   $$\hat{W} = B_{\text{orig}} \cdot A_{\text{orig}}$$

2. **Compute SVD:**
   $$\hat{W} = U \Sigma V^T$$

3. **Sign Alignment:**  
   Force canonical phase orientation of singular vectors to prevent inter-layer phase inversion.

4. **Energy Preservation:**  
   Restore Frobenius norm via scaling factor $\gamma$:
   $$\gamma = \frac{\Vert{}\hat{W}\Vert{}_F}{\Vert{}\hat{W}_k\Vert{}_F}$$
   Scale retained singular values: $\Sigma_k \leftarrow \gamma \cdot \Sigma_k$

5. **Symmetric Factorization:**  
   Split back into target matrices $A_{\text{new}}$ and $B_{\text{new}}$.

### Architectural Notes

- **Conv2d:** Works out of the box. 4D kernels are flattened to 2D ($[O, I \times H \times W]$). Spatial energy is redistributed across channels without phase skew.
- **Mixture-of-Experts (MoE):** Compute $\gamma$ independently per expert ($\gamma_k$) to avoid saturating individual routing paths.

---

## Results

Tested on:
- **Z-Image**
- **Flux.2 (28B)** — Van Gogh + Aivazovsky merge, distilled to rank 256

| Method       | Stable Weight Range     | Style & Detail Preservation      |
|--------------|-------------------------|----------------------------------|
| Classic SVD  | [0.01, 0.05]            | Poor at usable strengths         |
| **SAEP-SVD** | **[0.05, 1.00]** (Z-Image) | **High, approximately linear** |

> **Architectural Findings:**
> - **Z-Image:** SAEP-SVD shows a clear benefit, significantly expanding the usable weight range up to ~1.0.
> - **Flux.2 (28B):** The improvement is marginal. Previously the maximum stable weight was ≈ 0.05–0.07. With SAEP-SVD it is possible to go slightly higher (up to ≈ 0.15), but the adapter still operates best in the low-weight regime. Flux.2 remains largely resistant to expanding the operational window after high-rank distillation.

---

## Usage

```bash
python saep_svd.py --input my_high_rank_lora.safetensors --output distilled_rank256.safetensors --rank 256
Or edit parameters inside saep_svd.py:

Python
INPUT_LORA  = "your_high_rank_lora.safetensors"
OUTPUT_LORA = "distilled_saep_rank256.safetensors"
TARGET_RANK = 256
Citation
If you use SAEP-SVD in your research or workflows, please cite:


@software{orakul_saep_svd_2026,
  author       = {Orakul Studio},
  title        = {SAEP-SVD: Sign Alignment and Energy Preservation for Stable Distillation of Extreme-Rank LoRA Adapters},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
