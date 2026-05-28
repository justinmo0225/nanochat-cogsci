# Week 3: GPT from scratch

*Focus*: Own a working GPT-2 implementation.

**Resources:**

- Raschka Ch. 4
- Karpathy: "Let's build GPT from scratch" — full video
- Hugging Face GPT-2 model card — for weight-loading reference
- LayerNorm paper — skim, just understand what it does

**Checklist:**

- Implement full GPT block: attention + MLP + residual + LayerNorm (pre-norm variant) ✅
- Stack blocks into full GPT-2 small (124M params); match HF architecture exactly ✅
- Load GPT-2 pretrained weights from Hugging Face into your implementation ✅
- Verify: same prompt → same output (within numerical tolerance) as HF reference ✅
- Generate text from your loaded model; sanity-check it makes sense ✅
- README with architecture diagram and matching-output proof

*Skip*: Training GPT-2 from scratch this week — that's week 4.

---
**Matching-output proof:**

```bash
124.439808 M parameters
weights loaded successfully
The meaning of life is not the same as the meaning of death.
The meaning of life is not the same as the meaning of death.
The meaning of life is not the same as the meaning of death.
The meaning of life is not the
weights loaded successfully
max difference: 0.0001068115234375
match: True
```

![GPT-2 Architecture](architecture-diagram/(1).png)