# Week 2: Attention end-to-end
*Focus*: Explain attention with tensor shapes on a whiteboard cold.  

**Resources**:  

- Raschka Ch. 3 — all of it
- "Attention Is All You Need" — read section 3 carefully, skim the rest
- Jay Alammar: The Illustrated Transformer — best intuition-builder for attention
- Karpathy: "Let's build GPT from scratch" — first ~1 hour covers attention

**Checklist**:

- Implement scaled dot-product attention from scratch (no nn.MultiheadAttention) ✅
- Implement multi-head attention from scratch ✅
- Add casual masking; verify with a test that future tokens can't influence past ones ✅
- Write annotated notebook: every tensor shape labeled at every step (B, T, C, n_head, head_dim) ✅
- Read "Attention Is All You Need" section 3; write a 1-page summary in your own words ✅
- Push notebook + summary to week02/ ✅

***Interview prep note***: The shapes notebook is your single most-referenced artifact for interviews. Make it good.  
*Skip*: Flash attention, GQA, MQA, rotary embeddings — note these exist, move on.
