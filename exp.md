# Experiment backlog

1. Single-expert models on each dataset: 10 × 4 = 40 runs.
2. Single-expert models on the full combined dataset: 1 × 4 = 4 runs.
3. Other OpenFWI baselines on single datasets (numbers from the official site).
4. Classical numerical FWI on single datasets.

5. Single experts with no encoder (direct resize) vs different encoders (ViT / ConvNeXt): 3 × 10 = 30 runs.
6. Encoder pretraining on the full dataset; encoder + FNO end-to-end with encoder-only checkpoint export: 1 configuration.
7. MoE on the full dataset:
   1. `velocity_type` routing mode.
   2. `group` routing mode:
      1. Plain router + Sum / Attention fusion.
      2. Plain router + strong/weak gating + Sum / Attention.
      3. Task-aware router + Sum ↔ plain router + Sum (ablation pair).
8. Marmousi inference vs numerical baselines.

Highlighting model strengths:

1. Strong per velocity-class performance.
2. Generalization to unseen geological layouts.
