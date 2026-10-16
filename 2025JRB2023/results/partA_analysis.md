# Part A Report-Ready Analysis

## Final sweep
A1–A6 were trained on the same supplied 20,000 training and 2,000 validation examples with seed 42,
2 epochs, LR 2e-4, cosine schedule, LoRA dropout 0.05, max length 768, fp16, and global effective batch 16.

## Main findings
- Highest validation entity F1: A6 = 0.95257170
- Lowest validation leak rate: A3 = 0.03675467
- Recommended Part-B checkpoint: A3, because it has the lowest leakage with strong F1 and far fewer trainable
  parameters than A4/A5/A6.
- Placement gains:
  - A1→A2 F1 gain = 0.01628770
  - A2→A3 F1 gain = 0.01494675
- Rank curve flattening:
  - A3→A4 F1 gain = 0.00483361
  - trainable parameters increase from 9.23M to 36.93M.
- Alpha effect at r=32:
  - A4 (alpha=64) beats A5 (alpha=16) by 0.00222941 F1.
- QLoRA (A6) versus matched LoRA (A4):
  - F1 change = +0.00165613
  - peak GPU-memory saving = 0.762869 GB (9.29%)
  - leak-rate change = +0.00072225
  - QLoRA did not reduce F1 in this sweep, though it took longer wall-clock time.

## Caveat
For A6, software representation of quantized parameters changes the reported total parameter count and therefore
the displayed trainable percentage. Use the measured peak GPU memory for the most meaningful memory comparison.
