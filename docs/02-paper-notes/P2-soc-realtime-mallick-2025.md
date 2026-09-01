# P2 — SoC / DPU real-time detection (Mallick et al., 2025)

## Bibliographic entry
Mallick, M.T., Murty, D.O., Pal, R., Mandal, S., Saha, H.N., & Chakrabarti, A. (2025). **High-speed system-on-chip based platform for real-time crop disease and pest detection using deep learning techniques.** *Computers and Electrical Engineering* **123**: 110182. DOI: [10.1016/j.compeleceng.2025.110182](https://doi.org/10.1016/j.compeleceng.2025.110182). Received 16 May 2024; accepted 9 Feb 2025.

## Abstract (verbatim, excerpt)
> Crop diseases significantly threaten global agricultural productivity and food security … This research tackles disease classification in *mustard* and *mung bean* crops by employing transfer learning, a **MobileNetV3-based CNN model**, and a **System-on-Chip (SoC) computing platform**. … Xilinx **Deep Learning Processor Unit (DPU)** intellectual property (IP) accelerates disease classification **24 times** compared to software counterparts. … our proposed design enhances the throughput by around **29%** and reduces the power consumption by around **19%**. MobileNetV3 achieves classification accuracies of **96.14% on *mung bean* and 93.25% on *mustard*** datasets, surpassing other state-of-the-art methods.

## Key contributions
- Full CNN → SoC pipeline for real-time crop disease detection on Xilinx hardware.
- 24× inference speed-up vs. pure software baseline via DPU IP.
- +29% throughput, −19% power vs. previous SoC deployments.
- Mobile Android front-end that captures images and communicates with the SoC.
- Extends beyond *mung bean* to *mustard* without hardware modification (P1 → P2 lift).

## Method
Backbone: MobileNetV3 fine-tuned on *mung bean* and *mustard* datasets (transfer learning). Model exported as `.h5`, converted to the Xilinx DPU quantised representation, deployed on an SoC with a Deep Learning Processing Unit accelerator. Android app captures a leaf image, ships it to the SoC over Wi-Fi/Ethernet, receives the class prediction. Includes a data-augmentation stage (TF `ImageDataGenerator` + Adobe Photoshop augmentations) and a hazy-image de-noising step in the Android front-end (P2 improvement over P1).

## Results
- Mung bean: **96.14%** classification accuracy.
- Mustard: **93.25%** classification accuracy.
- 24× speed-up over CPU-only inference, +29% throughput, −19% power.
- Hardware not paper-specialised — same platform reused across crops.

## My take
P2 is the group's **acceleration** paper: it does not change the model class relative to P1 (still CNN + transfer learning), but it moves the deployment target from a smartphone to a real SoC + DPU, and it introduces the co-design mindset (model + hardware + augmentation + de-noising as one system). For MTP-2, P2 is the direct systems benchmark to beat if I claim a bounded-rank Transformer is cheaper than a MobileNetV3-CNN on the same SoC.

## Open questions
- P2 stays with a CNN. The moment a transformer enters the pipeline (P3, P4), the `O(N²d)` attention becomes the SoC bottleneck — is that where P4's bounded-rank attention pays off? MTP-2 should measure this on the same DPU.
- P2 uses hazy-image de-noising in the Android side. Could de-noising be replaced by a rank-controlled attention that is inherently robust to noise? (P4 reports a comparable 2.9% vs. 3.1% accuracy loss under noise — suggests yes.)
