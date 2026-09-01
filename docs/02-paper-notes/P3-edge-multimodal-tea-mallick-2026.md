# P3 — Edge-optimised multimodal tea framework (Mallick, Saha, Chakrabarti, 2026)

## Bibliographic entry
Mallick, M.T., Saha, H.N., & Chakrabarti, A. (2026). **Transformative Energy-Efficient Edge-Optimised Multimodal Deep Learning Framework for Pest Management and Severity Analysis in Tea Plants.** *ACM Transactions on Embedded Computing Systems*, **25(4)**, Article 63 (June 2026), 36 pp. DOI: [10.1145/3816754](https://doi.org/10.1145/3816754). CC BY 4.0.

## Abstract (verbatim, excerpt)
> Recurrent infestations reduce yield and quality in Indian *tea* cultivation. Early visual symptoms under field conditions are often obscured by low light, high humidity, and leaf occlusion, limiting the reliability of conventional vision-based classifiers to controlled settings. … The **multimodal, edge-optimised framework** integrates RGB pest imagery with real-time environmental data for **robust on-site monitoring**. A **hybrid CNN–Transformer backbone** captures fine pest-specific textures and long-range contextual cues, while an **attention-driven fusion layer** adaptively incorporates temperature, humidity, and illumination signals. A key contribution is the **custom dual-mode edge hardware**, integrating dedicated CNN and Transformer accelerators with sensor co-processors. Hierarchical on-chip SRAM buffers reduce memory energy by **200×**, supported by energy-aware scheduling. … The dataset comprises **1,520 field-collected pest images**, augmented to **7,600 samples across five classes**. Environmental conditioning preserves **85.5% accuracy under combined perturbations**, whereas unimodal inference degrades to 77.9% (~15% error reduction). Hardware–algorithm co-design confines inference latency to **25 ms** with **0.12 J** energy per inference. Energy consumption remains **60% lower than Jetson Nano** deployment.

## Key contributions
1. **Multimodal Pest Detection Framework** — dual-stream attention-based fusion of visual pest imagery and environmental signals (temperature, humidity, illumination), robust across heterogeneous field conditions.
2. **Hybrid CNN–Transformer Backbone** — CNN captures local features, Transformer captures long-range context, jointly lightweight.
3. **Custom Dual-Mode Edge Hardware** — dedicated CNN and Transformer accelerators + sensor co-processors, offline solar-powered inference for connectivity-limited plantations, cloud-assisted synchronisation for periodic model updates.
4. **Hardware–Algorithm Co-Optimisation** — structured pruning, quantisation, knowledge distillation on the edge pipeline; 25 ms / 0.12 J per inference.

## Method
Two modalities: RGB leaf images (CNN branch) + environmental sensor stream (temp/humidity/illumination, sensor-coprocessor branch). Both feed an attention-driven fusion layer sitting on top of the Transformer half of the hybrid backbone. The whole thing is trained with hardware-aware pruning/quantisation/distillation, then deployed on the custom dual-mode edge platform with hierarchical SRAM buffering. Offline solar operation is default; cloud sync is periodic.

Dataset: 1,520 real tea-plantation images across 5 classes (aphids, red spider mites, tea mosquito bugs, looper caterpillars, thrips), augmented to 7,600.

## Results
- **85.5%** accuracy under combined field perturbations (vs. 77.9% unimodal → ~15% relative error reduction).
- **25 ms** inference latency, **0.12 J/inference**.
- **60% lower energy** than a Jetson Nano deployment.
- **200× lower memory energy** via hierarchical on-chip SRAM.

## My take
P3 is the group's **system paper**: it commits to a specific deployment target (tea plantation, solar edge, connectivity-limited) and shows the full co-design. Two things stand out for MTP-2:
1. The hybrid CNN–Transformer backbone is exactly the place P4's bounded-rank attention should be dropped in — the paper explicitly lists Transformer accelerators as a hardware line item, so making the Transformer cheaper on that hardware is directly valuable.
2. The multimodal fusion layer is attention-based. Its rank is unconstrained today; the same bounded-rank primitive could regularise it.

## Open questions
- What is the actual FLOP / memory profile of the Transformer branch on the custom accelerator? If it dominates, P4 has a straightforward win. If it doesn't, the interesting substitution is elsewhere.
- Cloud sync is periodic — can federated learning (P5) replace it entirely?
