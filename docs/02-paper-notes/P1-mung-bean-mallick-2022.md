# P1 — Mung bean CNN + Android (Mallick et al., 2022)

## Bibliographic entry
Mallick, M.T., Biswas, S., Das, A.K., Saha, H.N., Chakrabarti, A., & Deb, N. (2022). **Deep learning based automated disease detection and pest classification in Indian *mung bean*.** *Multimedia Tools and Applications.* DOI: [10.1007/s11042-022-13673-7](https://doi.org/10.1007/s11042-022-13673-7). Published 17 Sep 2022.

## Abstract (verbatim)
> Crop pests and diseases are major threats to food security globally. The *mung bean* (Vigna Radiata) is one of the leading crops in India. A large part of the population in India is completely dependent on *mung bean*. So, high production efficiency for the *mung bean* is required, which does not happen due to the excessive damage from pests and diseases. Recently, with the advancement of Deep Learning techniques, remarkable performance has been achieved in the field of image classification by employing Convolutional Neural Networks (CNNs). This brings a lot of promise in the field of pest and disease identification by effective image classification. In this paper, we have proposed a novel deep learning-based technique to identify the *mung bean* pest and disease. In order to handle the limitation arising due to less number of *mung bean* crop images for the purpose of training, we have adopted transfer learning, which is able to generate a very promising result for quick and easy pest and disease detection. The developed model has successfully recognized 6 different types of *mung bean* diseases and 4 types of pests out of healthy and affected leaves collected in different seasons. Based on the experiments conducted, the proposed smartphone-based deep learning model for the *mung bean* pest and disease detection has achieved an average accuracy of **93.65%**.

## Key contributions
- Smartphone-native Android app that runs a lightweight CNN offline (no internet needed).
- Successful transfer-learning strategy from ImageNet-pretrained backbones onto a small, seasonally heterogeneous *mung bean* leaf dataset.
- Data-augmentation pipeline to compensate for the training-image scarcity.
- 10-way classification (6 diseases + 4 pest categories) with 93.65% average accuracy.

## Method
Transfer learning on a compact CNN backbone (details in paper). The model is fine-tuned on the *mung bean* leaf dataset, converted to a smartphone-runnable format, and deployed inside an Android app for on-field farmers. The pipeline explicitly avoids server-side inference so it works in low-connectivity rural settings.

## Results
- **Average accuracy: 93.65%** across the 10 classes.
- Practical smartphone deployment demonstrated.
- (See paper for per-class precision/recall.)

## My take
P1 is the group's "toehold" paper: crop + transfer-learning + smartphone. It establishes the **application scaffold** (Indian legume crops, offline mobile inference) that P2, P3, and P5 later scale up to hardware, multimodal, and federated versions. For MTP-2, its dataset-scarcity workaround (aggressive augmentation + transfer learning) is directly reusable if we do a small tea-pest ablation before scaling to full data.

## Open questions
- How brittle is the smartphone model to natural distribution shift (new season, new region)? P1 collects "different seasons" but does not explicitly report drift.
- What is the actual latency and energy budget on-device? P2 later gives a hardware-accelerated answer, but P1 does not.
