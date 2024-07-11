<div align="center">
<h1>OV-DINO </h1>
<h3>Unified Open-Vocabulary Detection with Language-Aware Selective Fusion</h3>

[Hao Wang](https://github.com/wanghao9610)<sup>1,2</sup>,[Pengzhen Ren](https://scholar.google.com/citations?user=yVxSn70AAAAJ&hl)<sup>1</sup>,[Zequn Jie](https://scholar.google.com/citations?user=4sKGNB0AAAAJ&hl)<sup>3</sup>, [Xiao Dong](https://scholar.google.com.sg/citations?user=jXLkbw8AAAAJ&hl)<sup>1</sup>, [Chengjian Feng](https://fcjian.github.io/)<sup>3</sup>, [Yinlong Qian](https://scholar.google.com/citations?user=8tPN5CAAAAAJ&hl)<sup>3</sup>,

[Lin Ma](https://forestlinma.com/)<sup>3</sup>, [Dongmei Jiang](https://scholar.google.com/citations?user=Awsue7sAAAAJ&hl)<sup>3</sup>, [Yaowei Wang](https://scholar.google.com/citations?user=o_DllmIAAAAJ&hl)<sup>3,4</sup>, [Xiangyuan Lan](https://scholar.google.com/citations?user=c3iwWRcAAAAJ&hl)<sup>3</sup><sup>:email:</sup>, [Xiaodan Liang](https://scholar.google.com/citations?user=voxznZAAAAAJ&hl)<sup>1,2</sup><sup>:email:</sup>

<sup>1</sup> Sun Yat-sen University, <sup>2</sup> Pengcheng Lab, <sup>3</sup> Meituan Inc, <sup>4</sup> HIT, Shenzhen

[[`Paper`](https://arxiv.org/abs/2407.07844)] [[`HuggingFace`]()] [[`BibTex`](#citation)]

</div>

## Updates

- **`11/07/2024`**: We release OV-DINO paper on arxiv. Code and pre-trained model are coming soon.

## Introduction
This project contains the official PyTorch implementation, pre-trained models, fine-tuning code, and inference demo for OV-DINO.

* OV-DINO is a novel unified open vocabulary detection approach that offers superior performance and effectiveness for practical real-world application.

* OV-DINO entails a Unified Data Integration pipeline that integrates diverse data sources for end-to-end pre-training, and a Language-Aware Selective Fusion module to improve the vision-language understanding of the model.

* OV-DINO shows significant performance improvement on COCO and LVIS benchmarks compared to previous methods, achieving relative improvements of +2.5\% AP on COCO and +13.6\% AP on LVIS compared to G-DINO in zero-shot evaluation.

## Overview

<img src="docs/ovdino_framework.png" width="800">

## TODO
- [ ] Release the pre-trained model.
- [ ] Release the evaluation and fine-tuning code.
- [ ] Support the local inference demo.
- [ ] Support the web demo.
- [ ] Release the pre-training code.

## Citation
If you find OV-DINO is helpful for your research or applications, please consider giving us a star 🌟 and citing it by the following BibTex entry.

```bibtex
@article{wang2024ovdino,
  title={OV-DINO: Unified Open-Vocabulary Detection with Language-Aware Selective Fusion}, 
  author={Hao Wang and Pengzhen Ren and Zequn Jie and Xiao Dong and Chengjian Feng and Yinlong Qian and Lin Ma and Dongmei Jiang and Yaowei Wang and Xiangyuan Lan and Xiaodan Liang},
  journal={arXiv preprint arXiv:2407.07844},
  year={2024}
}
```