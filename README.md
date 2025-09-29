# Decoding Uncertainty: The Impact of Decoding Strategies for Uncertainty Estimation in Large Language Models

This repository includes our code used in the following paper ([arXiv](https://arxiv.org/abs/2509.16696)) accepted at EMNLP 2025 Findings:

```
@misc{hashimoto2025decodinguncertaintyimpactdecoding,
    title={Decoding Uncertainty: The Impact of Decoding Strategies for Uncertainty Estimation in Large Language Models}, 
    author={Wataru Hashimoto and Hidetaka Kamigaito and Taro Watanabe},
    year={2025},
    eprint={2509.16696},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.16696}, 
}
```

We've modified the following codes: 
- [LM-Polygraph: Uncertainty Estimation for Language Models](https://aclanthology.org/2023.emnlp-demo.41/) (Fedeeva et al., 2023)
  - Code: https://github.com/IINemo/lm-polygraph 
- [SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Model](https://openreview.net/forum?id=t7wvJstsiV) (Zhang et al., 2024)
  - Code: https://github.com/JayZhang42/SLED
- [A Thorough Examination of Decoding Methods in the Era of LLMs](https://aclanthology.org/2024.emnlp-main.489/) (Shi et al., 2024)
  - Code: https://aclanthology.org/attachments/2024.emnlp-main.489.software.zip 

## Introduction

Decoding strategies manipulate the probability distribution underlying the output of a language model and can therefore affect both generation quality and its uncertainty. In this study, we investigate the impact of decoding strategies on uncertainty estimation in Large Language Models (LLMs). Our experiments show that Contrastive Search, which mitigates repetition, yields better uncertainty estimates on average across a range of preference-aligned LLMs. In contrast, the benefits of these strategies sometimes diverge when the model is only post-trained with supervised fine-tuning, i.e. without explicit alignment.


## Setup

我々はSingularity環境におけるSlurmを用いてSlurmでバッチジョブを投入することで実験結果を得た．もしこれらを使わない場合は、適宜 `.def` ファイルに記述されているライブラリをあなたの環境にインストールしてください．

```bash
module load singularity
singularity build --fakeroot llm-uncertainty.sif llm-uncertainty.def
```


## Run

データセットやデコーディング戦略のオプション一覧については、`batch_run.sh` 内に記述されている．


```bash
sbatch batch_run.sh
```

