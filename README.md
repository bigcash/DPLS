# DPLS: Dynamic Partial Label Smoothing Loss for Large Language Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)

Authors: [Xueming Hou]()\*

## 🔔 NEWS
- **[05/26/2026]** Our paper !

## Table of Contents

- [DPLS: Dynamic Partial Label Smoothing Loss for Large Language Models](#dpls)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
    - [Fineweb-Edu-100B](#fineweb-edu-100b)
  - [Pretraining](#pretraining)
  - [Evaluation](#evaluation)
  - [Acknowledgements](#acknowledgements)
  - [Star History](#star-history)
  - [Citation](#citation)


## Hardware Requirements
Pro6000 * 8 are recommended.

## Installation

Ensure you have Python 3.12 or higher installed. It's recommended to use a virtual environment to manage dependencies.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/bigcash/DPLS.git
   cd DPLS
   ```
2. **Create and Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   swanlab login
   ```

## Data Preparation

Prepare the necessary datasets before pretraining the model. Support [Fineweb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/)

### Fineweb-Edu-100B

Fineweb-Edu-100B is a large-scale educational dataset hosted on Hugging Face.

1. **Navigate to the Data Directory**

   ```bash
   cd data/fineweb-edu
   ```
2. **Run the Data Preparation Script**

   ```bash
   python fineweb-edu.py
   ```
3. **Move the Prepared Data**

   ```bash
   mv fineweb-edu100B ..
   cd ../..
   ```

## Pretraining

Pretrain the model using the prepared datasets. The provided scripts support distributed training across multiple GPUs.

1. **Baseline**

   For more control or customization, use `torchrun` to initiate training. Replace `config/train_llama_medium_adam_80g8.py` with your desired configuration file.

   ```bash
   torchrun --standalone --nproc_per_node=8  train_fw.py  config/train_llama_medium_adam_80g8.py
   ```

   - `--nproc_per_node=8` specifies the number of processes (typically matching the number of GPUs).

2. **DPLS**

   Update `dpls_epsilon` `dpls_top_k` `resume_dir` in file `config/train_llama_medium_adam_80g8_dpls.py`, model will resume, and use dpls loss start at step=60050

    ```bash
   torchrun --standalone --nproc_per_node=8 train_fw_pls.py config/train_llama_medium_adam_80g8_dpls.py
   ```

## Evaluation

### PPL & Entropy

   Update `resume_dir` in file `config/eval_llama_medium_adam_80g8.py`, model will resume from `resume_dir`

    ```bash
   python eval.py config/eval_llama_medium_adam_80g8.py
   ```

### Downstream Evaluation

Evaluate the performance of the pretrained model using standardized benchmarks.

1. **Navigate to the Evaluation Harness Directory**

   ```bash
   cd lm-evaluation-harness
   ```
2. **Follow the Instructions Within This Directory**

   *Ensure your model is compatible with the evaluation harness requirements.*

## Acknowledgements

- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT) provides the foundational codebase upon which this repo is built.
- [Hugging Face](https://huggingface.co/) for providing the [Fineweb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/) dataset.
- [EleutherAI](https://www.eleuther.ai/) for the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [tensorgi/TPA](https://github.com/tensorgi/TPA) provides the foundational codebase upon which this repo is built.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bigcash/DPLS&type=Date)](https://star-history.com/#bigcash/DPLS&Date)

## Citation

If you use DPLS in your research or application, please consider citing it!

```bibtex
@article{hou2026pls-loss,
    title={DPLS: Dynamic Partial Label Smoothing Loss for Large Language Models},
    author={Xueming Hou},
    journal={arXiv preprint arXiv:},
    year={2026},
}
```
