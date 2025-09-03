<div align="center">

<img src="assets/logo.png" width="200"/>

# Face-MoGLE

Mixture of Global and Local Experts with Diffusion Transformer for Controllable Face Generation

[![arXiv Paper](https://img.shields.io/badge/arXiv-2509.00428-B31B1B)](https://arxiv.org/abs/2509.00428)
[![Project Page](https://img.shields.io/badge/Project%20Page-Face--MoGLE-blue)](https://xavierjiezou.github.io/Face-MoGLE/)
[![HugginngFace Models](https://img.shields.io/badge/🤗HugginngFace-Models-orange)](https://huggingface.co/XavierJiezou/face-mogle-models)
[![HugginngFace Datasets](https://img.shields.io/badge/🤗HugginngFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/face-mogle-datasets)

![teaser](assets/framework.svg)

</div>

<!--This repository serves as the official implementation of the paper **"Mixture of Global and Local Experts with Diffusion Transformer for Controllable Face Generation"**. It provides a comprehensive pipeline for semantic segmentation, including data preprocessing, model training, evaluation, and deployment, specifically tailored for cloud segmentation tasks in remote sensing imagery.-->

---

<!-- ## Features

**Face-MoGLE** is a novel and flexible framework for high-quality, controllable face generation, built on top of Diffusion Transformer (DiT) models.

* **Global-Local Expert Mixture 🧠**: Incorporates a Mixture of Experts (MoE) with both global and local experts to simultaneously capture holistic facial structure and fine-grained regional details.

* **Decoupled Semantic Control 🎛️**: Semantic masks are disentangled from the diffusion process, enabling fine-grained control over facial attributes without sacrificing image quality.

* **Dynamic Expert Routing 🔄**: A spatially-aware gating network dynamically adjusts expert contributions during the diffusion process, allowing adaptive semantic alignment and visual fidelity.

* **Photorealism with Robustness 📸**: Generated faces are highly photorealistic and can fool state-of-the-art face forgery detectors, making the method valuable for both creative and security-related applications.

* **Multimodal & Zero-Shot Generalization 🌈**: Works effectively under both multimodal and monomodal conditions with strong zero-shot generalization across diverse face generation tasks.

---

## News

- **2025-04-11**: 🧪 Inference and demo scripts released! Try Face-MoGLE with your own semantic masks and test its control capability across different face attributes. 

<img src='assets/framework.svg' width='100%' /> -->

## Installation

```bash
conda create -n face-mogle python=3.11.11
conda activate face-mogle
pip install -r requirements.txt
```

### 📥 Download Checkpoints

Before running the inference or test, please download the following files:

- **Pretrain**  
  - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

- **SFT**  
  - [pytorch_lora_weights.safetensors](https://huggingface.co/XavierJiezou/face-mogle-models/resolve/main/pytorch_lora_weights.safetensors)  
  - [global_local_mask_moe.pt](https://huggingface.co/XavierJiezou/face-mogle-models/resolve/main/global_local_mask_moe.pt)  

### 📂 Directory Setup

After downloading, please place the files in the following structure:

```bash
Face-MoGLE
├── ...
├── checkpoints
│   ├── FLUX.1-dev
├── runs
│   ├── face-mogle
│   │   ├── pytorch_lora_weights.safetensors
│   │   ├── global_local_mask_moe.pt
│   │   ├── config.yaml
````

## 🖼️ Inference

Generate images with semantic mask + text:

```bash
python inference.py \
    --prompt "She is wearing lipstick. She is attractive and has straight hair." \
    --mask "data/mmcelebahq/mask/27000.png" \
    --output_dir output
```


## 🌐 Gradio Demo (Web UI)

You can also launch an interactive demo using **Gradio**:

```bash
python gradio_app.py
````

🎥 Demo

<video src="https://github.com/user-attachments/assets/fa2ba2e0-03d5-4d61-887c-53cdce0ccdf7" controls width="100%" playsinline preload="metadata"></video>

### 📂 Directory Setup

Make sure the pretrained backbone and model weights are placed in the following structure before running the demo:

```bash
Face-MoGLE
├── ...
├── checkpoints
│   ├── FLUX.1-dev
├── runs
│   ├── face-mogle
│   │   ├── pytorch_lora_weights.safetensors
│   │   ├── global_local_mask_moe.pt
│   │   ├── config.yaml
```

### ⚡ GPU Control (Optional)

If you want to specify which GPU to use, set the `CUDA_VISIBLE_DEVICES` environment variable before launching the demo.
For example, to use **GPU 1**:

```bash
export CUDA_VISIBLE_DEVICES=1
python gradio_app.py
```



## Prepare Data


### 📥 Download Datasets

You can download the datasets from Hugging Face:

| Dataset Name            | Download Link                                                                                                                | Usage                               |
|:------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|
| **MM-CelebA-HQ**   | [Hugging Face](https://huggingface.co/datasets/XavierJiezou/face-mogle-datasets/resolve/main/mmcelebahq.zip) <br> (Also available in [TediGAN](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)) | Training & Evaluation                |
| **MM-FairFace-HQ** | [Hugging Face](https://huggingface.co/datasets/XavierJiezou/face-mogle-datasets/resolve/main/mmfairfacehq.zip)                    | Just for Zero-shot Generalization Validation |
| **MM-FFHQ-Female** | [Hugging Face](https://huggingface.co/datasets/XavierJiezou/face-mogle-datasets/resolve/main/mmffhqfemale.zip)                    | Just for Zero-shot Generalization Validation |

**Note:**  
> The **MM-FairFace-HQ** and **MM-FFHQ-Female** datasets are multimodal extensions we constructed based on the original face image datasets, using a semi-automated annotation approach.

### 📂 Dataset Structure

After extraction, please organize the directory as follows:

```bash
Face-MoGLE
├── ...
├── data
│   ├── mmcelebahq
│   │   ├── face
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   ├── mask
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   ├── text
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   ├── text.json
│   ├── mmffhqfemale
│   │   ├── face
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   ├── mask
│   │   │   ├── 00001.png
│   │   │   ├── 00002.png
│   │   ├── text
│   │   │   ├── 00001.txt
│   │   │   ├── 00002.txt
│   │   ├── text.json
│   ├── mmfairfacehq
│   │   ├── face
│   │   │   ├── 52.jpg
│   │   │   ├── 55.jpg
│   │   ├── mask
│   │   │   ├── 52.png
│   │   │   ├── 55.png
│   │   ├── text
│   │   │   ├── 52.txt
│   │   │   ├── 55.txt
```




<br>

## 🚀 Training

```bash
bash script/train_face-mogle.sh
```

## 🧪 Testing

```bash
python test.py \
  --root data/mmcelebahq \
  --lora_ckpt runs/face-mogle/pytorch_lora_weights.safetensors \
  --moe_ckpt runs/face-mogle/global_local_mask_moe.pt \
  --pretrained_ckpt checkpoints/FLUX.1-dev \
  --config_path runs/face-mogle/config.yaml \
  --output_dir visualization/face-mogle
```

## 📊 Evaluation

Face-MoGLE is evaluated across multiple dimensions, including visual fidelity, semantic alignment, and structural consistency.

### 🎯 FID / KID / Text Consistency  
Measure generation quality and text-image alignment.

- 📌 **[FID & KID](https://github.com/GaParmar/clean-fid)**
- ✍️ **[Text Consistency](https://github.com/Taited/clip-score)**

```bash
python src/eval/eval_fid_kid_text.py \
    --fake_image visulization/face-mogle/face \
    --real_face_dir visulization/mmcelebahq/face \
    --real_text_dir visulization/mmcelebahq/text \
    --output_dir eval_result
```

### 🧠 CLIP Maximum Mean Discrepancy (CMMD)  

- 🤖 **[CMMD](https://github.com/sayakpaul/cmmd-pytorch)**

```bash
cd cmmd-pytorch
python main.py gt_dir pred_dir
```

### 🎭 Mask Consistency (DINO Structure Distance)  

- 🧩 **[Mask Consistency](https://github.com/omerbt/Splice)**

```bash
python src/eval/eval_mask.py \
    --real_dir visulization/mmcelebahq/face \
    --fake_img visulization/face-mogle/face \
    --output_dir eval_result
```

### 👤 Human Perference (ImageReward)  

- 🌟 **[ImageReward (IR)](https://github.com/THUDM/ImageReward)**

```bash
python src/eval/eval_ir.py \
    --image_path visulization/face-mogle/face \
    --text_path visulization/mmcelebahq/text \
    --output_dir eval_result
```


## Visualization

### Monomodal Generation

<table>
  <tr>
    <td align="center"><b>Mask-to-Face Generation</b></td>
    <td align="center"><b>Text-to-Face Generation</b></td>
  </tr>
  <tr>
    <td><img src="assets/mask2face.svg" width="100%"></td>
    <td><img src="assets/text2face.svg" width="100%"></td>
  </tr>
</table>

### Multimodal Generation

<p align="center">
  <img src="assets/multi_model.svg" width="100%"/>
</p>

### Ablation Study

<p align="center">
  <img src="assets/ablation.svg" width="100%"/>
</p>

### Zero-Shot Generalization

- MM-FFHQ-Female

<p align="center">
  <img src="assets/zero_ffhq.svg" width="100%"/>
</p>

- MM-FairFace-HQ

## Citation

```bibtex
@misc{zou2025mixturegloballocalexperts,
      title={Mixture of Global and Local Experts with Diffusion Transformer for Controllable Face Generation}, 
      author={Xuechao Zou and Shun Zhang and Xing Fu and Yue Li and Kai Li and Yushe Cao and Congyan Lang and Pin Tao and Junliang Xing},
      year={2025},
      eprint={2509.00428},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.00428}, 
}
```

## License

This project is licensed under the Apache License 2.0 License. See the [LICENSE](LICENSE) file for details.
