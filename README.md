# 🧠 Fine-Tuning Llama 2 7B for Sentiment Analysis using QLoRA

### Building a domain-specific LLM through Parameter-Efficient Fine-Tuning (PEFT)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-green)](https://huggingface.co/docs/peft/index)
[![Model](https://img.shields.io/badge/Model-Llama%202%207B-purple)](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)

## 📌 Project Overview

This project explores the practical application of **Parameter-Efficient Fine-Tuning (PEFT)** to adapt a Large Language Model (LLM) for a specialized Natural Language Processing (NLP) task: **Binary Sentiment Classification**.

Rather than training a language model from scratch, I fine-tuned the pretrained **Llama 2 7B Chat** model using **QLoRA (Quantized Low-Rank Adaptation)** to classify text into two sentiment categories:

- **Positive**
- **Negative**

The primary objective was to understand and implement an end-to-end LLM fine-tuning workflow, from dataset preparation and 4-bit quantization to adapter training, model merging, Hugging Face Hub integration, and inference using the fine-tuned model.

The project demonstrates how pretrained generative language models can be adapted into task-specific models while significantly reducing the number of trainable parameters and memory requirements associated with conventional full-model fine-tuning.

## 🎯 Key Objectives

- Adapt a pretrained Llama 2 7B Chat model for binary sentiment classification.
- Implement QLoRA to make fine-tuning more memory-efficient.
- Apply LoRA adapters to perform parameter-efficient model adaptation.
- Prepare and format supervised training data using the Llama 2 instruction template.
- Train the model using Hugging Face's Supervised Fine-Tuning Trainer.
- Merge trained LoRA adapters with the original base model.
- Publish the resulting fine-tuned model to the Hugging Face Hub.
- Load and perform sentiment predictions using the published model.

## 🏗️ Technical Architecture

The workflow consists of five major stages:

**1. Dataset Preparation**

A sentiment analysis dataset containing text samples and corresponding sentiment labels is loaded and transformed into instruction-following training examples compatible with Llama 2's chat format.

**2. Model Quantization**

The pretrained Llama 2 7B Chat model is loaded using 4-bit quantization through BitsAndBytes.

**3. Parameter-Efficient Fine-Tuning**

LoRA adapters are configured and trained using the QLoRA approach and the Supervised Fine-Tuning Trainer (SFTTrainer).

**4. Model Merging and Publishing**

The trained LoRA adapters are loaded onto the original base model and merged into the model weights. The resulting model and tokenizer are uploaded to Hugging Face Hub.

**5. Inference**

The published fine-tuned model is downloaded and loaded using Hugging Face Transformers. A text-generation pipeline is used to generate sentiment predictions for new text inputs.

## ⚙️ Fine-Tuning Methodology

### 1. Base Model

The project uses:

**NousResearch/Llama-2-7b-chat-hf**

A pretrained 7-billion-parameter conversational language model based on Meta's Llama 2 architecture.

### 2. QLoRA: 4-bit Quantization

To reduce GPU memory requirements during fine-tuning, the base model is loaded using 4-bit quantization.

The BitsAndBytes configuration includes:

| Configuration | Value |
|---|---|
| Quantization | 4-bit |
| Quantization type | NF4 |
| Compute dtype | Float16 |
| Double quantization | Disabled |

This allows the pretrained model to remain quantized while LoRA adapters are trained.

### 3. LoRA Configuration

Low-Rank Adaptation (LoRA) enables efficient model adaptation by introducing trainable low-rank matrices instead of updating all pretrained model parameters.

The implemented configuration:

| Parameter | Value |
|---|---|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Bias | None |
| Task Type | CAUSAL_LM |

This significantly reduces the number of parameters that need to be updated during fine-tuning.

### 4. Training Configuration

The model is fine-tuned using Hugging Face's `SFTTrainer` from the TRL library.

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Batch Size per Device | 4 |
| Gradient Accumulation | 1 |
| Learning Rate | 2e-4 |
| Optimizer | Paged AdamW 32-bit |
| Weight Decay | 0.001 |
| Learning Rate Scheduler | Cosine |
| Warmup Ratio | 0.03 |
| Maximum Gradient Norm | 0.3 |

The training process uses supervised fine-tuning on instruction-formatted text examples.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Model loading and inference |
| Hugging Face PEFT | Parameter-efficient fine-tuning |
| QLoRA | Quantized model fine-tuning |
| LoRA | Low-rank adapter training |
| BitsAndBytes | 4-bit quantization |
| Hugging Face TRL | Supervised fine-tuning with SFTTrainer |
| Hugging Face Datasets | Dataset preparation |
| Pandas | Data loading and preprocessing |
| Hugging Face Hub | Model publishing and distribution |

## 🔄 Project Workflow

```text
        Sentiment Dataset
                |
                v
       Data Preprocessing
                |
                v
      Llama 2 Prompt Formatting
                |
                v
       Pretrained Llama 2 7B
                |
                v
        4-bit Quantization
          (BitsAndBytes)
                |
                v
       QLoRA + LoRA Setup
                |
                v
        Supervised Fine-Tuning
             (SFTTrainer)
                |
                v
        Trained LoRA Adapters
                |
                v
      Merge Adapters with Base
                |
                v
        Fine-Tuned Llama 2
                |
                v
         Hugging Face Hub
                |
                v
         Model Inference
                |
                v
       Positive / Negative
          Prediction
```

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Llama-2-Sentiment-Analysis-QLoRA.git

cd Llama-2-Sentiment-Analysis-QLoRA
```

### 2. Install Dependencies

```bash
pip install torch
pip install transformers==4.41.2
pip install peft==0.11.1
pip install accelerate==0.30.1
pip install trl==0.8.6
pip install bitsandbytes
pip install datasets pandas
```

### 3. Prepare the Dataset

Place the sentiment dataset in the appropriate location and ensure it contains the following columns:

```text
text,label
"I absolutely love this product",positive
"The experience was disappointing",negative
```

Update the dataset path in the fine-tuning notebook if necessary.

### 4. Fine-Tune the Model

Open and execute:

`llama2_finetuning.ipynb`

The notebook covers:
- Dataset loading and formatting.
- Model loading and 4-bit quantization.
- LoRA configuration.
- Supervised fine-tuning.
- Saving trained adapters.
- Merging LoRA weights with the base model.
- Uploading the model and tokenizer to Hugging Face Hub.

**Note:** Fine-tuning requires a compatible GPU environment with sufficient memory and an appropriate CUDA/BitsAndBytes setup.

### 5. Run Inference

Open:

`use_finetuned_llama_model.ipynb`

The notebook downloads the published model and performs sentiment classification on custom text inputs.

## 💻 Example Inference

The fine-tuned model is prompted to identify the sentiment of a given text.

**Example 1:**

Input:

```text
I like that product.
```

Expected sentiment:

```text
positive
```

**Example 2:**

Input:

```text
I am very disappointed about that decision.
```

Expected sentiment:

```text
negative
```

These examples illustrate the intended classification behavior, rather than reporting independently measured evaluation results.

## 📂 Repository Structure

```text
Llama-2-Sentiment-Analysis-QLoRA/
│
├── llama2_finetuning.ipynb
│
├── use_finetuned_llama_model.ipynb
│
└── README.md
```

**Dataset:** The original notebook loads `sentiment_analysis_dataset.csv` from Google Colab. The dataset is not included in this repository and must be supplied separately.

## 🧠 Key Engineering Takeaways

Through this project, I gained hands-on experience with:

- Fine-tuning pretrained Large Language Models for downstream NLP tasks.
- Understanding the practical differences between full fine-tuning and parameter-efficient fine-tuning.
- Implementing QLoRA using 4-bit NF4 quantization.
- Configuring and training LoRA adapters using PEFT.
- Formatting instruction-following datasets for conversational LLMs.
- Using supervised fine-tuning techniques with Hugging Face TRL.
- Merging adapter weights into pretrained model architectures.
- Publishing and loading custom fine-tuned models through Hugging Face Hub.
- Building an inference workflow around a specialized LLM.

## 🔮 Potential Future Improvements

- Introduce a dedicated validation and test dataset.
- Evaluate classification performance using accuracy, precision, recall, and F1-score.
- Compare the fine-tuned model against the original pretrained model.
- Experiment with different LoRA ranks and learning rates.
- Implement structured output validation to enforce positive/negative predictions.
- Explore inference optimization and model serving through an API.
- Extend the classifier to support multiclass sentiment analysis.


---

*This project was developed as a hands-on exploration of parameter-efficient LLM fine-tuning, model adaptation, and inference using the Hugging Face ecosystem.*
