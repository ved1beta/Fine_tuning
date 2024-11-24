# Fine-tuning Project

This repository contains code for fine-tuning the OPT-350M language model using the SFT (Supervised Fine-Tuning) approach. The project utilizes the Hugging Face Transformers library and TRL (Transformer Reinforcement Learning) for training.

## 🚀 Features

- Fine-tune OPT-350M model using SFT
- Memory-efficient training with gradient checkpointing
- Support for 8-bit quantization
- Configurable batch size and gradient accumulation
- Optional Flash Attention 2.0 support

## 📋 Requirements

```bash
pip install transformers
pip install torch
pip install trl
pip install accelerate
pip install bitsandbytes  # for 8-bit quantization
```

## 💾 Hardware Requirements

- CUDA-compatible GPU with at least 8GB VRAM
- 16GB+ RAM recommended
- SSD with at least 10GB free space

## 🛠️ Usage

### Basic Training Setup

```python
from trl import SFTTrainer, SFTConfig
import torch

# Clear CUDA cache
torch.cuda.empty_cache()

# Training configuration
training_args = SFTConfig(
    max_seq_length=512,
    output_dir='/tmp',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch"
)

# Initialize trainer
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args
)

# Start training
trainer.train()
```

### Memory-Optimized Setup

```python
from transformers import AutoModelForCausalLM
import torch

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    device_map="auto",
    load_in_8bit=True,
)

# Enhanced training configuration
training_args = SFTConfig(
    max_seq_length=256,  # Reduced sequence length
    output_dir='/tmp',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    use_flash_attention_2=True,  # For PyTorch 2.0+
    optim="adamw_torch"
)
```

## 🔧 Troubleshooting

### CUDA Out of Memory Issues

If you encounter CUDA out of memory errors, try these solutions:

1. **Reduce Batch Size**: Already set to 1 in current configuration
2. **Enable Gradient Checkpointing**: Implemented in current setup
3. **Use 8-bit Quantization**: See memory-optimized setup
4. **Reduce Sequence Length**: Try reducing max_seq_length
5. **Clear CUDA Cache**: Run `torch.cuda.empty_cache()` before training

### Common Error Messages

```python
RuntimeError: CUDA error: out of memory
```
Solution: Implement the memory optimization techniques described above.

## 📊 Dataset Format

Your dataset should be formatted as follows:
- Compatible with Hugging Face's Dataset format
- Contains necessary text fields for training
- Properly preprocessed and tokenized

## 🤝 Contributing

Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face team for Transformers library
- Meta AI for the OPT model
- TRL team for the SFT Trainer

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [OPT Model Card](https://huggingface.co/facebook/opt-350m)
