# LLaMA LoRA Fine-tuning

Implementing LoRA (Low-Rank Adaptation) from scratch in PyTorch and applying it to fine-tune a LLaMA model. LoRA freezes the pretrained weights and injects small trainable low-rank matrices into linear layers, drastically reducing the number of trainable parameters needed for fine-tuning.
