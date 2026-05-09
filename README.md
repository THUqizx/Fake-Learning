# Fake-Learning for Qwen

This repository provides an implementation of Fake-Learning for the Qwen-7B language model, enhancing token generation quality through token collocation spike value modeling.

## How to Use

### Method 1: Manual Integration

To integrate Fake-Learning into your Qwen model, follow these steps:

1. **Copy the logits modifier**: Copy `modify_logits.py` to your Qwen model directory.

2. **Replace the modeling file**: Replace the original `modeling_qwen.py` with the provided `modeling_qwen.py` from this repository.

3. **Download the token collocation matrix**: Download the pre-computed `TokenCollocationSpikeValues_Matrix.mtx` file from HuggingFace and place it in your Qwen model directory.
   
   Download link: [TokenCollocationSpikeValues_Matrix.mtx](https://huggingface.co/ZhixiaoQi/Qwen-7B-chat-Fake-Learning/blob/main/TokenCollocationSpikeValues_Matrix.mtx)

### Method 2: Direct Download from HuggingFace

Alternatively, you can directly download our pre-packaged Qwen-7B Fake-Learning model from HuggingFace:

**Model Link**: [Qwen-7B-chat-Fake-Learning](https://huggingface.co/ZhixiaoQi/Qwen-7B-chat-Fake-Learning)

## Loading and Inference

Loading and inference with the Fake-Learning enhanced model is identical to the original Qwen model.

### Loading the Model

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL = "path/to/your/Qwen-7B-chat-Fake-Learning"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", trust_remote_code=True).eval()

# Set generation config
model.generation_config = GenerationConfig.from_pretrained(MODEL, trust_remote_code=True)
```

### Running Inference

```python
# Prepare your prompt
prompt = "Your input prompt here"

# Generate response
response, history = model.chat(tokenizer, prompt, history=None)

print(response)
```

## Configuration

The Fake-Learning implementation uses an alpha parameter to control the influence of token collocation spike values on the generation process. You can adjust this parameter when running inference:

```bash
python eval.py --alpha 0.9
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- SciPy
- CUDA (recommended for GPU acceleration)

## License

This project is licensed under the same license as the original Qwen model.
