
# ZeroLM - Training-Free Language Model

A context-aware, self-learning chatbot that adapts through conversation without traditional training.

## Features

- 🧠 Training-free learning through conversation
- 🔍 Context-aware response generation
- 📈 Confidence-based answer selection
- 💾 Persistent memory with automatic saving
- ⚡ Real-time interaction via Gradio interface

## Installation

```bash
git clone https://github.com/yourusername/zerolm.git
cd zerolm
pip install -r requirements.txt
```

## Quick Start

```python
from zerolm import ZeroShotLM

# Initialize chatbot
chatbot = ZeroShotLM(
    use_vectors=True,
    context_window=5,
    language="en"
)

# Interactive learning
chatbot.learn("What is AI?", "Artificial Intelligence")
response = chatbot.process_query("Explain artificial intelligence")
print(response.text)
```

## Documentation

### Key Components

| Component          | Description                                  |
|--------------------|----------------------------------------------|
| `ZeroShotLM`       | Core language model implementation          |
| `ChatbotInterface` | Gradio-based web interface                  |
| `MemoryManager`    | Handles pattern storage and retrieval       |
| `LearningValidator`| Ensures response consistency                |

### Usage Examples

**Basic Chat:**
```python
response = chatbot.process_query("What is machine learning?")
print(f"Response: {response.text} (Confidence: {response.confidence:.2f})")
```

**Batch Learning:**
```python
questions = ["What is AI?", "Define NLP"]
answers = ["Artificial Intelligence", "Natural Language Processing"]
chatbot.learn_batch(questions, answers)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by memory-augmented neural networks
- Built with Gradio for intuitive UI
- Vector operations powered by NumPy
```
