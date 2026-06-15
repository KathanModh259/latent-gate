# Contributing to LatentGate

First off, thanks for considering contributing! 🎉

## How to Contribute

### 🐛 Bug Reports
- Use the Bug Report issue template
- Include your Python version, Ollama version, and OS
- Provide minimal reproduction steps

### 💡 Feature Requests
- Use the Feature Request issue template
- Describe the problem your feature would solve
- Suggest an implementation approach if possible

### 🔧 Pull Requests

1. **Fork** the repo and create your branch from `main`
2. **Install** dev dependencies: `pip install -r requirements-dev.txt`
3. **Write tests** for any new functionality
4. **Run tests**: `pytest tests/`
5. **Format** your code: `black latent_gate/`
6. **Submit** your PR with a clear description

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to public methods
- Keep functions focused and small

### Priority Areas
We especially welcome contributions in these areas:
- 🖼️ New vision model integrations
- 📹 Video processing improvements
- 🧮 Better semantic similarity algorithms
- 🌐 API server wrappers (FastAPI, Flask)
- 📊 Cost tracking and analytics
- 🧪 Test coverage improvements
- 📖 Documentation and examples

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/latent-gate.git
cd latent-gate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
