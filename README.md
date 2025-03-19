## Matering structured output with LangChain

### Setup and Installation

This project uses Poetry for dependency management. Follow these steps to get started:

1. Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository and navigate to the project directory:

```bash
git clone git@github.com:adocherty/langchain-structured-output-evaluation.git
cd langchain-structured-output-evaluation
```

3. Create a virtual environment and install dependencies:

```bash
poetry install
```

4. Activate the virtual environment:

```bash
poetry shell
```

5. Launch Jupyter to run the notebooks:

```bash
jupyter notebook
```

### Running Examples

These examples can be run on Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adocherty/mastering-structured-output/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb)

See `index.ipynb` for a description and link to the notebooks.

### Notes

httpx has been fixed to version 0.27.2 to avoid a bug due to changed parameters
https://community.openai.com/t/error-with-openai-1-56-0-client-init-got-an-unexpected-keyword-argument-proxies/1040332

### Requirements

- Python 3.12
- Poetry for dependency management
- Ollama installed locally for running local LLMs
- Anthropic API key for Claude access (optional)
- Fireworks.ai API key for cloud models (optional)
