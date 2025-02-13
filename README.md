## Matering structured output with LangChain

See explanation on blog post:

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

The examples are provided as Jupyter notebooks in the `notebooks/` directory:

- `1-langchain-structured-output-examples.ipynb`: Demonstrates different methods for getting structured output from LLMs using LangChain in initial blog:
  https://medium.com/@docherty/mastering-structured-output-in-llms-choosing-the-right-model-for-json-output-with-langchain-be29fb6f6675

- `2-langchain-structured-output-evaluation.ipynb`: Shows how to evaluate the quality of structured outputs

- `3-langchain-structured-output-examples-updated.ipynb`: Update of initial examples with new style methods for getting structured output from LLMs using LangChain. See updated blog

- `4-langchain-structured-output-evaluation-updated.ipynb`: Update of initial examples with new style methods for getting structured output from LLMs using LangChain. See updated blog

- `5-langchain-xml-output-examples.ipynb`: Notebook of examples for XML output using LangChain.

### Notes

httpx has been fixed to version 0.27.2 to avoid a bug due to changed parameters
https://community.openai.com/t/error-with-openai-1-56-0-client-init-got-an-unexpected-keyword-argument-proxies/1040332

### Requirements

- Python 3.12
- Poetry for dependency management
- Ollama installed locally for running local LLMs
- Anthropic API key for Claude access (optional)
- Fireworks.ai API key for cloud models (optional)
