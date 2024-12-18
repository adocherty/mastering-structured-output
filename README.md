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

- `1-langchain-structured-output-examples.ipynb`: Demonstrates different methods for getting structured output from LLMs using LangChain
- `2-langchain-structured-output-evaluation.ipynb`: Shows how to evaluate the quality of structured outputs

### Requirements

- Python 3.12
- Poetry for dependency management
- Ollama installed locally for running local LLMs
- Anthropic API key for Claude access (optional)
