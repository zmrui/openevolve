# HuggingFace Dataset Prompt Optimization with OpenEvolve 🚀

This example demonstrates how to use OpenEvolve to automatically optimize prompts for any HuggingFace dataset. The system uses evolutionary search to discover high-performing prompts by testing them against ground truth data.

## 🎯 Overview

OpenEvolve automatically:
- Loads any HuggingFace dataset
- Evolves prompts through multiple generations
- Uses cascading evaluation for efficiency
- Finds optimal prompts for your specific task and model

The system uses a clean YAML format for configuration, making it easy to set up prompt optimization for any dataset.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd examples/llm_prompt_optimization
pip install -r requirements.txt
```

### 2. Configure Your Model

Update `config.yaml` with your LLM settings:

```yaml
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "your_api_key_here"
  models:
    - name: "google/gemini-2.5-flash"  # Or any OpenAI-compatible model
      weight: 1.0
```

### 3. Set Up Your Dataset and Prompt

Configure your dataset in `dataset.yaml`:

```yaml
# HuggingFace dataset configuration
dataset_name: "stanfordnlp/imdb"  # Any HuggingFace dataset
input_field: "text"               # Field containing input data
target_field: "label"             # Field containing ground truth
split: "test"                     # Dataset split to use

# Evaluation samples
max_samples: 50    # Number of samples to evaluate
```

Create your initial prompt in `initial_prompt.txt`:

```
Your initial prompt here with {input_text} as placeholder
```

### 4. Run OpenEvolve

```bash
python ../../openevolve-run.py initial_prompt.txt evaluator.py --config config.yaml --iterations 100
```

The system will:
- Evolve the prompt in `initial_prompt.txt`
- Use dataset configuration from `dataset.yaml`
- Test evolved prompts against the HuggingFace dataset

## 📊 Supported Datasets

This optimizer works with any HuggingFace dataset. Example configurations are provided in the `examples/` directory:

- **AG News**: `ag_news_dataset.yaml` + `ag_news_prompt.txt`
- **Emotion**: `emotion_dataset.yaml` + `emotion_prompt.txt`

To use an example:
```bash
# Copy the example files
cp examples/ag_news_dataset.yaml dataset.yaml
cp examples/ag_news_prompt.txt initial_prompt.txt

# Run optimization
python ../../openevolve-run.py initial_prompt.txt evaluator.py --config config.yaml --iterations 100
```

### Common Dataset Configurations:

### Sentiment Analysis
```yaml
dataset_name: "stanfordnlp/imdb"
input_field: "text"
target_field: "label"  # 0 or 1
```

### Question Answering
```yaml
dataset_name: "squad"
input_field: "question"
target_field: "answers"  # Dict with 'text' field
```

### Text Classification
```yaml
dataset_name: "ag_news"
input_field: "text"
target_field: "label"  # 0-3 for categories
```

### Summarization
```yaml
dataset_name: "xsum"
input_field: "document"
target_field: "summary"
```

## ⚙️ How It Works

### Simple Evaluation

The evaluator uses a straightforward single-stage evaluation:

1. **Load Dataset**: Downloads the specified HuggingFace dataset
2. **Sample Data**: Takes `max_samples` examples from the dataset
3. **Test Prompt**: Sends each example through the LLM with the prompt
4. **Calculate Accuracy**: Compares LLM outputs to ground truth labels

### Evolution Process

1. OpenEvolve starts with your initial prompt
2. The LLM generates variations based on performance feedback
3. Each variant is tested using cascading evaluation
4. Best performers are kept and evolved further
5. Process continues for specified iterations

### 🎭 Custom Templates for Prompt Evolution

By default, OpenEvolve is designed for code evolution. To make it work properly for prompt evolution, this example includes custom templates in the `templates/` directory:

- **`full_rewrite_user.txt`**: Replaces the default code evolution template with prompt-specific language

This ensures the LLM understands it should evolve the prompt text itself, not generate code. The configuration automatically uses these templates via:

```yaml
prompt:
  template_dir: "templates"  # Use custom templates for prompt evolution
```

## 🎯 Configuration Options

### Evaluation Configuration

In `config.yaml`:
```yaml
evaluator:
  parallel_evaluations: 4      # Run 4 evaluations in parallel
  cascade_evaluation: false    # Simple single-stage evaluation
```

### Sample Size

Adjust in `dataset.yaml`:
```yaml
max_samples: 50    # Number of samples to evaluate
```

## 📈 Example Results

Starting prompt:
```
Analyze the sentiment: "{input_text}"
```

Evolved prompt after 100 iterations:
```
Analyze the sentiment of the following text. Determine if the overall emotional tone is positive or negative.

Text: "{input_text}"

Response: Provide only a single digit - either 1 for positive sentiment or 0 for negative sentiment. Do not include any explanation or additional text.
```

Accuracy improvement: 72% → 94%

## 🔧 Advanced Usage

### Custom Evaluation Metrics

The evaluator extracts predictions and compares them to ground truth. For classification tasks, it looks for:
- Exact number matches (0, 1, etc.)
- Keywords (positive/negative, yes/no)
- Custom patterns you define

### Different Task Types

While the default setup is for classification, you can modify the evaluator for:
- **Regression**: Compare numeric outputs
- **Generation**: Use BLEU/ROUGE scores
- **Extraction**: Check if key information is present

## 🐛 Troubleshooting

### Dataset Not Found
- Check the exact name on HuggingFace
- Some datasets require acceptance of terms

### Low Stage 1 Accuracy
- Your initial prompt may be too vague
- Check if the output format matches expectations
- Verify the dataset fields are correct

### API Errors
- Ensure your API key is valid
- Check rate limits
- Verify the model name is correct

## 🚀 Tips for Best Results

1. **Start Simple**: Begin with a clear, working prompt
2. **Clear Output Format**: Specify exactly what output you expect
3. **Appropriate Samples**: More samples = better evaluation but slower
4. **Multiple Runs**: Evolution has randomness; try multiple runs
5. **Monitor Progress**: Check intermediate best_program.txt files

## 📚 Next Steps

- Try different datasets from HuggingFace
- Experiment with different models
- Adjust evolution parameters in config.yaml
- Create task-specific evaluation metrics

Happy prompt evolving! 🧬✨