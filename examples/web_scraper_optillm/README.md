# Web Scraper Evolution with optillm

This example demonstrates how to use [optillm](https://github.com/codelion/optillm) with OpenEvolve to leverage test-time compute techniques for improved code evolution accuracy. We'll evolve a web scraper that extracts structured data from documentation pages, showcasing two key optillm features:

1. **readurls plugin**: Automatically fetches webpage content when URLs are mentioned in prompts
2. **Inference optimization**: Uses techniques like Mixture of Agents (MoA) to improve response accuracy

## Why optillm?

Traditional LLM usage in code evolution has limitations:
- LLMs may not have knowledge of the latest library documentation
- Single LLM calls can produce inconsistent or incorrect code
- No ability to dynamically fetch relevant documentation during evolution

optillm solves these problems by:
- **Dynamic Documentation Fetching**: The readurls plugin automatically fetches and includes webpage content when URLs are detected in prompts
- **Test-Time Compute**: Techniques like MoA generate multiple responses and synthesize the best solution
- **Flexible Routing**: Can route requests to different models based on requirements

## Problem Description

We're evolving a web scraper that extracts API documentation from Python library documentation pages. The scraper needs to:
1. Parse HTML documentation pages
2. Extract function signatures, descriptions, and parameters
3. Structure the data in a consistent format
4. Handle various documentation formats

This is an ideal problem for optillm because:
- The LLM benefits from seeing actual documentation HTML structure
- Accuracy is crucial for correct parsing
- Different documentation sites have different formats

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenEvolve    │────▶│     optillm     │────▶│   Local LLM     │
│                 │     │  (proxy:8000)   │     │  (Qwen-0.5B)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ├── readurls plugin
                               │   (fetches web content)
                               │
                               └── MoA optimization
                                   (improves accuracy)
```

## Setup Instructions

### 1. Install and Configure optillm

```bash
# Clone optillm
git clone https://github.com/codelion/optillm.git
cd optillm

# Install dependencies
pip install -r requirements.txt

# Start optillm proxy with local inference server (in a separate terminal)
export OPTILLM_API_KEY=optillm
python optillm.py --port 8000
```

optillm will now be running on `http://localhost:8000` with its built-in local inference server.

**Note for Non-Mac Users**: This example uses `Qwen/Qwen3-0.6B-MLX-bf16` which is optimized for Apple Silicon (M1/M2/M3 chips). If you're not using a Mac, you should:

1. **For NVIDIA GPUs**: Use a CUDA-compatible model like:
   - `Qwen/Qwen2.5-32B-Instruct` (best quality, high VRAM)
   - `Qwen/Qwen2.5-14B-Instruct` (good balance)
   - `meta-llama/Llama-3.1-8B-Instruct` (efficient option)
   - `Qwen/Qwen2.5-7B-Instruct` (lower VRAM)

2. **For CPU-only**: Use a smaller model like:
   - `Qwen/Qwen2.5-7B-Instruct` (7B parameters)
   - `meta-llama/Llama-3.2-3B-Instruct` (3B parameters)
   - `Qwen/Qwen2.5-3B-Instruct` (3B parameters)

3. **Update the config**: Replace the model names in `config.yaml` with your chosen model:
   ```yaml
   models:
     - name: "readurls-your-chosen-model"
       weight: 0.6
     - name: "moa&readurls-your-chosen-model"
       weight: 0.4
   ```

### 2. Install Web Scraping Dependencies

```bash
# Install required Python packages for the example
pip install -r examples/web_scraper_optillm/requirements.txt
```

### 3. Run the Evolution

```bash
# From the openevolve root directory
export OPENAI_API_KEY=optillm
python openevolve-run.py examples/web_scraper_optillm/initial_program.py \
    examples/web_scraper_optillm/evaluator.py \
    --config examples/web_scraper_optillm/config.yaml \
    --iterations 100
```

The configuration demonstrates both optillm capabilities:
- **Primary model (90%)**: `readurls-Qwen/Qwen3-0.6B-MLX-bf16` - fetches URLs mentioned in prompts
- **Secondary model (10%)**: `moa&readurls-Qwen/Qwen3-0.6B-MLX-bf16` - uses Mixture of Agents for improved accuracy

## How It Works

### 1. readurls Plugin

When the evolution prompt contains URLs (e.g., "Parse the documentation at https://docs.python.org/3/library/json.html"), the readurls plugin:
1. Detects the URL in the prompt
2. Fetches the webpage content
3. Extracts text and table data
4. Appends it to the prompt as context

This ensures the LLM has access to the latest documentation structure when generating code.

### 2. Mixture of Agents (MoA)

The MoA technique improves accuracy by:
1. Generating 3 different solutions to the problem
2. Having each "agent" critique all solutions
3. Synthesizing a final, improved solution based on the critiques

This is particularly valuable for complex parsing logic where multiple approaches might be valid.

### 3. Evolution Process

1. **Initial Program**: A basic BeautifulSoup scraper that extracts simple text
2. **Evaluator**: Tests the scraper against real documentation pages, checking:
   - Correct extraction of function names
   - Accurate parameter parsing
   - Proper handling of edge cases
3. **Evolution**: The LLM improves the scraper by:
   - Fetching actual documentation HTML (via readurls)
   - Generating multiple parsing strategies (via MoA)
   - Learning from evaluation feedback

## Example Evolution Trajectory

**Generation 1** (Basic scraper):
```python
# Simple text extraction
soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()
```

**Generation 10** (With readurls context):
```python
# Targets specific documentation structures
functions = soup.find_all('dl', class_='function')
for func in functions:
    name = func.find('dt').get('id')
    desc = func.find('dd').text
```

**Generation 50** (With MoA refinement):
```python
# Robust parsing with error handling
def extract_function_docs(soup):
    # Multiple strategies for different doc formats
    strategies = [
        lambda: soup.select('dl.function dt'),
        lambda: soup.select('.sig-name'),
        lambda: soup.find_all('code', class_='descname')
    ]
    
    for strategy in strategies:
        try:
            results = strategy()
            if results:
                return parse_results(results)
        except:
            continue
```

## Monitoring Progress

Watch the evolution progress and see how optillm enhances the process:

```bash
# View optillm logs (in the terminal running optillm)
# You'll see:
# - URLs being fetched by readurls
# - Multiple completions generated by MoA
# - Final synthesized responses

# View OpenEvolve logs
tail -f examples/web_scraper_optillm/openevolve_output/evolution.log
```

## Results

After evolution, you should see:
1. **Improved Accuracy**: The scraper correctly handles various documentation formats
2. **Better Error Handling**: Robust parsing that doesn't break on edge cases
3. **Optimized Performance**: Efficient extraction strategies

Compare the checkpoints to see the evolution:
```bash
# Initial vs evolved program
diff examples/web_scraper_optillm/openevolve_output/checkpoints/checkpoint_10/best_program.py \
     examples/web_scraper_optillm/openevolve_output/checkpoints/checkpoint_100/best_program.py
```

## Key Insights

1. **Documentation Access Matters**: The readurls plugin significantly improves the LLM's ability to generate correct parsing code by providing actual HTML structure

2. **Test-Time Compute Works**: MoA's multiple generation and critique approach produces more robust solutions than single-shot generation

3. **Powerful Local Models**: Large models like Qwen-32B with 4-bit quantization provide excellent results while being memory efficient when enhanced with optillm techniques

## Customization

You can experiment with different optillm features by modifying `config.yaml`:

1. **Different Plugins**: Try the `executecode` plugin for runtime validation
2. **Other Techniques**: Experiment with `cot_reflection`, `rstar`, or `bon`
3. **Model Combinations**: Adjust weights or try different technique combinations

Example custom configuration:
```yaml
llm:
  models:
    - name: "cot_reflection&readurls-Qwen/Qwen3-0.6B-MLX-bf16"
      weight: 0.7
    - name: "moa&executecode-Qwen/Qwen3-0.6B-MLX-bf16"
      weight: 0.3
```

## Troubleshooting

1. **optillm not responding**: Ensure it's running on port 8000 with `OPTILLM_API_KEY=optillm`
2. **Model not found**: Make sure optillm's local inference server is working (check optillm logs)
3. **Slow evolution**: MoA generates multiple completions, so it's slower but more accurate

## Further Reading

- [optillm Documentation](https://github.com/codelion/optillm)
- [OpenEvolve Configuration Guide](../../configs/default_config.yaml)
- [Mixture of Agents Paper](https://arxiv.org/abs/2406.04692)