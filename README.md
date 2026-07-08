# Backward Traceability in Agile Development

This repository is the replication package for a thesis on backward traceability between natural-language requirements and TESTAR-extracted GUI state-transition models. It compares traditional information retrieval methods and contextual embedding methods for linking requirements to GUI widgets and transitions, then derives backward GUI-to-requirement rankings by inverting the same forward similarity scores.

The target application is the Practice Software Testing Toolshop. The active experiment set contains four GUI model scenarios:

- `Focus-GPT5`
- `Focus-Gemma3`
- `LLMReq-GPT5`
- `LLMReq-Gemma3`

The final thesis figures are stored in `Thesis Figures/`. The plotting workflow regenerates those figures directly from CSV files in `Thesis Figures/data/`.

## Repository Layout

```text
.
|-- Models/
|-- Filtered models/
|-- Filtered relaxed models/
|-- Filtering/
|-- Requirement Specifications/
|-- Ground Truth/
|-- Results/
|   |-- IR/
|   |   |-- Strict/
|   |   `-- Relaxed/
|   `-- CE/
|       |-- Strict/
|       `-- Relaxed/
|-- Thesis Figures/
|   |-- Backward_GT_Only_Figures/
|   |-- Backward_Traceability_Figures/
|   |-- Descriptive_Statistics/
|   |-- Filtering_Figures/
|   |-- Forward_Traceability_Figures/
|   |-- Forward_VS_Backward_Figures/
|   |-- GuidingLLM_Req_Source_Figures/
|   |-- Scalability_Figures/
|   `-- data/
|-- Kaggle/
|-- requirements.txt
|-- requirements-stella.txt
|-- calculate_metrics_charts.py
|-- plot_metrics_charts.py
|-- chart_thesis_figures.py
|-- metrics_common.py
|-- candidate_construction.py
|-- text_preprocessing.py
|-- results_common.py
|-- embedding_requirements_compare_common.py
|-- vsm_requirements_compare.py
|-- lsi_requirements_compare.py
|-- jsm_requirements_compare.py
|-- qwen3_embedding_0_6b_requirements_compare.py
|-- qwen3_embedding_4b_requirements_compare.py
|-- jina_embeddings_v3_requirements_compare.py
`-- stella_en_1_5b_v5_requirements_compare.py
```

Directory roles:

| Path | Role |
| --- | --- |
| `Models/` | Raw TESTAR GUI state-transition model exports. |
| `Filtered models/` | Strict filtered GUI model JSON files used as traceability inputs. |
| `Filtered relaxed models/` | Relaxed filtered GUI model JSON files used as traceability inputs. |
| `Filtering/` | Scripts that create strict and relaxed filtered models from raw TESTAR exports. |
| `Requirement Specifications/` | Requirement text files for the focus group and LLM-generated requirement sets. |
| `Ground Truth/` | Scenario-specific ground-truth CSV files and the source workbooks. |
| `Results/IR/` | Traditional IR ranking outputs for VSM, LSI, and JSM. |
| `Results/CE/` | Contextual embedding ranking outputs for Qwen, Jina, and Stella. |
| `Thesis Figures/` | Current final thesis figures, grouped by thesis result section. |
| `Thesis Figures/data/` | Current metric CSV baseline used by the final plotting workflow. |
| `Kaggle/` | Standalone Kaggle scripts and packaged input copies. These are separate from the root-level replication workflow. |

## Active Inputs

| Scenario | Requirement file | Raw model | Strict filtered model | Relaxed filtered model | Ground truth |
| --- | --- | --- | --- | --- | --- |
| `Focus-GPT5` | `Requirement Specifications/Requirements focus group.txt` | `Models/GPT5_focus.json` | `Filtered models/filtered_model_GPT5_Focus.json` | `Filtered relaxed models/filtered_model_relaxed_GPT5_Focus.json` | `Ground Truth/Ground truth Focus_GPT5.csv` |
| `Focus-Gemma3` | `Requirement Specifications/Requirements focus group.txt` | `Models/Gemma3_focus.json` | `Filtered models/filtered_model_Gemma3_Focus.json` | `Filtered relaxed models/filtered_model_relaxed_Gemma3_Focus.json` | `Ground Truth/Ground truth Focus_Gemma3.csv` |
| `LLMReq-GPT5` | `Requirement Specifications/Agentic Requirements.txt` | `Models/GPT5_Llmreq.json` | `Filtered models/filtered_model_GPT5_Llmreq.json` | `Filtered relaxed models/filtered_model_relaxed_GPT5_Llmreq.json` | `Ground Truth/Ground truth LLM_GPT5.csv` |
| `LLMReq-Gemma3` | `Requirement Specifications/Agentic Requirements.txt` | `Models/Gemma3_Llmreq.json` | `Filtered models/filtered_model_Gemma3_Llmreq.json` | `Filtered relaxed models/filtered_model_relaxed_Gemma3_Llmreq.json` | `Ground Truth/Ground truth LLM_Gemma3.csv` |

## Dependencies

Run scripts from the repository root. The repository uses two dependency manifests:

- `requirements.txt` covers all non-Stella scripts: filtering, IR baselines, metrics/charts, Qwen, and Jina.
- `requirements-stella.txt` covers `stella_en_1_5b_v5_requirements_compare.py`, because Stella uses different `sentence-transformers`, `transformers`, and `huggingface-hub` versions.

Create the main environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a separate Stella environment when rerunning the Stella script:

```powershell
python -m venv .venv-stella
.\.venv-stella\Scripts\python.exe -m pip install -r requirements-stella.txt
```

The pinned packages are:

| Environment | Manifest | Main use |
| --- | --- | --- |
| Main | `requirements.txt` | Filtering, IR, metrics/charts, Qwen, and Jina. |
| Stella | `requirements-stella.txt` | Stella embedding runs only. |

## End-To-End Replication Workflow

Run the full workflow from the repository root. After cloning or downloading the repository, replace the path below with the folder where the repository is stored on your machine:

```powershell
Set-Location "C:\path\to\GUI-Traceability-Framework"
```

The commands below use direct virtual-environment Python paths instead of activating the environments. They write regenerated ranking, metric, and figure outputs under `replication_run/`. The filtering step rewrites `Filtered models/` and `Filtered relaxed models/`, because the metric code uses those active folders for ground-truth resolution and candidate-space summaries.

### 1. Create Python Environments

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

python -m venv .venv-stella
.\.venv-stella\Scripts\python.exe -m pip install -r requirements-stella.txt

$Py = ".\.venv\Scripts\python.exe"
$StellaPy = ".\.venv-stella\Scripts\python.exe"
```

### 2. Filter The Raw TESTAR Models

```powershell
& $Py Filtering\filter_model_elements.py --input "Models\GPT5_focus.json" --output "Filtered models\filtered_model_GPT5_Focus.json"
& $Py Filtering\filter_model_elements.py --input "Models\Gemma3_focus.json" --output "Filtered models\filtered_model_Gemma3_Focus.json"
& $Py Filtering\filter_model_elements.py --input "Models\GPT5_Llmreq.json" --output "Filtered models\filtered_model_GPT5_Llmreq.json"
& $Py Filtering\filter_model_elements.py --input "Models\Gemma3_Llmreq.json" --output "Filtered models\filtered_model_Gemma3_Llmreq.json"

& $Py Filtering\filter_model_elements_relaxed.py --input "Models\GPT5_focus.json" --output "Filtered relaxed models\filtered_model_relaxed_GPT5_Focus.json"
& $Py Filtering\filter_model_elements_relaxed.py --input "Models\Gemma3_focus.json" --output "Filtered relaxed models\filtered_model_relaxed_Gemma3_Focus.json"
& $Py Filtering\filter_model_elements_relaxed.py --input "Models\GPT5_Llmreq.json" --output "Filtered relaxed models\filtered_model_relaxed_GPT5_Llmreq.json"
& $Py Filtering\filter_model_elements_relaxed.py --input "Models\Gemma3_Llmreq.json" --output "Filtered relaxed models\filtered_model_relaxed_Gemma3_Llmreq.json"
```

Check that eight filtered JSON files are present:

```powershell
Get-ChildItem -LiteralPath "Filtered models", "Filtered relaxed models" -File -Filter *.json |
  Measure-Object
```

### 3. Define The Scenario List

```powershell
Remove-Item -LiteralPath "replication_run" -Recurse -Force -ErrorAction SilentlyContinue

$Scenarios = @(
  @{
    Suffix = "Focus_GPT5"
    Requirements = "Requirement Specifications\Requirements focus group.txt"
    StrictData = "Filtered models\filtered_model_GPT5_Focus.json"
    RelaxedData = "Filtered relaxed models\filtered_model_relaxed_GPT5_Focus.json"
  },
  @{
    Suffix = "Focus_Gemma3"
    Requirements = "Requirement Specifications\Requirements focus group.txt"
    StrictData = "Filtered models\filtered_model_Gemma3_Focus.json"
    RelaxedData = "Filtered relaxed models\filtered_model_relaxed_Gemma3_Focus.json"
  },
  @{
    Suffix = "Llmreq_GPT5"
    Requirements = "Requirement Specifications\Agentic Requirements.txt"
    StrictData = "Filtered models\filtered_model_GPT5_Llmreq.json"
    RelaxedData = "Filtered relaxed models\filtered_model_relaxed_GPT5_Llmreq.json"
  },
  @{
    Suffix = "Llmreq_Gemma3"
    Requirements = "Requirement Specifications\Agentic Requirements.txt"
    StrictData = "Filtered models\filtered_model_Gemma3_Llmreq.json"
    RelaxedData = "Filtered relaxed models\filtered_model_relaxed_Gemma3_Llmreq.json"
  }
)
```

### 4. Run The IR Methods

```powershell
$IrMethods = @(
  @{ Name = "VSM"; Script = "vsm_requirements_compare.py"; Prefix = "vsm" },
  @{ Name = "LSI"; Script = "lsi_requirements_compare.py"; Prefix = "lsi" },
  @{ Name = "JSM"; Script = "jsm_requirements_compare.py"; Prefix = "jsm" }
)

foreach ($Scenario in $Scenarios) {
  foreach ($Method in $IrMethods) {
    & $Py $Method.Script `
      --requirements $Scenario.Requirements `
      --data $Scenario.StrictData `
      --output "replication_run\results\IR\Strict\$($Method.Name)\$($Method.Prefix)_matches_$($Scenario.Suffix).json"

    & $Py $Method.Script `
      --requirements $Scenario.Requirements `
      --data $Scenario.RelaxedData `
      --output "replication_run\results\IR\Relaxed\$($Method.Name)\$($Method.Prefix)_matches_relaxed_$($Scenario.Suffix).json"
  }
}
```

### 5. Run Qwen And Jina

```powershell
$CeMethods = @(
  @{ Name = "Qwen0.6B"; Script = "qwen3_embedding_0_6b_requirements_compare.py"; Prefix = "qwen3_embedding_0.6b" },
  @{ Name = "Qwen4B"; Script = "qwen3_embedding_4b_requirements_compare.py"; Prefix = "qwen3_embedding_4b" },
  @{ Name = "Jina"; Script = "jina_embeddings_v3_requirements_compare.py"; Prefix = "jina_embeddings_v3" }
)

foreach ($Scenario in $Scenarios) {
  foreach ($Method in $CeMethods) {
    & $Py $Method.Script `
      --requirements $Scenario.Requirements `
      --data $Scenario.StrictData `
      --output "replication_run\results\CE\Strict\$($Method.Name)\$($Method.Prefix)_matches_$($Scenario.Suffix).json"

    & $Py $Method.Script `
      --requirements $Scenario.Requirements `
      --data $Scenario.RelaxedData `
      --output "replication_run\results\CE\Relaxed\$($Method.Name)\$($Method.Prefix)_matches_relaxed_$($Scenario.Suffix).json"
  }
}
```

### 6. Run Stella

Run Stella with the separate Stella environment:

```powershell
foreach ($Scenario in $Scenarios) {
  & $StellaPy stella_en_1_5b_v5_requirements_compare.py `
    --requirements $Scenario.Requirements `
    --data $Scenario.StrictData `
    --output "replication_run\results\CE\Strict\Stella\stella_en_1_5b_v5_matches_$($Scenario.Suffix).json"

  & $StellaPy stella_en_1_5b_v5_requirements_compare.py `
    --requirements $Scenario.Requirements `
    --data $Scenario.RelaxedData `
    --output "replication_run\results\CE\Relaxed\Stella\stella_en_1_5b_v5_matches_relaxed_$($Scenario.Suffix).json"
}
```

After all ranking scripts finish, the full run should contain 112 JSON result files:

```powershell
Get-ChildItem -LiteralPath "replication_run\results" -Recurse -File -Filter *.json |
  Measure-Object
```

### 7. Compute Metrics

```powershell
& $Py -B calculate_metrics_charts.py `
  --ground-truth "Ground Truth" `
  --results-root "replication_run\results\IR\Strict" `
  --results-root "replication_run\results\IR\Relaxed" `
  --results-root "replication_run\results\CE\Strict" `
  --results-root "replication_run\results\CE\Relaxed" `
  --output-dir "replication_run\figures" `
  --no-plots
```

Check that 19 CSV files were generated:

```powershell
Get-ChildItem -LiteralPath "replication_run\figures\data" -File -Filter *.csv |
  Measure-Object
```

### 8. Plot The Figures

```powershell
& $Py -B plot_metrics_charts.py --output-dir "replication_run\figures"
```

Check the final output counts:

```powershell
Get-ChildItem -LiteralPath "replication_run\figures" -Recurse -File |
  Group-Object Extension |
  Sort-Object Name |
  ForEach-Object { "$($_.Name) $($_.Count)" }
```

Expected final counts:

```text
.csv 19
.pdf 25
.png 25
```

### Reproducibility Note For CE Runs

The classical IR outputs are expected to be exactly reproducible when the same inputs and package versions are used. Contextual embedding outputs can differ slightly across execution environments. The stored CE baselines were generated in CUDA-enabled environments, while a CPU-only rerun may produce small embedding-score differences that change the order of near-tied candidates. These differences are consistent with CPU/GPU numerical variation in embedding inference.

## Traceability Methods

The replication package evaluates three classical IR baselines and four contextual embedding methods. Each method first produces forward requirement-to-GUI rankings. Backward GUI-to-requirement rankings are then produced by inverting the forward ranked `TopMatches` output with `invert_ranked_results()`, so no separate backward similarity model is trained or computed.

**Classical IR Baselines**

| Method | Script | Ranking idea |
| --- | --- | --- |
| VSM | `vsm_requirements_compare.py` | TF-IDF vector space model with cosine similarity. |
| LSI | `lsi_requirements_compare.py` | TF-IDF projected with SVD-based latent semantic indexing. |
| JSM | `jsm_requirements_compare.py` | Count-vector Jensen-Shannon distance converted to similarity. |

**Contextual Embedding Methods**

| Method | Script | Hugging Face model | Pinned revision |
| --- | --- | --- | --- |
| Qwen3 0.6B | `qwen3_embedding_0_6b_requirements_compare.py` | `Qwen/Qwen3-Embedding-0.6B` | `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418` |
| Qwen3 4B | `qwen3_embedding_4b_requirements_compare.py` | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| Jina v3 | `jina_embeddings_v3_requirements_compare.py` | `jinaai/jina-embeddings-v3-hf` | `d18862d9a48706220815554fac3ebb4dfa46fc28` |
| Stella 1.5B | `stella_en_1_5b_v5_requirements_compare.py` | `it-just-works/stella_en_1.5B_v5_bf16` | `b6f39e45892c6edd44f1e602d84b6adf8891a1e3` |

## Shared Modules

These files keep the repeated parts of the replication workflow in one place. They support the filtering, ranking, metric, and plotting scripts; they are not separate experiments.

| Workflow area | Files | What they handle |
| --- | --- | --- |
| Candidate and text preparation | `candidate_construction.py`, `text_preprocessing.py` | Loading requirements, building GUI candidates, aggregating candidate text, and applying shared text preprocessing. |
| Result handling | `results_common.py` | Writing ranked results, deriving backward rankings from forward output, tracking runtime and memory, and recording package metadata. |
| Contextual embedding support | `embedding_requirements_compare_common.py` | Loading embedding models, selecting the device, ranking candidates, tracking runtime, and writing results for CE methods. |
| Metric computation | `metrics_common.py`, `calculate_metrics_charts.py` | Computing forward and backward metrics, writing the CSV files, and exposing the command-line metric entry point. |
| Final figure generation | `plot_metrics_charts.py`, `chart_thesis_figures.py`, `chart_common.py`, `chart_bar.py`, `chart_heatmap.py`, `chart_distribution.py` | Recreating the curated thesis figures from the CSV files in `Thesis Figures/data/`. |

## Reproduce The Final Figures

The quickest final-figure reproduction path uses the committed CSV baseline in `Thesis Figures/data/`:

```powershell
python -B plot_metrics_charts.py --output-dir "Thesis Figures"
```

This regenerates the curated final thesis figure set in the section subfolders under `Thesis Figures/`.

To recompute the metric CSVs from the existing result JSON files and regenerate the figures, run:

```powershell
python -B calculate_metrics_charts.py --output-dir "Thesis Figures"
```

For a safe verification run, write to a temporary output folder first:

```powershell
python -B calculate_metrics_charts.py --output-dir tmp_pipeline_check_NEW
```

The default metric command scans:

- `Results/IR/Strict`
- `Results/IR/Relaxed`
- `Results/CE/Strict`
- `Results/CE/Relaxed`

Use `--no-plots` to recompute CSV files only:

```powershell
python -B calculate_metrics_charts.py --no-plots --output-dir "Thesis Figures"
```

Use `--plots-only` to regenerate figures from existing CSV files only:

```powershell
python -B calculate_metrics_charts.py --plots-only --output-dir "Thesis Figures"
```

The metric defaults are:

| Setting | Value |
| --- | --- |
| Forward k-values | `1,5,10,20,30,40,50` |
| Backward k-values | `1,3,5,10` |
| Main forward k | `10` |
| Main backward k | `3` |
| Main evaluation view | `combined_actions_and_widgets` |

## Current Metric CSV Files

The active `Thesis Figures/data/` folder contains 19 CSV files. They are grouped by the part of the evaluation they support.

**Forward Traceability**

| CSV | What it contains |
| --- | --- |
| `forward_metrics_per_requirement_by_k.csv` | Per-requirement forward metrics for each k-value and evaluation view. |
| `forward_metrics_summary_by_k.csv` | Aggregate forward metrics by method, scenario, filter variant, k-value, and evaluation view. |
| `forward_per_requirement.csv` | Per-requirement rows for the main forward evaluation setting. |
| `forward_summary.csv` | Summary rows for the main forward evaluation setting. |

**Backward Traceability**

| CSV | What it contains |
| --- | --- |
| `backward_metrics_per_gui_candidate_by_k.csv` | Per-GUI-candidate backward metrics for each k-value and evaluation view. |
| `backward_metrics_summary_by_k.csv` | Aggregate backward metrics for all GUI candidates. |
| `backward_metrics_summary_gold_only_by_k.csv` | Backward metrics restricted to ground-truth-relevant GUI candidates. |
| `backward_per_gui_candidate.csv` | Per-candidate rows for the main backward evaluation setting. |
| `backward_all_candidate_summary.csv` | Backward summary rows for all GUI candidates. |
| `backward_gold_only_diagnostic_summary.csv` | Backward diagnostic rows for ground-truth-relevant candidates only. |
| `backward_output_candidate_summary.csv` | Counts of candidates present in backward ranking outputs. |
| `backward_output_candidate_lengths.csv` | Per-candidate backward ranking length diagnostics. |

**Evaluation Setup and Diagnostics**

| CSV | What it contains |
| --- | --- |
| `candidate_space_summary.csv` | Candidate-space composition used by the descriptive statistics figure. |
| `evaluation_views.csv` | Evaluation-view definitions emitted with the metric outputs. |
| `chart_ground_truth_warnings.csv` | Detailed ground-truth mapping warnings for chart diagnostics. |
| `ground_truth_warning_summary.csv` | Compact warning-count summary. |

**Filtering, Variants, and Scalability**

| CSV | What it contains |
| --- | --- |
| `filtering_delta_summary.csv` | Strict-vs-relaxed filtering delta summaries. |
| `model_variant_summary.csv` | Scenario and model-variant summary rows. |
| `scalability_runtime_memory.csv` | Runtime and memory data extracted from result metadata. |

## Current Figure Outputs

The active final figure folder contains 25 PNG files and 25 PDF files under:

- `Thesis Figures/Backward_GT_Only_Figures/`
- `Thesis Figures/Backward_Traceability_Figures/`
- `Thesis Figures/Descriptive_Statistics/`
- `Thesis Figures/Filtering_Figures/`
- `Thesis Figures/Forward_Traceability_Figures/`
- `Thesis Figures/Forward_VS_Backward_Figures/`
- `Thesis Figures/GuidingLLM_Req_Source_Figures/`
- `Thesis Figures/Scalability_Figures/`

## Evaluation Views

The metric scripts report separate evaluation views so transition evidence is not confused with widget-tree evidence:

- An orphan action is a relevant transition that is not tied to a concrete widget-tree candidate. It is still evaluated as action evidence.
- A resolved widget is a ground-truth widget reference that could be matched to an actual widget candidate in the filtered GUI model.

| Evaluation view | Ranked field | Meaning |
| --- | --- | --- |
| `actions_all` | `TopTransitions` | All relevant transition links, including orphan transitions. |
| `actions_with_linked_widget` | `TopTransitions` | Transition links from rows with `Relevance_Type = Linked` and a widget identifier. |
| `actions_orphan` | `TopTransitions` | Only orphan transitions whose concrete widget ID is not resolved as a widget-tree candidate. |
| `widgets_linked_resolved` | `TopWidgets` | Resolved widget-tree candidates with `Relevance_Type = Linked`. |
| `widgets_functional_resolved` | `TopWidgets` | Resolved widget-tree candidates with `Relevance_Type = Functional`. |
| `widgets_all_resolved` | `TopWidgets` | All resolved widget-tree candidates, both linked and functional. |
| `combined_actions_and_linked_widgets` | `TopMatches` | All relevant transitions, including orphan transitions, together with linked widgets only. |
| `combined_actions_and_widgets` | `TopMatches` | All relevant transitions, including orphan transitions, together with all resolved widget-tree candidates. |
