# Artifact for Paper: *Demystifying Deep Learning Compiler Frontend Bugs: An LLM-Aided Empirical Study*

This is the artifact for our paper **"Demystifying Deep Learning Compiler Frontend Bugs: An LLM-Aided Empirical Study"** (submitted to ASE 2026).

------

## Overview

This artifact supports an empirical study of bugs in the frontend of deep learning compilers, specifically focusing on **TorchDynamo** (`torch.compile`), the frontend component of PyTorch's compilation stack. We collect real-world bug reports from GitHub, analyze and categorize them using LLMs, manually validate the results, and generate new test cases for bug detection.

------

## Repository Structure

### Data Files

#### `raw_issue_data.json`

Raw issue data collected from the PyTorch GitHub repository via the GitHub Issues API using the query:

```
is:issue state:closed closed:2024-06-30..2025-06-30 label:"module: dynamo" "Describe the bug" "torch.compile"
```

- **Accessed:** 2025-09-30
- **Total issues:** 194
- Each entry contains the following fields for an issue: `number`, `title`, `labels`, `body`, `comments`

#### `dynamo_mech.json`

A reference file containing the names and definitions of core Dynamo mechanisms (e.g., symbolic execution, guards, graph breaks, side effects). Used as domain knowledge to guide LLM-based analysis.

#### `llm_analysis.json`

Results of the automated LLM-based issue report analysis (corresponding to **Section 4.2** of the paper). After an initial filtering step, 178 issues are retained. Each entry includes:

- `trigger` — What user action or code pattern triggered the bug
- `root_cause` — The Dynamo mechanism responsible
- `symptom` — Observable failure behavior (e.g., crash, timeout, incorrect output)
- `summary` — A brief natural language summary of the bug

Example entry:

```json
{
  "issue_id": 154707,
  "analysis": {
    "trigger": "altering compiler options between invocations",
    "root_cause": "Guards",
    "symptom": "timeout",
    "summary": "Configuration changes induce recompilation without guard attribution, causing repeated compile overhead and confusing empty guard messages."
  }
}
```

#### `detected_bugs.txt`

A list of newly submitted GitHub issues found by executing the generated test cases against the latest version of TorchDynamo.

------

### Analysis Files

#### `issue_analysis_v5.xlsx`

The final manually curated dataset after human review and filtering (123 issues). Each row includes: `issue_id`, `Pull Request`, `Trigger` , `Entity` , `Dynamo Task`, `Symptom` , `Summary` , `Category` 

This file corresponds to **Table 1** and **Table 2** in the paper.

#### `bug_pattern.xlsx`

The final bug taxonomy derived from clustering. Contains the name and definition of each identified bug pattern, organized into **7 root cause categories** and **15 subcategories**.

------

### Scripts

#### `analyze.py`

Performs LLM-aided issue report analysis (Section 4.2). For each issue in `raw_issue_data.json`, it queries the LLM to extract:

- `trigger`
- `root_cause`
- `symptom`
- `summary`

Output is saved to `llm_analysis.json`.

#### `generate.py`

Generates new test cases  (Section 4.3) based on the bug taxonomy in `bug_pattern.xlsx`. Covers all **7 root cause categories** and **15 subcategories**, producing **170 test cases** in total. These test cases are used to detect previously unknown bugs in TorchDynamo.

------

## Methodology Summary

```
GitHub Issues (194), TorchDynamo mechanisms
       │
       ▼ LLM-based filtering & analysis (analyze.py)
llm_analysis.json (178 issues)
       │
       ▼ Categorization
issue_analysis_v5.xlsx (123 issues)
bug_pattern.xlsx (7 categories, 15 subcategories)
       │
       ▼ Test case generation (generate.py)
170 test cases → detected_bugs.txt
```

------

## Citation

If you use this artifact, please cite our paper:

```bibtex
@inproceedings{ase2026dynamo,
  title     = {Demystifying Deep Learning Compiler Frontend Bugs: An LLM-Aided Empirical Study},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  year      = {2026}
}
```