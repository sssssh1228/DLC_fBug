# Artifact for Paper: Understanding and Detecting Deep Learning Compiler Frontend Bugs of PyTorch 2

This is the artifact for our paper **"Understanding and Detecting Deep Learning Compiler Frontend Bugs of PyTorch 2"** (submitted to TOSEM).

This repository contains the scripts, intermediate data, taxonomy, bug-pattern definitions, and representative generated test cases used in our empirical study of **TorchDynamo frontend bugs**.

The artifact supports the overall analysis workflow from collecting PyTorch issues, characterizing their modeled entities and root causes, extracting recurring bug patterns, and generating pattern-guided test cases.

---

## Overview

The artifact consists of the following stages:

```text
PyTorch GitHub Issues
        │
        ▼
   Issue Collection
        │
        ▼
   Issue Analysis
        │
        ├───────────────┐
        ▼               ▼
Modeled Entities   Root-Cause Taxonomy
        │               │
        └───────┬───────┘
                ▼
         Bug Pattern Extraction
                │
                ▼
      Pattern-Guided Test Generation
                │
                ▼
       Representative Test Cases
                │
                ▼
          Test Execution
```

The artifact is primarily intended to document and support the empirical analysis presented in the paper. Some scripts have been partially masked and are therefore **not intended to be executed directly**.

------

## Contents

```text
.
├── generated_cases_gpt-5.6-sol/   # 300 representative generated test cases
│
├── analyze_issue.py                # Analyze and summarize collected issues
├── draw_pattern.py                 # Visualize bug-pattern distributions
├── example.py                      # Example analysis code
├── fetch_issues.py                 # Collect PyTorch issues
├── gen_testcases.py                # Generate pattern-guided test cases
├── run_testcases.py                # Execute generated test cases
│
├── issues.json                     # Collected raw issues
├── issue_summary_v5_9.json         # Structured issue summaries
├── variableTracker.json            # Definitions of modeled entities
├── taxonomy_v5_1.json              # Root-cause taxonomy
├── pattern_definition.json         # Bug-pattern definitions
│
└── README.md
```

------

## Issue Collection

### `fetch_issues.py`

Collects closed issues from the PyTorch repository that are labeled `module: dynamo` within the study period.

The collected data includes issue metadata, descriptions, comments, labels, and pull requests identified through GitHub Search. The resulting raw dataset is stored in `issues.json`.

The collection criteria used in the study are:

- Repository: `pytorch/pytorch`
- Label: `module: dynamo`
- State: closed
- Creation period: January 1, 2025 – January 31, 2026

The provided `issues.json` represents the dataset snapshot used in the study.

------

## Issue Analysis

### `analyze_issue.py`

Processes the collected GitHub issues and produces structured summaries for subsequent empirical analysis.

The resulting `issue_summary_v5_9.json` provides a normalized representation of individual issues, including information used to characterize their trigger, modeled entity, root cause, and observed symptom.

The analysis combines automated assistance with manual review to improve the consistency of issue characterization.

------

## Modeled Entities

### `variableTracker.json`

Defines the **modeled entities** used in the study to characterize the Python runtime objects and PyTorch data structures handled by Dynamo.

Dynamo represents these runtime objects symbolically through specialized `VariableTracker` classes. The definitions in this file provide the entity-level vocabulary used when analyzing and grouping bugs.

The file serves as the reference for the modeled-entity dimension of the empirical taxonomy.

------

## Root-Cause Taxonomy

### `taxonomy_v5_1.json`

Defines the **root-cause taxonomy** used to classify Dynamo frontend bugs according to the underlying Dynamo mechanism involved.

The taxonomy contains four major root-cause categories:

- **Semantic Modeling**
- **Graph Break**
- **Side Effect**
- **Guard**

Semantic Modeling is further distinguished according to the lifecycle of symbolic objects, including:

- **Variable Construction**
- **Operation Dispatch**

The taxonomy provides the root-cause dimension used together with the modeled-entity definitions to organize and analyze the collected bugs.

------

## Bug Pattern Extraction

### `pattern_definition.json`

Contains the recurring **bug patterns** identified from the classified issues.

The pattern extraction process is statistics-guided and iterative. It identifies concentrated groups of issues based on modeled entities and root-cause categories and then examines recurring defect signatures within these groups.

Each pattern captures a recurring underlying defect behavior and is described using attributes such as its name, trigger, root cause, and symptom.

The resulting pattern definitions form the knowledge base for the pattern-guided testing stage.

------

## Pattern Visualization

### `draw_pattern.py`

Generates visualizations of the identified bug patterns and their distributions across the analyzed Dynamo bugs.

The script is used to produce the pattern-distribution visualizations reported in the empirical analysis.

------

## Pattern-Guided Test Generation

### `gen_testcases.py`

Generates test cases based on the identified bug patterns.

The generation process is **root-cause-aware** and uses the characteristics of individual bug patterns to guide test synthesis. The generated tests are intended to exercise the relevant Dynamo behavior and expose potential frontend defects through observable execution behavior.

The complete generation process is larger than the subset included in this artifact.

------

## Representative Generated Test Cases

### `generated_cases_gpt-5.6-sol/`

This directory contains **300 representative test cases** generated for **15 bug patterns**.

The included patterns exclude the lifecycle-related patterns. We selected these 15 patterns because they provide representative coverage of the recurring Dynamo frontend defect behaviors identified in the study.

The 300 cases correspond to:

```text
15 bug patterns × 20 generated test cases per pattern
= 300 test cases
```

These cases are provided as a representative subset of the generated tests rather than the complete test-generation corpus.

The test cases are intended to illustrate how the extracted bug patterns are translated into concrete test inputs and how pattern-guided generation can be used to explore Dynamo frontend behavior.

------

## Test Execution

### `run_testcases.py`

Provides the test-execution and result-processing workflow for the generated test cases.

The execution stage evaluates the synthesized programs under the target PyTorch/Dynamo environment and checks their observable behavior, including differential behavior between eager and compiled execution where applicable.

The script is included to document the validation workflow; it is not necessarily directly executable in the distributed artifact because parts of the experimental infrastructure have been masked.

------

## Example

### `example.py`

Provides example code illustrating the data structures and analysis workflow used in the artifact.

It is intended primarily as a reference for understanding how the collected data, taxonomy, and pattern definitions are organized.

------

## Data Files

| File                           | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `issues.json`                  | Raw snapshot of the collected PyTorch issues                 |
| `issue_summary_v5_9.json`      | Structured summaries of the analyzed issues                  |
| `variableTracker.json`         | Definitions of modeled entities represented by Dynamo's `VariableTracker` mechanism |
| `taxonomy_v5_1.json`           | Root-cause taxonomy for Dynamo frontend bugs                 |
| `pattern_definition.json`      | Definitions of recurring bug patterns                        |
| `generated_cases_gpt-5.6-sol/` | 300 representative test cases covering 15 bug patterns       |

------

## Relationship to the Study

The artifact separates the empirical analysis into three complementary levels:

### Modeled Entities

`variableTracker.json` defines the vocabulary of runtime objects and PyTorch structures that Dynamo models symbolically.

### Root Causes

`taxonomy_v5_1.json` organizes bugs according to the Dynamo mechanisms responsible for their failures.

### Bug Patterns

`pattern_definition.json` captures recurring combinations of trigger characteristics and underlying defect behaviors within the root-cause taxonomy.

Together, these components support the following analysis:

```text
Modeled Entities + Root Causes
              │
              ▼
        Bug Patterns
              │
              ▼
    Pattern-Guided Testing
```

This organization separates **what Dynamo is modeling**, **where the defect occurs in the compilation process**, and **how recurring defects manifest as reusable bug patterns**.

------

## Reproducibility and Artifact Scope

The files included in this repository represent snapshots and selected artifacts used in the study.

The GitHub-derived data may change if the collection procedure is repeated because issues, comments, labels, and pull-request relationships can evolve over time. Therefore, `issues.json` and the derived JSON files should be regarded as the versions used for the reported analysis.

Likewise, the generated test cases are a representative subset of the complete generation results. The artifact includes 300 cases covering 15 non-lifecycle bug patterns rather than the full set of generated tests.

Some scripts have been partially masked to remove study-specific infrastructure or configuration. Consequently, the scripts should be viewed primarily as documentation of the analysis and testing procedures rather than as standalone executable programs.