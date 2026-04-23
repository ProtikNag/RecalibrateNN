# ANOVA Boxplots

A small Python utility for comparing class-based sensitivity scores before and after a transformation using one-way ANOVA and boxplot visualizations.

## Overview

This script reads a CSV file of sensitivity scores, groups rows by class, runs statistical comparisons, and writes both text-based summaries and boxplot images for every ordered before/after feature pair it finds.

It is intended for workflows where you want to measure how sensitivity values change across classes and inspect those differences both numerically and visually.

## Features

- Loads sensitivity data from a CSV file
- Groups samples by Full Class Index
- Detects all columns prefixed with:
  - sensitivityscore_before_
  - sensitivityscore_After_
- Builds all ordered before/after feature combinations
- Runs one-way ANOVA across classes for each selected feature
- Runs per-class ANOVA comparing before vs after values
- Computes descriptive statistics for each class:
  - sample count
  - mean
  - standard deviation
  - median
  - minimum
  - maximum
  - 95% confidence interval
- Saves per-pair text reports
- Saves per-pair boxplot figures
- Creates output folders automatically

## Input Data
The script expects a CSV file containing at least the following:

- Full Class Index
- One or more columns starting with sensitivityscore_before_
- One or more columns starting with sensitivityscore_After_
- Optional column:

- Full filepath
Example columns:


Full Class Indexsensitivityscore_before_layer1sensitivityscore_before_layer2sensitivityscore_After_layer1sensitivityscore_After_layer2
## How It Works
For each ordered pair of detected features:

The script collects all before values per class.
The script collects all after values per class.
It runs ANOVA across classes for the selected before feature.
It runs ANOVA across classes for the selected after feature.
It runs a within-class ANOVA comparing before vs after values.
It computes summary statistics and 95% confidence intervals.
It writes a text report to the Results folder.
It saves a boxplot image to the Images folder.
## Usage
The script currently includes a usage message of the form:


python anova_boxplots.py <input_csv_file>

## Output
The script creates these folders automatically if they do not already exist:


## Images/Results/
Results Reports
For each before/after feature pair, the script generates a text file like:


## Results/<before_feature>_vs_<after_feature>_results.txt
Each report includes:

ANOVA across classes for the before feature
ANOVA across classes for the after feature
Per-class ANOVA comparing before vs after
Descriptive statistics for both groups
95% confidence intervals
Boxplots
For each before/after feature pair, the script generates an image like:


Images/<before_feature>_vs_<after_feature>_boxplots.png
Each figure contains one subplot per class and shows:

## Before distribution
##`After distribution
All class plots for a given feature pair use the same y-axis scale to make comparison easier.

## Statistical Details
The script uses scipy.stats.f_oneway to perform one-way ANOVA in three places:

Across classes for a selected before feature
Across classes for a selected after feature
Within each class for before vs after comparison
It also computes 95% confidence intervals using the Student t distribution.

Notes
Missing values are dropped before analysis.
The Full filepath column is removed if present.
Class order is sorted before reporting and plotting.
If a class has insufficient values for ANOVA, the corresponding result is recorded as nan.

## Limitations
The script assumes valid before and after sensitivity columns are present.
If the input data is incomplete or empty for all feature pairs, output may be limited or fail.
Example Workflow
Prepare a CSV file with class labels and sensitivity score columns.
Place the CSV file in the same working directory as the script.
Run the script.
Open the Results folder to review the statistical summaries.
Open the Images folder to inspect the generated boxplots.

## Requirements

Install the required packages:

```bash
pip install pandas numpy scipy matplotlib 


