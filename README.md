# Ant Attraction Analysis

**Statistical and machine-learning analysis of factors influencing ant attraction to sandwich samples.**

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Usage](#usage)  
4. [Figures](#figures)  
5. [Report](#report)  

---

## Project Overview

This project investigates how bread type, spread topping, and the presence of butter affect the number of ants attracted to a food sample. It combines:

- Descriptive statistics  
- Hypothesis testing (t-tests, ANOVAs)  
- Random Forest regression for feature importance  

---

## Repository Structure

```
ant-attraction-analysis/
├── data/
│   └── ants.csv                 # Cleaned dataset (48 observations)
├── code/
│   └── main.py                  # Analysis pipeline: stats, graphics, tests, RF
├── figures/
│   ├── Distribution of Ant Counts.png
│   ├── Ant Count by Bread Type.png
│   ├── Ant Count by Topping.png
│   ├── Mean Ant Count by Bread Type.png
│   ├── Mean Ant Count by Topping.png
│   ├── Mean Ant Count by Butter Presence.png
│   └── Feature Importances (Random Forest).png
├── report/
│   └── Ant_Attraction_Report.pdf # Final PDF report
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```


## Usage

To reproduce all analyses and figures, run:

```bash
python code/main.py
```

This script will:

- Compute descriptive statistics and print to console  
- Perform t-tests and ANOVAs  
- Fit a Random Forest regressor and output feature importances  
- Save all plots to the `figures/` directory  

---

## Figures

The following graphics illustrate key results:

- **Distribution of Ant Counts**  
- **Ant Count by Bread Type**  
- **Ant Count by Topping**  
- **Mean Ant Count by Bread Type**  
- **Mean Ant Count by Topping**  
- **Mean Ant Count by Butter Presence**  
- **Feature Importances (Random Forest)**  

---

## Report

A comprehensive report with detailed methods, evaluation, and summary is available:

```
report/Report.pdf
```

---
