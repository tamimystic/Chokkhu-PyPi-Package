# Getting Started

## Installation

Install Chokkhu directly from PyPI via pip:

`ash
pip install --upgrade chokkhu
`

## Requirements

Chokkhu is designed to be lightweight. Its core dependencies are automatically installed:
- 
umpy
- pandas
- scipy
- matplotlib
- seaborn
- opencv-python-headless
- 	qdm

## Basic Philosophy

Chokkhu operates on a functional API design. Instead of instantiating complex classes, you pass your dataset through a series of pure functions. Each function accepts kwargs that allow maximum customizability for advanced users, while providing sensible defaults for beginners.

### The 8 Pillars of Chokkhu

1. ck.load()
2. ck.eda.tabular() / ck.eda.image()
3. ck.clean()
4. ck.preprocess()
5. ck.transform()
6. ck.split()
7. ck.train()
8. ck.evaluate()

Explore the **Core API Reference** on the left menu to learn how to master each pillar.
