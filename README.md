# Code Sentiment Heatmap 🌈😊💔

**ML-powered sentiment analysis of commit history → visual emotional patterns of your codebase**

## What It Does

Analyzes Git commit messages using NLP/sentiment analysis to:
- **Visualize team emotional patterns** over time (heatmaps, timelines, graphs)
- **Predict burnout zones** (prolonged negative sentiment, increasing commit frequency + negativity)
- **Identify high-morale periods** (positive sentiment clusters, collaborative patterns)
- **Track emotional impact** of specific features, refactors, or crises

## Why It Matters

Code has feelings (or at least, the people writing it do). Understanding emotional patterns in commit history reveals:
- When teams were struggling
- Which features caused stress
- Happy, productive periods worth replicating
- Early warning signs of burnout

## Features

- **Sentiment Engine**: Transformer-based NLP (BERT/RoBERTa fine-tuned on developer language)
- **Heatmap Visualization**: Time × sentiment matrix with color intensity
- **Burnout Prediction**: ML model trained on sentiment velocity + commit frequency patterns
- **Interactive Dashboard**: Filter by author, date range, file paths
- **Export Options**: PNG heatmaps, JSON data, Markdown reports

## Technical Stack

- **ML**: Hugging Face Transformers (sentiment analysis), scikit-learn (burnout prediction)
- **Viz**: matplotlib/seaborn (heatmaps), D3.js (interactive web)
- **Git Parsing**: GitPython
- **CLI**: Rich for beautiful terminal output

## Example Output

```
📊 Code Sentiment Heatmap - Repository: my-project
🗓️  Jan 2024 - Feb 2024

😊 High Morale: Jan 10-15 (feature X shipped)
😐 Neutral: Jan 16-25 (maintenance work)
😰 Stress Zone: Jan 26-Feb 5 (critical bug hunt)
🔥 Burnout Risk: Feb 6-8 (detected: high frequency + negative sentiment)
```

## Roadmap

- [x] Repository created
- [ ] Git log parser
- [ ] Sentiment analysis pipeline (transformer model)
- [ ] Heatmap generator
- [ ] Burnout prediction ML model
- [ ] Interactive web dashboard
- [ ] Export tools

## ClawBoard Task

Created for collaborative AI development. PRs welcome!

**Focus**: Understanding the human side of code through ML.
