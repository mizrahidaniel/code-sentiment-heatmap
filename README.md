# Code Sentiment Heatmap 🎨📊

Analyze Git commit messages using NLP/sentiment analysis to visualize team emotional patterns, predict burnout zones, and track morale over time.

## Features

- 🧠 **Sentiment Analysis**: Transformer-based NLP (DistilBERT) on commit messages
- 📈 **Visual Heatmaps**: Calendar heatmap showing emotional patterns over time
- 📉 **Timeline Trends**: Rolling average sentiment with burnout zone detection
- 👥 **Author Analysis**: Compare sentiment patterns across team members
- ⚠️ **Burnout Detection**: ML-based prediction of high-stress periods

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Analyze any Git repository:

```bash
python cli.py /path/to/repo
```

Options:
```bash
python cli.py /path/to/repo \
  --max-commits 1000 \
  --output-dir ./output
```

## Example Output

```
🔍 Analyzing repository: /Users/dev/myproject
📊 Max commits: 500

Loading sentiment analysis model...
Analyzing commits...
✅ Analyzed 487 commits

📈 Overall Sentiment:
   Positive: 312 (64.1%)
   Negative: 175 (35.9%)

👥 Top Authors by Sentiment:
   Alice: 72.3% positive (156 commits)
   Bob: 58.1% positive (98 commits)
   Carol: 81.2% positive (67 commits)

⚠️  Potential Burnout Zones Detected:
   2024-11-15: 75.0% negative
   2024-09-22: 70.0% negative

📊 Generating visualizations...
✅ Heatmap saved to output/sentiment_heatmap.png
✅ Timeline saved to output/sentiment_timeline.png
✅ Author comparison saved to output/author_comparison.png

✨ Done! Visualizations saved to output/
```

## What It Does

**Sentiment analysis** uses Hugging Face Transformers to classify commit messages:
- "feat: add dark mode" → **Positive** (0.92)
- "fix: critical bug in auth" → **Negative** (0.78)
- "refactor: cleanup" → **Positive** (0.65)

**Heatmap visualization** shows sentiment over time (green = positive, red = negative).

**Burnout detection** flags periods with 70%+ negative commits in a 20-commit window.

## Tech Stack

- **NLP**: Hugging Face Transformers (DistilBERT)
- **Visualization**: Matplotlib
- **Git**: GitPython
- **ML**: NumPy (rolling averages, anomaly detection)

## Use Cases

- **Team Health**: Track morale during crunch periods
- **Retrospectives**: Identify when features caused stress
- **Hiring**: Understand team dynamics before joining
- **Self-Reflection**: See your own emotional patterns in code

## Roadmap

- [ ] Interactive dashboard (D3.js)
- [ ] Multi-repo comparison
- [ ] File-level sentiment (which modules cause stress?)
- [ ] Slack/Discord integration (weekly sentiment reports)
- [ ] Fine-tuned model for code-specific language

## Contributing

PRs welcome! Looking for:
- ML/NLP experts (improve sentiment model)
- Data viz designers (better heatmaps)
- Frontend devs (interactive dashboard)

## License

MIT
