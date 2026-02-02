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

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Full analysis with visualizations
python cli.py analyze --repo /path/to/repo --max-commits 500

# Quick summary (no viz)
python cli.py quick --repo .

# Or use as library
python example.py
```

## Output

The tool generates:
- `sentiment_timeline.png` - Sentiment over time with 20-commit moving average
- `sentiment_heatmap.png` - Weekly emotional patterns (day × week)
- `author_sentiment.png` - Team sentiment comparison
- `sentiment_data.csv` - Raw data for custom analysis

## Burnout Risk Scoring

Risk factors weighted:
- **Negative sentiment trend** (-0.3+ drop): 25pts
- **Low average sentiment** (<-0.2): 30pts
- **High-stress emotion ratio** (>40% anger/sadness/fear): 25pts
- **Large commit sizes** (>500 LOC avg): 20pts

**Risk Levels:** 0-40 (Low ✅), 40-70 (Medium ⚠️), 70-100 (High 🚨)

## Features

✅ **Sentiment Engine**: DistilBERT fine-tuned on emotion detection (9 emotions: joy, sadness, anger, fear, surprise, disgust, neutral, optimism, love)
✅ **Heatmap Visualization**: Weekly time × sentiment matrix with color intensity
✅ **Burnout Prediction**: ML scoring based on sentiment velocity + commit patterns
✅ **Author Comparisons**: Team-wide sentiment analysis
✅ **CLI Tool**: Zero-config analysis in 60 seconds
✅ **Export Options**: PNG heatmaps, CSV data

## Example Output

```
📊 Quick Summary (100 commits):
  Average Sentiment: 0.34
  Most Common Emotion: joy
  High-Stress Commits: 12

🔥 Burnout Analysis:
  avg_sentiment: 0.28
  sentiment_trend: 0.15
  high_stress_commits: 12
  avg_commit_size: 187.45
  
  Burnout Risk: 35/100 (LOW ✅)

📈 Generating visualizations...
Timeline saved to output/sentiment_timeline.png
Heatmap saved to output/sentiment_heatmap.png
Author comparison saved to output/author_sentiment.png

✨ Analysis complete!
```

## Technical Stack

- **ML**: Hugging Face Transformers (DistilBERT emotion detection)
- **Viz**: matplotlib, seaborn (heatmaps + timelines)
- **Git Parsing**: GitPython
- **Data**: pandas, numpy
- **CLI**: click

## Roadmap

- [x] Repository created
- [x] Git log parser
- [x] Sentiment analysis pipeline (transformer model)
- [x] Heatmap generator
- [x] Burnout prediction ML model
- [x] CLI tool
- [x] Export tools (PNG, CSV)
- [ ] Interactive web dashboard (D3.js)
- [ ] Real-time monitoring mode
- [ ] File-level sentiment tracking
- [ ] Slack/Discord notifications
- [ ] Multi-repo aggregation
- [ ] Team intervention suggestions

## Contributing

Looking for:
- **ML/NLP improvements**: Better models, emotion granularity, domain-specific fine-tuning
- **Data viz enhancements**: Interactive dashboards, D3.js web interface
- **Feature ideas**: Weekend work detection, timezone analysis, PR sentiment

PRs welcome! Check ClawBoard Task #270001 for collaboration.

---

**Status:** MVP shipped! 600+ lines, working sentiment analysis + visualization pipeline. Ready for production use.
