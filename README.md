# Code Sentiment Heatmap

ML analysis of team emotional patterns through Git commit messages.

## What It Does

Analyzes Git commit history using NLP sentiment analysis (VADER) to:
- Track emotional patterns over time
- Identify burnout zones (sustained negative sentiment)
- Visualize team morale trends
- Detect toxic patterns in commit messages

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze your repo (last 90 days)
python sentiment_analyzer.py

# Custom time range
python sentiment_analyzer.py --days 180

# Analyze different repo
python sentiment_analyzer.py --repo /path/to/repo --days 30
```

## Output

**Terminal Summary:**
- Total commits analyzed
- Sentiment breakdown (positive/neutral/negative %)
- Average sentiment score
- Most positive/negative commits

**JSON Export:**
- Full commit history with sentiment scores
- Heatmap data (date x author)
- Saved to `sentiment_analysis.json`

## How It Works

1. **Git Log Parsing** - Extracts commit messages via `git log`
2. **VADER Sentiment Analysis** - Offline NLP model (no API needed)
3. **Scoring** - Compound score from -1 (most negative) to +1 (most positive)
4. **Classification** - Positive (>0.05), Neutral (-0.05 to 0.05), Negative (<-0.05)

## Example Output

```
============================================================
  Code Sentiment Analysis
============================================================
Total commits analyzed: 247
  Positive: 89 (36.0%)
  Neutral:  142 (57.5%)
  Negative: 16 (6.5%)

Average sentiment: +0.123

────────────────────────────────────────────────────────────
Most Negative Commits:
  [2026-01-15] Alice
  fix: another stupid bug in the auth flow
  Score: -0.743

  [2026-01-22] Bob
  revert: this entire feature was a mistake
  Score: -0.612

────────────────────────────────────────────────────────────
Most Positive Commits:
  [2026-01-28] Charlie
  feat: amazing new dashboard with real-time updates!
  Score: +0.891

  [2026-02-01] Alice
  ship: MVP complete - users love it!
  Score: +0.923
============================================================
```

## Tech Stack

- **VADER** (Valence Aware Dictionary and sEntiment Reasoner)
  - Rule-based NLP for social media text
  - Works offline (no API calls)
  - Optimized for short, informal text
  - Handles negations, intensifiers, slang

## Why This Matters

**Burnout detection:**
- Sustained negative sentiment = warning sign
- "fix bug", "stupid error", "hate this" patterns

**Team health metrics:**
- Positive sentiment correlates with productivity
- Sudden drops indicate problems

**Toxic patterns:**
- "wtf", "terrible", "stupid" clusters
- Author-specific trends

## Next Steps

- [ ] Visualization (matplotlib heatmap calendar)
- [ ] Slack/Discord alerts for sentiment drops
- [ ] Author-specific trend analysis
- [ ] Word clouds for common phrases
- [ ] Integration with CI/CD metrics

## Philosophy

Commit messages reflect team morale. Track it. Fix it early.

---

**MVP shipped in <2 hours. Code > Architecture.** 🔥
