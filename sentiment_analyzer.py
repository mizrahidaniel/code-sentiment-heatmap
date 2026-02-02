#!/usr/bin/env python3
"""
Code Sentiment Heatmap - ML Analysis of Team Emotional Patterns
Analyze Git commit messages using NLP sentiment analysis.
"""

import subprocess
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Using VADER (Valence Aware Dictionary and sEntiment Reasoner)
# Lightweight, no external API, works offline
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing vaderSentiment...")
    subprocess.run(["pip", "install", "-q", "vaderSentiment"], check=True)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_commit_history(repo_path=".", since_days=90):
    """Extract commit messages from Git history."""
    since_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    
    cmd = [
        "git", "-C", repo_path, "log",
        f"--since={since_date}",
        "--pretty=format:%H|%an|%ad|%s",
        "--date=short"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3]
            })
    
    return commits


def analyze_sentiment(commits):
    """Analyze sentiment of commit messages using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    
    for commit in commits:
        scores = analyzer.polarity_scores(commit["message"])
        commit["sentiment"] = {
            "compound": scores["compound"],  # -1 (most negative) to +1 (most positive)
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"]
        }
        
        # Classify sentiment
        if scores["compound"] >= 0.05:
            commit["label"] = "positive"
        elif scores["compound"] <= -0.05:
            commit["label"] = "negative"
        else:
            commit["label"] = "neutral"
    
    return commits


def generate_heatmap_data(commits):
    """Generate heatmap data: date x author sentiment."""
    heatmap = defaultdict(lambda: defaultdict(list))
    
    for commit in commits:
        date = commit["date"]
        author = commit["author"]
        score = commit["sentiment"]["compound"]
        heatmap[date][author].append(score)
    
    # Average scores per day per author
    for date in heatmap:
        for author in heatmap[date]:
            scores = heatmap[date][author]
            heatmap[date][author] = sum(scores) / len(scores)
    
    return heatmap


def print_summary(commits):
    """Print sentiment analysis summary."""
    if not commits:
        print("No commits found in the specified time range.")
        return
    
    total = len(commits)
    positive = sum(1 for c in commits if c["label"] == "positive")
    negative = sum(1 for c in commits if c["label"] == "negative")
    neutral = sum(1 for c in commits if c["label"] == "neutral")
    
    avg_sentiment = sum(c["sentiment"]["compound"] for c in commits) / total
    
    print(f"\n{'='*60}")
    print(f"  Code Sentiment Analysis")
    print(f"{'='*60}")
    print(f"Total commits analyzed: {total}")
    print(f"  Positive: {positive} ({positive/total*100:.1f}%)")
    print(f"  Neutral:  {neutral} ({neutral/total*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/total*100:.1f}%)")
    print(f"\nAverage sentiment: {avg_sentiment:+.3f}")
    
    # Find most positive/negative commits
    sorted_commits = sorted(commits, key=lambda c: c["sentiment"]["compound"])
    
    print(f"\n{'─'*60}")
    print("Most Negative Commits:")
    for commit in sorted_commits[:3]:
        print(f"  [{commit['date']}] {commit['author']}")
        print(f"  {commit['message'][:70]}")
        print(f"  Score: {commit['sentiment']['compound']:+.3f}\n")
    
    print(f"{'─'*60}")
    print("Most Positive Commits:")
    for commit in sorted_commits[-3:]:
        print(f"  [{commit['date']}] {commit['author']}")
        print(f"  {commit['message'][:70]}")
        print(f"  Score: {commit['sentiment']['compound']:+.3f}\n")
    
    print(f"{'='*60}\n")


def save_results(commits, heatmap, output_file="sentiment_analysis.json"):
    """Save results to JSON file."""
    data = {
        "analyzed_at": datetime.now().isoformat(),
        "total_commits": len(commits),
        "commits": commits,
        "heatmap": {
            date: dict(authors) for date, authors in heatmap.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Git commit sentiment")
    parser.add_argument("--repo", default=".", help="Path to Git repository")
    parser.add_argument("--days", type=int, default=90, help="Days of history to analyze")
    parser.add_argument("--output", default="sentiment_analysis.json", help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Analyzing commits from the last {args.days} days...")
    
    commits = get_commit_history(args.repo, args.days)
    commits = analyze_sentiment(commits)
    heatmap = generate_heatmap_data(commits)
    
    print_summary(commits)
    save_results(commits, heatmap, args.output)


if __name__ == "__main__":
    main()
