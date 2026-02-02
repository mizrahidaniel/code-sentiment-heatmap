#!/usr/bin/env python3
"""CLI for sentiment analysis of Git repositories."""
import argparse
import sys
from pathlib import Path

from sentiment_analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of Git commit messages and generate visualizations"
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to Git repository"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=500,
        help="Maximum number of commits to analyze (default: 500)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for visualizations (default: output/)"
    )
    
    args = parser.parse_args()
    
    # Validate repo path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"❌ Error: Repository not found at {repo_path}")
        sys.exit(1)
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Analyzing repository: {repo_path}")
    print(f"📊 Max commits: {args.max_commits}\n")
    
    # Initialize analyzer
    print("Loading sentiment analysis model...")
    analyzer = SentimentAnalyzer()
    
    # Analyze commits
    print("Analyzing commits...")
    commits = analyzer.analyze_repo(str(repo_path), args.max_commits)
    
    if not commits:
        print("❌ No commits found")
        sys.exit(1)
        
    print(f"✅ Analyzed {len(commits)} commits\n")
    
    # Calculate statistics
    positive = sum(1 for c in commits if c.sentiment == "positive")
    negative = sum(1 for c in commits if c.sentiment == "negative")
    pos_pct = positive / len(commits) * 100
    
    print("📈 Overall Sentiment:")
    print(f"   Positive: {positive} ({pos_pct:.1f}%)")
    print(f"   Negative: {negative} ({100-pos_pct:.1f}%)\n")
    
    # Author analysis
    author_stats = analyzer.analyze_by_author(commits)
    print("👥 Top Authors by Sentiment:")
    sorted_authors = sorted(
        author_stats.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:5]
    for author, stats in sorted_authors:
        print(f"   {author}: {stats['positive_pct']:.1f}% positive ({stats['total']} commits)")
    print()
    
    # Detect burnout zones
    burnout_zones = analyzer.detect_burnout_zones(commits)
    if burnout_zones:
        print("⚠️  Potential Burnout Zones Detected:")
        for date, ratio in burnout_zones[:5]:
            print(f"   {date.strftime('%Y-%m-%d')}: {ratio*100:.1f}% negative")
        print()
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    viz = SentimentVisualizer()
    
    viz.create_heatmap(commits, str(output_dir / "sentiment_heatmap.png"))
    viz.create_timeline(commits, str(output_dir / "sentiment_timeline.png"))
    viz.create_author_comparison(author_stats, str(output_dir / "author_comparison.png"))
    
    print(f"\n✨ Done! Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
