#!/usr/bin/env python3
"""Command-line interface for Code Sentiment Heatmap."""
import click
from pathlib import Path
from analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer


@click.group()
def cli():
    """Analyze Git commit sentiment and visualize emotional patterns."""
    pass


@cli.command()
@click.option('--repo', '-r', default='.', help='Path to Git repository')
@click.option('--max-commits', '-n', default=500, help='Maximum commits to analyze')
@click.option('--output', '-o', default='output', help='Output directory for visualizations')
def analyze(repo, max_commits, output):
    """Run full sentiment analysis pipeline."""
    click.echo(f"🔍 Analyzing repository: {repo}")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(repo)
    
    # Extract commits
    click.echo(f"📊 Extracting last {max_commits} commits...")
    commits = analyzer.extract_commits(max_count=max_commits)
    click.echo(f"Found {len(commits)} commits")
    
    # Analyze sentiment
    df = analyzer.analyze_sentiment(commits)
    
    # Save data
    csv_path = output_dir / 'sentiment_data.csv'
    df.to_csv(csv_path, index=False)
    click.echo(f"💾 Data saved to {csv_path}")
    
    # Detect burnout signals
    click.echo("\n🔥 Burnout Analysis:")
    signals = analyzer.detect_burnout_signals(df)
    for key, value in signals.items():
        click.echo(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Risk assessment
    risk = signals['burnout_risk']
    risk_level = 'HIGH 🚨' if risk > 70 else 'MEDIUM ⚠️' if risk > 40 else 'LOW ✅'
    click.echo(f"\n  Burnout Risk: {risk}/100 ({risk_level})")
    
    # Generate visualizations
    click.echo("\n📈 Generating visualizations...")
    viz = SentimentVisualizer(df)
    viz.plot_timeline(str(output_dir / 'sentiment_timeline.png'))
    viz.plot_heatmap(str(output_dir / 'sentiment_heatmap.png'))
    viz.plot_author_comparison(str(output_dir / 'author_sentiment.png'))
    
    click.echo(f"\n✨ Analysis complete! Check {output_dir}/ for results")


@cli.command()
@click.option('--repo', '-r', default='.', help='Path to Git repository')
def quick(repo):
    """Quick sentiment summary (no visualizations)."""
    analyzer = SentimentAnalyzer(repo)
    commits = analyzer.extract_commits(max_count=100)
    df = analyzer.analyze_sentiment(commits)
    
    click.echo(f"\n📊 Quick Summary ({len(commits)} commits):")
    click.echo(f"  Average Sentiment: {df['sentiment_score'].mean():.2f}")
    click.echo(f"  Most Common Emotion: {df['emotion'].mode()[0]}")
    click.echo(f"  High-Stress Commits: {len(df[df['emotion'].isin(['anger', 'sadness', 'fear'])])}")


if __name__ == '__main__':
    cli()
