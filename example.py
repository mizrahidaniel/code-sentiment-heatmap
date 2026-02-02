#!/usr/bin/env python3
"""Example usage of sentiment analysis library."""
from analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer


def main():
    # Initialize analyzer
    print("🔍 Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer(".")
    
    # Extract commits
    print("📊 Extracting commits...")
    commits = analyzer.extract_commits(max_count=100)
    print(f"Found {len(commits)} commits")
    
    # Analyze sentiment
    df = analyzer.analyze_sentiment(commits)
    
    # Show summary
    print("\n📈 Sentiment Summary:")
    print(f"  Average sentiment: {df['sentiment_score'].mean():.2f}")
    print(f"  Most common emotion: {df['emotion'].mode()[0]}")
    
    emotion_dist = df['emotion'].value_counts()
    print("\n🎭 Emotion Distribution:")
    for emotion, count in emotion_dist.head(5).items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    # Burnout analysis
    print("\n🔥 Burnout Analysis:")
    signals = analyzer.detect_burnout_signals(df)
    print(f"  Risk Score: {signals['burnout_risk']}/100")
    print(f"  Sentiment Trend: {signals['sentiment_trend']:.2f}")
    print(f"  High-stress commits: {signals['high_stress_commits']}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    viz = SentimentVisualizer(df)
    viz.plot_timeline("timeline.png")
    viz.plot_heatmap("heatmap.png")
    viz.plot_author_comparison("authors.png")
    
    print("\n✨ Done! Check the .png files")


if __name__ == "__main__":
    main()
