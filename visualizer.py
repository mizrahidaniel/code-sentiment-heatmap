"""Visualization tools for commit sentiment analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SentimentVisualizer:
    """Generate heatmaps and charts for sentiment data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_style("darkgrid")
        
    def plot_timeline(self, output_path: str = "sentiment_timeline.png"):
        """Plot sentiment over time with moving average."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Sentiment timeline
        ax1.scatter(self.df['date'], self.df['sentiment_score'], 
                   alpha=0.5, s=30, c=self.df['sentiment_score'],
                   cmap='RdYlGn', vmin=-1, vmax=1)
        ax1.plot(self.df['date'], self.df['sentiment_ma'], 
                color='blue', linewidth=2, label='20-commit MA')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Sentiment Score')
        ax1.set_title('Commit Sentiment Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Emotion distribution
        emotion_counts = self.df['emotion'].value_counts()
        colors = {
            'joy': '#2ecc71', 'optimism': '#3498db', 'love': '#e91e63',
            'surprise': '#f39c12', 'neutral': '#95a5a6',
            'fear': '#e67e22', 'sadness': '#34495e', 
            'anger': '#e74c3c', 'disgust': '#c0392b'
        }
        emotion_colors = [colors.get(e.lower(), '#95a5a6') for e in emotion_counts.index]
        ax2.bar(range(len(emotion_counts)), emotion_counts.values, color=emotion_colors)
        ax2.set_xticks(range(len(emotion_counts)))
        ax2.set_xticklabels(emotion_counts.index, rotation=45)
        ax2.set_ylabel('Count')
        ax2.set_title('Emotion Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Timeline saved to {output_path}")
        
    def plot_heatmap(self, output_path: str = "sentiment_heatmap.png"):
        """Generate weekly heatmap of sentiment."""
        # Prepare data
        self.df['week'] = pd.to_datetime(self.df['date']).dt.isocalendar().week
        self.df['year'] = pd.to_datetime(self.df['date']).dt.year
        self.df['weekday'] = pd.to_datetime(self.df['date']).dt.dayofweek
        
        # Create pivot table
        heatmap_data = self.df.pivot_table(
            values='sentiment_score',
            index='weekday',
            columns='week',
            aggfunc='mean'
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, 
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Sentiment'})
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_xlabel('Week of Year')
        ax.set_ylabel('Day of Week')
        ax.set_title('Weekly Sentiment Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
        
    def plot_author_comparison(self, output_path: str = "author_sentiment.png", top_n: int = 10):
        """Compare sentiment across authors."""
        author_stats = self.df.groupby('author').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(3)
        author_stats.columns = ['avg_sentiment', 'std_sentiment', 'commit_count']
        author_stats = author_stats[author_stats['commit_count'] >= 5]  # Min 5 commits
        author_stats = author_stats.sort_values('avg_sentiment', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in author_stats['avg_sentiment']]
        bars = ax.barh(range(len(author_stats)), author_stats['avg_sentiment'], color=colors)
        ax.set_yticks(range(len(author_stats)))
        ax.set_yticklabels(author_stats.index)
        ax.set_xlabel('Average Sentiment Score')
        ax.set_title(f'Top {top_n} Authors by Sentiment')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Author comparison saved to {output_path}")
