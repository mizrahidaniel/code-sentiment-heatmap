"""Visualization tools for sentiment analysis."""
import calendar
from collections import defaultdict
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sentiment_analyzer import CommitSentiment


class SentimentVisualizer:
    """Generate visualizations for sentiment data."""
    
    def create_heatmap(self, commits: List[CommitSentiment], output_path: str):
        """Create calendar heatmap of sentiment over time."""
        # Group by date
        daily_sentiment = defaultdict(list)
        for c in commits:
            date = c.date.date()
            score = c.score if c.sentiment == "positive" else -c.score
            daily_sentiment[date].append(score)
            
        # Calculate daily averages
        dates = sorted(daily_sentiment.keys())
        scores = [np.mean(daily_sentiment[d]) for d in dates]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Convert to numpy arrays for plotting
        x = [datetime.combine(d, datetime.min.time()) for d in dates]
        y = scores
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(x, [0]*len(x), c=y, cmap="RdYlGn", 
                           s=200, marker="s", vmin=-1, vmax=1)
        
        # Styling
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title("Commit Sentiment Heatmap", fontsize=16, pad=20)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, ha="right")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.1)
        cbar.set_label("Sentiment (Negative ← → Positive)", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Heatmap saved to {output_path}")
        
    def create_timeline(self, commits: List[CommitSentiment], output_path: str):
        """Create timeline showing sentiment trends."""
        dates = [c.date for c in commits]
        scores = [c.score if c.sentiment == "positive" else -c.score for c in commits]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot sentiment over time
        ax.plot(dates, scores, alpha=0.3, color="gray", linewidth=0.5)
        
        # Add rolling average
        window = 20
        if len(scores) >= window:
            rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            rolling_dates = dates[window-1:]
            ax.plot(rolling_dates, rolling_avg, color="blue", linewidth=2, 
                   label=f"{window}-commit moving average")
        
        # Styling
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sentiment Score", fontsize=12)
        ax.set_title("Sentiment Timeline", fontsize=16, pad=20)
        ax.legend()
        ax.grid(alpha=0.2)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Timeline saved to {output_path}")
        
    def create_author_comparison(self, author_stats: dict, output_path: str):
        """Create bar chart comparing sentiment by author."""
        authors = list(author_stats.keys())[:15]  # Top 15 authors
        positive_pcts = [author_stats[a]["positive_pct"] for a in authors]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(authors))
        bars = ax.barh(y_pos, positive_pcts, color="steelblue")
        
        # Color bars based on sentiment
        for i, bar in enumerate(bars):
            if positive_pcts[i] < 40:
                bar.set_color("tomato")
            elif positive_pcts[i] > 60:
                bar.set_color("lightgreen")
                
        ax.set_yticks(y_pos)
        ax.set_yticklabels(authors)
        ax.set_xlabel("Positive Sentiment %", fontsize=12)
        ax.set_title("Sentiment by Author", fontsize=16, pad=20)
        ax.set_xlim(0, 100)
        
        # Add percentage labels
        for i, v in enumerate(positive_pcts):
            ax.text(v + 2, i, f"{v:.1f}%", va="center")
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Author comparison saved to {output_path}")
