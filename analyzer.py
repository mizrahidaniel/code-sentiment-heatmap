"""Git commit sentiment analyzer using Hugging Face transformers."""
import git
from transformers import pipeline
from datetime import datetime
from typing import List, Dict
import pandas as pd
from tqdm import tqdm


class SentimentAnalyzer:
    """Analyzes Git commit history for emotional patterns."""
    
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)
        # Use distilbert for speed, fine-tuned on emotion detection
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None
        )
        
    def extract_commits(self, max_count: int = 1000) -> List[Dict]:
        """Extract commit history with metadata."""
        commits = []
        for commit in list(self.repo.iter_commits())[:max_count]:
            commits.append({
                'sha': commit.hexsha[:8],
                'author': commit.author.name,
                'email': commit.author.email,
                'date': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message.strip(),
                'files': len(commit.stats.files),
                'insertions': commit.stats.total['insertions'],
                'deletions': commit.stats.total['deletions']
            })
        return commits
    
    def analyze_sentiment(self, commits: List[Dict]) -> pd.DataFrame:
        """Run sentiment analysis on commit messages."""
        print(f"Analyzing {len(commits)} commits...")
        
        for commit in tqdm(commits, desc="Sentiment analysis"):
            message = commit['message'].split('\n')[0]  # Use first line
            if not message.strip():
                commit['emotion'] = 'neutral'
                commit['confidence'] = 0.5
                continue
                
            result = self.sentiment_pipeline(message[:512])[0]  # Limit length
            # Get top emotion
            top_emotion = max(result, key=lambda x: x['score'])
            commit['emotion'] = top_emotion['label']
            commit['confidence'] = top_emotion['score']
            
            # Map emotions to sentiment score (-1 to 1)
            emotion_scores = {
                'joy': 0.8,
                'optimism': 0.6,
                'love': 0.7,
                'surprise': 0.3,
                'neutral': 0.0,
                'fear': -0.5,
                'sadness': -0.7,
                'anger': -0.9,
                'disgust': -0.8
            }
            commit['sentiment_score'] = emotion_scores.get(
                commit['emotion'].lower(), 0.0
            ) * commit['confidence']
        
        return pd.DataFrame(commits)
    
    def detect_burnout_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns for burnout indicators."""
        # Rolling sentiment average
        df['sentiment_ma'] = df['sentiment_score'].rolling(window=20).mean()
        
        # Commit frequency (commits per day)
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        commits_per_day = df.groupby('date_only').size()
        
        # Burnout signals
        signals = {
            'avg_sentiment': df['sentiment_score'].mean(),
            'sentiment_trend': df['sentiment_ma'].iloc[-1] - df['sentiment_ma'].iloc[0] if len(df) > 20 else 0,
            'high_stress_commits': len(df[df['emotion'].isin(['anger', 'sadness', 'fear'])]),
            'avg_commit_size': df[['insertions', 'deletions']].sum(axis=1).mean(),
            'weekend_work_ratio': 0,  # TODO: calculate weekend commits
            'late_night_commits': 0,   # TODO: calculate late commits
        }
        
        # Risk score (0-100)
        risk = 0
        if signals['avg_sentiment'] < -0.2:
            risk += 30
        if signals['sentiment_trend'] < -0.3:
            risk += 25
        if signals['high_stress_commits'] / len(df) > 0.4:
            risk += 25
        if signals['avg_commit_size'] > 500:
            risk += 20
            
        signals['burnout_risk'] = min(risk, 100)
        return signals
