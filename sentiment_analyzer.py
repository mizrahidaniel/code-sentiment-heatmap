"""Sentiment analysis for Git commit messages."""
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import git
from transformers import pipeline


@dataclass
class CommitSentiment:
    """Sentiment analysis result for a commit."""
    hash: str
    author: str
    date: datetime
    message: str
    sentiment: str  # positive, negative, neutral
    score: float  # 0.0 to 1.0
    
    
class SentimentAnalyzer:
    """Analyze sentiment of Git commit messages."""
    
    def __init__(self, model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize sentiment analyzer with Hugging Face model."""
        self.classifier = pipeline("sentiment-analysis", model=model)
        
    def analyze_repo(self, repo_path: str, max_commits: int = 1000) -> List[CommitSentiment]:
        """Analyze sentiment of commits in a Git repository."""
        repo = git.Repo(repo_path)
        results = []
        
        for i, commit in enumerate(repo.iter_commits()):
            if i >= max_commits:
                break
                
            # Clean commit message
            msg = self._clean_message(commit.message)
            if not msg:
                continue
                
            # Run sentiment analysis
            result = self.classifier(msg[:512])[0]  # Truncate to model max length
            
            # Map to simpler labels
            sentiment = "positive" if result["label"] == "POSITIVE" else "negative"
            
            results.append(CommitSentiment(
                hash=commit.hexsha[:8],
                author=commit.author.name,
                date=datetime.fromtimestamp(commit.committed_date),
                message=msg,
                sentiment=sentiment,
                score=result["score"]
            ))
            
        return results
    
    def _clean_message(self, msg: str) -> str:
        """Clean and normalize commit message."""
        # Take first line only
        msg = msg.split("\n")[0].strip()
        
        # Remove conventional commit prefixes
        msg = re.sub(r"^(feat|fix|docs|style|refactor|test|chore)(\(.+?\))?:\s*", "", msg)
        
        # Remove ticket numbers
        msg = re.sub(r"#\d+", "", msg)
        msg = re.sub(r"\[.*?\]", "", msg)
        
        return msg.strip()
    
    def analyze_by_author(self, commits: List[CommitSentiment]) -> dict:
        """Calculate average sentiment per author."""
        authors = {}
        for c in commits:
            if c.author not in authors:
                authors[c.author] = {"positive": 0, "negative": 0, "total": 0}
            authors[c.author][c.sentiment] += 1
            authors[c.author]["total"] += 1
            
        # Calculate percentages
        for author, stats in authors.items():
            stats["positive_pct"] = stats["positive"] / stats["total"] * 100
            stats["negative_pct"] = stats["negative"] / stats["total"] * 100
            
        return authors
    
    def detect_burnout_zones(self, commits: List[CommitSentiment], 
                            window_size: int = 20, 
                            threshold: float = 0.7) -> List[Tuple[datetime, float]]:
        """Detect periods with high negative sentiment (potential burnout)."""
        burnout_zones = []
        
        for i in range(len(commits) - window_size):
            window = commits[i:i+window_size]
            neg_ratio = sum(1 for c in window if c.sentiment == "negative") / window_size
            
            if neg_ratio >= threshold:
                avg_date = window[window_size // 2].date
                burnout_zones.append((avg_date, neg_ratio))
                
        return burnout_zones
