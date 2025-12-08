# -*- coding: utf-8 -*-
"""
Bluesky Toxicity Analysis - Fixed Version
Analyzes toxicity in Bluesky posts using transformers
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    # from atproto import Client  # Commented out due to pydantic compatibility issue
    ATPROTO_AVAILABLE = False  # Disabled
except ImportError:
    ATPROTO_AVAILABLE = False
    print("⚠️  atproto not installed. Install with: pip install atproto")

# Configuration
OUTPUT_DIR = r'C:\Users\reide\Downloads'
USERNAME = 'sds-hackathon.bsky.social'
PASSWORD = 'buq@AQT1jde0bux3gyr'  # WARNING: Use environment variables for credentials in production!
START_DATE = datetime.date(2024, 2, 6)
END_DATE = datetime.date(2025, 12, 1)  # Updated to 2025
BATCH_SIZE = 64
MAX_WORKERS = 10
ROLLING_WINDOW = 7  # 7-day rolling average for smoother trends
PERSPECTIVE_API_KEY = None  # Set your Perspective API key here if available


class BlueskyToxicityAnalyzer:
    """Analyzes toxicity in Bluesky posts"""
    
    def __init__(self, username: str, password: str, perspective_api_key: Optional[str] = None):
        self.username = username
        self.password = password
        self.access_jwt = None
        self.posts = []
        self.text = []
        self.embed_links = []
        self.toxicity_results = []
        self.perspective_results = []
        self.toxicity_classifier = None
        self.perspective_api_key = perspective_api_key
        self.toxic_keywords = []
        self.atproto_client = None
        
    def load_toxic_keywords(self):
        """Load toxic keywords list for additional analysis"""
        try:
            print("\n📚 Loading toxic keywords list...")
            
            # Try multiple sources for toxic keywords
            urls = [
                'https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en',
                'https://raw.githubusercontent.com/surge-ai/profanity/main/profanity_en.csv',
                'https://raw.githubusercontent.com/coffee-and-fun/google-profanity-words/main/data/en.txt'
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Handle different file formats
                        if url.endswith('.csv'):
                            self.toxic_keywords = [line.split(',')[0].strip() for line in response.text.split('\n') if line.strip()]
                        else:
                            self.toxic_keywords = [word.strip() for word in response.text.split('\n') if word.strip() and not word.startswith('#')]
                        
                        if len(self.toxic_keywords) > 0:
                            print(f"✓ Loaded {len(self.toxic_keywords)} toxic keywords from {url.split('/')[-2]}")
                            return True
                except Exception as e:
                    print(f"  ⚠️  Failed to load from {url.split('/')[-2]}: {e}")
                    continue
            
            # If all sources fail, use a basic fallback list
            print("⚠️  Using fallback keyword list")
            self.toxic_keywords = [
                'abuse', 'attack', 'awful', 'bad', 'crap', 'damn', 'hate', 'hell',
                'idiot', 'jerk', 'kill', 'loser', 'moron', 'pathetic', 'shut up',
                'stupid', 'suck', 'terrible', 'trash', 'ugly', 'worthless'
            ]
            print(f"✓ Loaded {len(self.toxic_keywords)} fallback toxic keywords")
            return True
            
        except Exception as e:
            print(f"✗ Error loading toxic keywords: {e}")
            # Use minimal fallback
            self.toxic_keywords = ['hate', 'stupid', 'idiot', 'awful', 'terrible']
            return False
    
    def authenticate(self) -> bool:
        """Authenticate and get JWT token"""
        try:
            # HTTP authentication for API access
            payload = {
                'identifier': self.username,
                'password': self.password
            }
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(
                'https://bsky.social/xrpc/com.atproto.server.createSession',
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            session_data = response.json()
            self.access_jwt = session_data['accessJwt']
            print("✓ Authentication successful")
            return True
            
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            return False
    
    def fetch_daily_posts(self, current_date: datetime.date) -> List[Dict]:
        """Fetch posts for a single day with retry logic"""
        next_date = current_date + datetime.timedelta(days=1)
        since_str = current_date.isoformat() + 'T00:00:00Z'
        until_str = next_date.isoformat() + 'T00:00:00Z'
        
        params = {
            'q': '*',
            'sort': 'top',
            'lang': 'en',
            'since': since_str,
            'until': until_str,
            'limit': 100
        }
        
        headers = {'Authorization': f'Bearer {self.access_jwt}'}
        
        # Retry logic for transient errors
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    'https://bsky.social/xrpc/app.bsky.feed.searchPosts',
                    params=params,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                return response.json().get('posts', [])
                
            except requests.exceptions.HTTPError as e:
                # Handle specific HTTP errors
                if e.response.status_code in [502, 503, 504]:  # Bad Gateway, Service Unavailable, Gateway Timeout
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    # Last attempt failed, return empty
                    return []
                elif e.response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay * 5)  # Longer wait for rate limit
                        continue
                    return []
                else:
                    # Other HTTP errors, don't retry
                    return []
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                return []
                
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    continue
                return []
                
            except Exception:
                return []
        
        return []
    
    def collect_posts(self, start_date: datetime.date, end_date: datetime.date):
        """Collect posts in parallel for date range"""
        print(f"\n📥 Collecting posts from {start_date} to {end_date}...")
        
        # Generate date list
        date_list = []
        current = start_date
        while current <= end_date:
            date_list.append(current)
            current += datetime.timedelta(days=1)
        
        total_days = len(date_list)
        print(f"   Fetching data for {total_days} days...")
        
        # Parallel fetch with progress tracking
        all_posts = []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_date = {executor.submit(self.fetch_daily_posts, date): date for date in date_list}
                
                # Collect results with progress bar
                completed = 0
                failed_days = []
                for future in concurrent.futures.as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        day_posts = future.result(timeout=35)
                        all_posts.extend(day_posts)
                        completed += 1
                        
                        # Track failed days silently
                        if len(day_posts) == 0:
                            failed_days.append(date)
                        
                        # Show progress every 30 days or at milestones
                        if completed % 30 == 0 or completed == total_days:
                            progress_pct = (completed / total_days) * 100
                            print(f"   Progress: {completed}/{total_days} days ({progress_pct:.1f}%) - {len(all_posts)} posts collected")
                    except concurrent.futures.TimeoutError:
                        completed += 1
                        failed_days.append(date)
                    except Exception as e:
                        completed += 1
                        failed_days.append(date)
                        
        except KeyboardInterrupt:
            print("\n\n⚠️  Collection interrupted by user!")
            print(f"   Collected {len(all_posts)} posts before interruption")
            if len(all_posts) < 100:
                print("   ❌ Not enough data to continue analysis. Exiting...")
                raise
            print("   ✓ Continuing with partial data...\n")
        
        self.posts = all_posts
        
        # Report failed days if any
        if failed_days:
            print(f"\n   ℹ️  Note: {len(failed_days)} days had no data (server errors or no posts)")
            if len(failed_days) <= 10:
                print(f"   Failed dates: {', '.join(str(d) for d in failed_days)}")
        
        # Extract text and embed links
        self.text = []
        self.embed_links = []
        for post in self.posts:
            if 'record' in post and 'text' in post['record']:
                self.text.append(post['record']['text'])
            
            # Generate embed link
            if 'uri' in post:
                embed_link = "https://embed.bsky.app/embed/"
                uri_trim = post['uri'][5:]  # Remove 'at://' prefix
                self.embed_links.append(embed_link + uri_trim)
            else:
                self.embed_links.append('')
        
        print(f"✓ Collected {len(self.posts)} posts with {len(self.text)} texts and {len(self.embed_links)} embed links")
    
    def load_toxicity_model(self):
        """Load the toxicity classification model"""
        try:
            print("\n🤖 Loading toxicity classification model...")
            from transformers import pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            device_name = 'GPU' if device == 0 else 'CPU'
            print(f"Using device: {device_name}")
            
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/unbiased-toxic-roberta",
                truncation=True,
                padding=True,
                device=device
            )
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("💡 Install transformers: pip install transformers torch")
            return False
    
    def classify_toxicity(self, batch_size: int = BATCH_SIZE):
        """Classify toxicity for all posts"""
        if not self.toxicity_classifier:
            print("✗ Model not loaded. Call load_toxicity_model() first")
            return
        
        print(f"\n🔍 Analyzing toxicity for {len(self.text)} posts...")
        self.toxicity_results = []
        
        total = len(self.text)
        for i in range(0, total, batch_size):
            batch = self.text[i:i + batch_size]
            try:
                batch_predictions = self.toxicity_classifier(batch)
                self.toxicity_results.extend(batch_predictions)
                
                # Progress update every ~2000 posts
                if (i + batch_size) % 2048 < batch_size:
                    print(f"  Progress: {min(i + batch_size, total)}/{total} posts")
                    
            except Exception as e:
                print(f"✗ Error at batch {i}: {e}")
                break
        
        print(f"✓ Classification complete: {len(self.toxicity_results)} results")
    
    def classify_with_perspective(self, batch_size: int = 10):
        """Classify toxicity using Google Perspective API (optional)"""
        if not self.perspective_api_key:
            print("⚠️  Perspective API key not provided, skipping Perspective analysis")
            return
        
        print(f"\n🔍 Analyzing toxicity with Perspective API for {len(self.text)} posts...")
        self.perspective_results = []
        
        url = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'
        total = len(self.text)
        
        for i in range(0, total, batch_size):
            batch = self.text[i:i + batch_size]
            
            for text in batch:
                try:
                    data = {
                        'comment': {'text': text},
                        'languages': ['en'],
                        'requestedAttributes': {'TOXICITY': {}}
                    }
                    response = requests.post(
                        url,
                        params={'key': self.perspective_api_key},
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        score = result['attributeScores']['TOXICITY']['summaryScore']['value']
                        self.perspective_results.append({'label': 'toxic' if score > 0.5 else 'nontoxic', 'score': score})
                    else:
                        self.perspective_results.append({'label': 'error', 'score': 0.0})
                except Exception as e:
                    self.perspective_results.append({'label': 'error', 'score': 0.0})
            
            if (i + batch_size) % 1000 < batch_size:
                print(f"  Progress: {min(i + batch_size, total)}/{total} posts")
        
        print(f"✓ Perspective API classification complete: {len(self.perspective_results)} results")
    
    def compare_toxicity_methods(self, df: pd.DataFrame):
        """Compare RoBERTa vs Perspective API toxicity measures"""
        if 'perspective_score' not in df.columns:
            print("⚠️  Perspective API results not available for comparison")
            return
        
        print("\n🔍 Comparing toxicity detection methods...")
        
        # Remove rows with missing data
        df_compare = df[['toxicity_score', 'perspective_score', 'createdAt']].dropna()
        
        if len(df_compare) == 0:
            print("⚠️  No data available for comparison")
            return
        
        # Statistical comparison
        from scipy.stats import pearsonr, spearmanr
        pearson_corr, pearson_p = pearsonr(df_compare['toxicity_score'], df_compare['perspective_score'])
        spearman_corr, spearman_p = spearmanr(df_compare['toxicity_score'], df_compare['perspective_score'])
        
        print(f"\n--- Method Comparison Statistics ---")
        print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.6f})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.6f})")
        
        # Agreement analysis
        threshold = 0.5
        df_compare['roberta_toxic'] = df_compare['toxicity_score'] > threshold
        df_compare['perspective_toxic'] = df_compare['perspective_score'] > threshold
        
        agreement = (df_compare['roberta_toxic'] == df_compare['perspective_toxic']).mean() * 100
        roberta_toxic_pct = df_compare['roberta_toxic'].mean() * 100
        perspective_toxic_pct = df_compare['perspective_toxic'].mean() * 100
        
        print(f"\nRoBERTa toxic posts: {roberta_toxic_pct:.2f}%")
        print(f"Perspective toxic posts: {perspective_toxic_pct:.2f}%")
        print(f"Agreement rate: {agreement:.2f}%")
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plot
        ax1.scatter(df_compare['toxicity_score'], df_compare['perspective_score'], 
                   alpha=0.3, s=10)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
        ax1.set_xlabel('RoBERTa Toxicity Score')
        ax1.set_ylabel('Perspective API Toxicity Score')
        ax1.set_title(f'Method Comparison (r={pearson_corr:.3f})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Time series comparison
        daily_roberta = df_compare.set_index('createdAt')['toxicity_score'].resample('D').mean()
        daily_perspective = df_compare.set_index('createdAt')['perspective_score'].resample('D').mean()
        
        ax2.plot(daily_roberta.index, daily_roberta.values, label='RoBERTa', linewidth=2)
        ax2.plot(daily_perspective.index, daily_perspective.values, label='Perspective API', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average Toxicity Score')
        ax2.set_title('Daily Toxicity Trends by Method', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Distribution comparison
        ax3.hist(df_compare['toxicity_score'], bins=50, alpha=0.5, label='RoBERTa', density=True)
        ax3.hist(df_compare['perspective_score'], bins=50, alpha=0.5, label='Perspective', density=True)
        ax3.set_xlabel('Toxicity Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_compare['roberta_toxic'], df_compare['perspective_toxic'])
        im = ax4.imshow(cm, cmap='Blues', aspect='auto')
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Non-toxic', 'Toxic'])
        ax4.set_yticklabels(['Non-toxic', 'Toxic'])
        ax4.set_xlabel('Perspective API')
        ax4.set_ylabel('RoBERTa')
        ax4.set_title('Agreement Matrix', fontsize=14, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                              ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black',
                              fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'method_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("✓ Saved: method_comparison.png")
        plt.close()
    
    def analyze_keyword_toxicity(self):
        """Analyze toxicity using keyword matching (alternative method)"""
        if not self.toxic_keywords:
            print("⚠️  Toxic keywords not loaded. Call load_toxic_keywords() first")
            return
        
        print(f"\n🔍 Analyzing posts using keyword matching...")
        keyword_toxic_count = 0
        keyword_results = []
        
        for text in self.text:
            if not text:
                keyword_results.append({'is_toxic': False, 'matched_keywords': []})
                continue
            
            text_lower = text.lower()
            matched = [kw for kw in self.toxic_keywords if kw.lower() in text_lower]
            
            is_toxic = len(matched) > 0
            keyword_results.append({
                'is_toxic': is_toxic,
                'matched_keywords': matched,
                'keyword_count': len(matched)
            })
            
            if is_toxic:
                keyword_toxic_count += 1
        
        percent_toxic = (keyword_toxic_count / len(self.text)) * 100 if self.text else 0
        print(f"✓ Keyword analysis complete: {keyword_toxic_count}/{len(self.text)} ({percent_toxic:.2f}%) posts contain toxic keywords")
        
        return keyword_results
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from posts and results"""
        print("\n📊 Creating DataFrame...")
        
        data = []
        min_len = min(len(self.posts), len(self.toxicity_results))
        
        for i in range(min_len):
            post = self.posts[i]
            result = self.toxicity_results[i]
            
            row = {
                'text': post.get('record', {}).get('text'),
                'uri': post.get('uri'),
                'embed_link': self.embed_links[i] if i < len(self.embed_links) else '',
                'handle': post.get('author', {}).get('handle'),
                'createdAt': post.get('record', {}).get('createdAt'),
                'toxicity_label': result.get('label'),
                'toxicity_score': result.get('score')
            }
            
            # Add Perspective API results if available
            if self.perspective_results and i < len(self.perspective_results):
                perspective = self.perspective_results[i]
                row['perspective_label'] = perspective.get('label')
                row['perspective_score'] = perspective.get('score')
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
        df = df.dropna(subset=['createdAt'])
        # Remove timezone info to avoid comparison issues
        df['createdAt'] = df['createdAt'].dt.tz_localize(None)
        df['date'] = df['createdAt'].dt.date
        
        print(f"✓ DataFrame created with {len(df)} rows")
        return df
    
    def analyze_toxicity(self, df: pd.DataFrame, threshold: float = 0.5):
        """Analyze toxicity statistics"""
        print("\n📈 Analyzing toxicity statistics...")
        
        df['is_toxic'] = df['toxicity_score'] > threshold
        percent_toxic = df['is_toxic'].mean() * 100
        
        toxic_posts = df[df['is_toxic']]
        
        print(f"\n--- Analysis Results ---")
        print(f"Total Posts: {len(df)}")
        print(f"Toxic Posts (score > {threshold}): {percent_toxic:.2f}%")
        
        if not toxic_posts.empty:
            print(f"\nMost Common Toxic Labels:")
            print(toxic_posts['toxicity_label'].value_counts().head(5))
            
            print(f"\nTop 5 Authors by Toxic Post Count:")
            print(toxic_posts['handle'].value_counts().head(5))
        else:
            print("No toxic posts found")
        
        return df
    
    def plot_daily_trends(self, df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW, 
                         events: Optional[Dict[str, str]] = None):
        """Plot daily toxicity trends with rolling average and event annotations"""
        print("\n📉 Generating visualizations...")
        
        # Daily percentage of toxic posts
        df_plot = df.set_index('createdAt')
        daily_toxicity = df_plot['is_toxic'].resample('D').mean() * 100
        daily_toxicity_smooth = daily_toxicity.rolling(window=rolling_window, center=True).mean()
        
        plt.figure(figsize=(16, 8))
        
        # Plot raw daily data (lighter)
        plt.plot(daily_toxicity.index, daily_toxicity.values, 
                alpha=0.3, linewidth=1, label='Daily', color='steelblue')
        
        # Plot rolling average (prominent)
        plt.plot(daily_toxicity_smooth.index, daily_toxicity_smooth.values, 
                linewidth=2.5, label=f'{rolling_window}-Day Rolling Average', color='darkblue')
        
        # Add event annotations if provided
        if events:
            for date_str, event_label in events.items():
                event_date = pd.to_datetime(date_str)
                if daily_toxicity.index[0] <= event_date <= daily_toxicity.index[-1]:
                    plt.axvline(x=event_date, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
                    plt.text(event_date, plt.ylim()[1] * 0.95, event_label, 
                            rotation=90, verticalalignment='top', fontsize=9, color='red')
        
        plt.title('Daily Percentage of Toxic Posts on Bluesky (with Smoothing)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Toxic Posts (%)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'daily_toxicity_trend.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def sensitivity_analysis(self, df: pd.DataFrame, k_values: List[int] = [50, 75, 100]):
        """Perform comprehensive sensitivity analysis for different k values"""
        print(f"\n🔬 Performing detailed sensitivity analysis for k={k_values}...")
        
        # Sort and rank posts within each day
        df_sorted = df.sort_values(by=['date', 'createdAt'])
        df_sorted['rank_within_day'] = df_sorted.groupby('date').cumcount() + 1
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Main trend comparison
        ax1 = fig.add_subplot(gs[0, :])
        k_series = {}
        
        for k in k_values:
            df_k = df_sorted[df_sorted['rank_within_day'] <= k]
            daily_avg = df_k.groupby('date')['toxicity_score'].mean()
            k_series[k] = daily_avg
            ax1.plot(daily_avg.index, daily_avg.values, label=f'k={k}', linewidth=2.5, alpha=0.8)
        
        ax1.set_title('Daily Average Toxicity by Post Limit (k) - Sensitivity Analysis', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Average Toxicity Score', fontsize=12)
        ax1.legend(title='Posts per day limit', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.4)
        
        # Statistical comparison
        print("\n--- K-Value Sensitivity Statistics ---")
        stats_data = []
        
        for k in k_values:
            df_k = df_sorted[df_sorted['rank_within_day'] <= k]
            mean_tox = df_k['toxicity_score'].mean()
            std_tox = df_k['toxicity_score'].std()
            median_tox = df_k['toxicity_score'].median()
            toxic_pct = (df_k['toxicity_score'] > 0.5).mean() * 100
            
            stats_data.append({
                'k': k,
                'mean': mean_tox,
                'std': std_tox,
                'median': median_tox,
                'toxic_pct': toxic_pct,
                'n_posts': len(df_k)
            })
            
            print(f"k={k:3d}: Mean={mean_tox:.4f}, Std={std_tox:.4f}, Median={median_tox:.4f}, "
                  f"Toxic%={toxic_pct:.2f}%, N={len(df_k)}")
        
        stats_df = pd.DataFrame(stats_data)
        
        # Distribution comparison
        ax2 = fig.add_subplot(gs[1, 0])
        positions = np.arange(len(k_values))
        for i, k in enumerate(k_values):
            df_k = df_sorted[df_sorted['rank_within_day'] <= k]
            ax2.violinplot([df_k['toxicity_score'].values], positions=[i], widths=0.7,
                          showmeans=True, showmedians=True)
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f'k={k}' for k in k_values])
        ax2.set_title('Toxicity Score Distribution by k', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Toxicity Score')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Correlation heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        # Align series by date for correlation
        aligned_data = pd.DataFrame({f'k={k}': k_series[k] for k in k_values})
        correlation_matrix = aligned_data.corr()
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=0.8, vmax=1.0)
        ax3.set_xticks(np.arange(len(k_values)))
        ax3.set_yticks(np.arange(len(k_values)))
        ax3.set_xticklabels([f'k={k}' for k in k_values])
        ax3.set_yticklabels([f'k={k}' for k in k_values])
        ax3.set_title('Correlation Between k Values', fontsize=12, fontweight='bold')
        
        # Add correlation values
        for i in range(len(k_values)):
            for j in range(len(k_values)):
                text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=10)
        
        fig.colorbar(im, ax=ax3)
        
        # Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for stat in stats_data:
            table_data.append([
                f"k={stat['k']}",
                f"{stat['mean']:.4f}",
                f"{stat['std']:.4f}",
                f"{stat['median']:.4f}",
                f"{stat['toxic_pct']:.2f}%",
                f"{stat['n_posts']:,}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['K Value', 'Mean', 'Std Dev', 'Median', 'Toxic %', 'N Posts'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        output_path = os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        # Robustness check
        print("\n--- Robustness Analysis ---")
        min_corr = correlation_matrix.min().min()
        print(f"Minimum correlation between k values: {min_corr:.4f}")
        if min_corr > 0.90:
            print("✓ Results are highly robust across different k values")
        elif min_corr > 0.80:
            print("✓ Results are moderately robust across different k values")
        else:
            print("⚠️  Results show some variation across k values")
        
        plt.close()
    
    def arima_analysis(self, df: pd.DataFrame):
        """Perform ARIMA modeling on aggregated trends"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from scipy import stats as scipy_stats
            
            print("\n📊 Performing ARIMA time series analysis...")
            
            # Prepare daily aggregated data
            df_plot = df.set_index('createdAt')
            daily_toxicity = df_plot['toxicity_score'].resample('D').mean()
            daily_toxicity = daily_toxicity.dropna()
            
            if len(daily_toxicity) < 30:
                print("⚠️  Not enough data points for ARIMA analysis (need at least 30 days)")
                return
            
            # Fit ARIMA model (p=1, d=1, q=1)
            model = ARIMA(daily_toxicity, order=(1, 1, 1))
            fitted_model = model.fit()
            
            print("\n--- ARIMA Model Summary ---")
            print(fitted_model.summary())
            
            # Forecast next 7 days with confidence intervals
            forecast_result = fitted_model.get_forecast(steps=7)
            forecast = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            # Get fitted values and residuals
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid
            
            # Comprehensive diagnostics plot
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # 1. Forecast plot with confidence intervals
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(daily_toxicity.index, daily_toxicity.values, label='Observed', linewidth=2, color='black')
            ax1.plot(fitted_values.index, fitted_values.values, label='Fitted', linewidth=1.5, 
                    color='blue', alpha=0.7, linestyle='--')
            
            forecast_index = pd.date_range(start=daily_toxicity.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
            ax1.plot(forecast_index, forecast, 'r--', label='7-Day Forecast', linewidth=2.5)
            ax1.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                           color='red', alpha=0.2, label='95% Confidence Interval')
            
            ax1.set_title('ARIMA(1,1,1): Toxicity Trends & Forecast', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Average Toxicity Score', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, linestyle='--', alpha=0.4)
            
            # 2. Residuals plot
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(residuals.index, residuals.values, linewidth=1, color='steelblue')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax2.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Residuals')
            ax2.grid(alpha=0.3)
            
            # 3. Residuals histogram with normal distribution
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            from scipy.stats import norm
            ax3.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
            ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # 4. ACF plot
            ax4 = fig.add_subplot(gs[2, 0])
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, lags=20, ax=ax4, alpha=0.05)
            ax4.set_title('Residual Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)
            
            # 5. Q-Q plot
            ax5 = fig.add_subplot(gs[2, 1])
            from scipy.stats import probplot
            probplot(residuals, dist="norm", plot=ax5)
            ax5.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
            ax5.grid(alpha=0.3)
            
            output_path = os.path.join(OUTPUT_DIR, 'arima_forecast.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
            
            # Model diagnostics
            print("\n--- Model Diagnostics ---")
            print(f"AIC: {fitted_model.aic:.2f}")
            print(f"BIC: {fitted_model.bic:.2f}")
            
            # Ljung-Box test for residual autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            print(f"Ljung-Box test p-value: {lb_test['lb_pvalue'].values[0]:.4f}")
            if lb_test['lb_pvalue'].values[0] > 0.05:
                print("✓ No significant autocorrelation in residuals")
            else:
                print("⚠️  Residuals show autocorrelation")
            
            # Trend analysis
            x = np.arange(len(daily_toxicity))
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, daily_toxicity.values)
            
            print(f"\n--- Trend Analysis ---")
            print(f"Slope: {slope:.8f} (toxicity change per day)")
            print(f"P-value: {p_value:.6f}")
            if p_value < 0.05:
                trend = "increasing" if slope > 0 else "decreasing"
                print(f"✓ Statistically significant {trend} trend detected")
            else:
                print("No statistically significant trend detected")
            
        except ImportError:
            print("⚠️  Install statsmodels for ARIMA analysis: pip install statsmodels")
        except Exception as e:
            print(f"✗ ARIMA analysis failed: {e}")
    
    def analyze_platform_growth(self, df: pd.DataFrame):
        """Analyze correlation between platform growth and toxicity"""
        print("\n📈 Analyzing platform growth effects...")
        
        try:
            df_sorted = df.sort_values('createdAt')
            df_sorted['month'] = df_sorted['createdAt'].dt.to_period('M')
            
            # Monthly metrics
            monthly_stats = df_sorted.groupby('month').agg({
                'toxicity_score': ['mean', 'std'],
                'text': 'count',  # post volume
                'handle': 'nunique'  # unique users
            }).reset_index()
            
            monthly_stats.columns = ['month', 'avg_toxicity', 'std_toxicity', 'post_count', 'unique_users']
            
            # Calculate growth rate
            monthly_stats['user_growth_rate'] = monthly_stats['unique_users'].pct_change() * 100
            monthly_stats['volume_growth_rate'] = monthly_stats['post_count'].pct_change() * 100
            
            # Correlation analysis
            from scipy.stats import pearsonr
            
            valid_data = monthly_stats.dropna()
            if len(valid_data) > 2:
                corr_users, p_users = pearsonr(valid_data['unique_users'], valid_data['avg_toxicity'])
                corr_volume, p_volume = pearsonr(valid_data['post_count'], valid_data['avg_toxicity'])
                
                print(f"\n--- Growth vs Toxicity Correlation ---")
                print(f"User growth correlation: {corr_users:.3f} (p={p_users:.4f})")
                print(f"Volume growth correlation: {corr_volume:.3f} (p={p_volume:.4f})")
                
                if abs(corr_users) > 0.3 and p_users < 0.05:
                    print(f"✓ Significant correlation between user growth and toxicity detected")
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            ax1.plot(monthly_stats['month'].astype(str), monthly_stats['unique_users'], 
                    'o-', linewidth=2, label='Unique Users', color='green')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(monthly_stats['month'].astype(str), monthly_stats['avg_toxicity'], 
                         's--', linewidth=2, label='Avg Toxicity', color='red')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Unique Users', color='green')
            ax1_twin.set_ylabel('Avg Toxicity Score', color='red')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_title('Platform Growth vs Toxicity')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            ax1.grid(alpha=0.3)
            
            ax2.bar(monthly_stats['month'].astype(str), monthly_stats['post_count'], 
                   alpha=0.7, color='steelblue')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Post Volume')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_title('Monthly Post Volume')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, 'platform_growth_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"✗ Growth analysis failed: {e}")
    
    def analyze_user_cohorts(self, df: pd.DataFrame):
        """Analyze early adopters vs mainstream users behavior patterns"""
        print("\n👥 Analyzing user cohort behavior...")
        
        try:
            # Identify when each user first posted
            user_first_post = df.groupby('handle')['createdAt'].min().reset_index()
            user_first_post.columns = ['handle', 'first_post_date']
            
            # Define cohorts based on quartiles
            quartiles = user_first_post['first_post_date'].quantile([0.25, 0.5, 0.75])
            
            def assign_cohort(date):
                if date <= quartiles[0.25]:
                    return 'Early Adopters'
                elif date <= quartiles[0.5]:
                    return 'Early Majority'
                elif date <= quartiles[0.75]:
                    return 'Late Majority'
                else:
                    return 'Laggards'
            
            user_first_post['cohort'] = user_first_post['first_post_date'].apply(assign_cohort)
            
            # Merge back to main dataframe
            df_cohort = df.merge(user_first_post[['handle', 'cohort']], on='handle', how='left')
            
            # Analyze toxicity by cohort
            cohort_stats = df_cohort.groupby('cohort').agg({
                'toxicity_score': ['mean', 'median', 'std'],
                'text': 'count',
                'handle': 'nunique'
            }).reset_index()
            
            print("\n--- Cohort Toxicity Analysis ---")
            print(cohort_stats.to_string())
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            cohort_means = df_cohort.groupby('cohort')['toxicity_score'].mean().sort_values(ascending=False)
            ax1.bar(cohort_means.index, cohort_means.values, color=['#e74c3c', '#e67e22', '#f39c12', '#95a5a6'])
            ax1.set_title('Average Toxicity by User Cohort', fontsize=14)
            ax1.set_ylabel('Average Toxicity Score')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(alpha=0.3, axis='y')
            
            # Toxicity distribution over time by cohort
            for cohort in ['Early Adopters', 'Early Majority', 'Late Majority', 'Laggards']:
                cohort_data = df_cohort[df_cohort['cohort'] == cohort]
                if not cohort_data.empty:
                    daily_avg = cohort_data.set_index('createdAt')['toxicity_score'].resample('W').mean()
                    ax2.plot(daily_avg.index, daily_avg.values, label=cohort, linewidth=2, alpha=0.8)
            
            ax2.set_title('Toxicity Trends by Cohort Over Time', fontsize=14)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Weekly Avg Toxicity Score')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, 'cohort_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"✗ Cohort analysis failed: {e}")
    
    def analyze_event_impact(self, df: pd.DataFrame, events: Dict[str, str]):
        """Analyze toxicity spikes around external events"""
        print("\n🎯 Analyzing event impact on toxicity...")
        
        try:
            results = []
            
            for event_date_str, event_name in events.items():
                event_date = pd.to_datetime(event_date_str)
                
                # Define windows: 7 days before and after
                before_start = event_date - pd.Timedelta(days=7)
                before_end = event_date - pd.Timedelta(days=1)
                after_start = event_date
                after_end = event_date + pd.Timedelta(days=7)
                
                # Filter data
                baseline = df[(df['createdAt'] >= before_start) & (df['createdAt'] <= before_end)]
                event_period = df[(df['createdAt'] >= after_start) & (df['createdAt'] <= after_end)]
                
                if len(baseline) > 0 and len(event_period) > 0:
                    baseline_toxicity = baseline['toxicity_score'].mean()
                    event_toxicity = event_period['toxicity_score'].mean()
                    percent_change = ((event_toxicity - baseline_toxicity) / baseline_toxicity) * 100
                    
                    # Statistical test
                    from scipy.stats import mannwhitneyu
                    statistic, p_value = mannwhitneyu(baseline['toxicity_score'], 
                                                     event_period['toxicity_score'],
                                                     alternative='two-sided')
                    
                    results.append({
                        'event': event_name,
                        'date': event_date_str,
                        'baseline_toxicity': baseline_toxicity,
                        'event_toxicity': event_toxicity,
                        'percent_change': percent_change,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                print("\n--- Event Impact Analysis ---")
                print(results_df.to_string(index=False))
                
                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = ['red' if sig else 'gray' for sig in results_df['significant']]
                bars = ax.bar(results_df['event'], results_df['percent_change'], color=colors, alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_title('Toxicity Change Around Major Events', fontsize=14)
                ax.set_ylabel('% Change in Toxicity')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')
                
                # Add significance markers
                for i, (bar, sig) in enumerate(zip(bars, results_df['significant'])):
                    if sig:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               '*', ha='center', va='bottom', fontsize=20, color='red')
                
                plt.tight_layout()
                output_path = os.path.join(OUTPUT_DIR, 'event_impact_analysis.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {output_path}")
                print("\n* = Statistically significant change (p < 0.05)")
                plt.close()
            
        except Exception as e:
            print(f"✗ Event analysis failed: {e}")
    
    def generate_theoretical_report(self, df: pd.DataFrame):
        """Generate theoretical framework report explaining toxicity mechanisms"""
        print("\n📋 Generating theoretical framework analysis...")
        
        try:
            # Calculate key metrics for theoretical analysis
            df_sorted = df.sort_values('createdAt')
            
            # Early vs late period comparison
            midpoint = df_sorted['createdAt'].quantile(0.5)
            early_period = df_sorted[df_sorted['createdAt'] <= midpoint]
            late_period = df_sorted[df_sorted['createdAt'] > midpoint]
            
            early_toxicity = early_period['toxicity_score'].mean()
            late_toxicity = late_period['toxicity_score'].mean()
            toxicity_change = ((late_toxicity - early_toxicity) / early_toxicity) * 100
            
            early_users = early_period['handle'].nunique()
            late_users = late_period['handle'].nunique()
            user_growth = ((late_users - early_users) / early_users) * 100
            
            # Content report
            report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    THEORETICAL FRAMEWORK ANALYSIS                          ║
║                Explaining Toxicity Dynamics on Bluesky                     ║
╚════════════════════════════════════════════════════════════════════════════╝

1. NETWORK EFFECTS & PLATFORM GROWTH THEORY
   ─────────────────────────────────────────
   
   Finding: {toxicity_change:+.1f}% change in toxicity from early to late period
   User growth: {user_growth:+.1f}% increase in unique users
   
   Theoretical Explanation:
   • Eternal September Effect: As platforms grow, influx of mainstream users
     may dilute early community norms, potentially increasing toxic behavior
   • Network Externalities: Larger user base → more diverse viewpoints → 
     increased potential for conflict and toxic interactions
   • Critical Mass Theory: Platform growth creates conditions for emergence
     of toxic subcultures and echo chambers

2. USER ADOPTION & BEHAVIOR PATTERNS
   ───────────────────────────────────
   
   Theoretical Frameworks:
   • Innovation Diffusion Theory (Rogers, 1962): Different adopter categories
     exhibit distinct behavioral patterns
     - Early Adopters: Tech-savvy, mission-driven, often more civil
     - Late Majority: Diverse motivations, less invested in platform culture
   
   • Social Identity Theory: Early users develop strong in-group identity,
     later users may not internalize community norms as strongly

3. CONTENT MODERATION & GOVERNANCE
   ────────────────────────────────
   
   Mechanisms:
   • Moderation Scaling Challenge: As platform grows, maintaining consistent
     moderation becomes increasingly difficult
   • Decentralized Governance (Bluesky): Federation model may create uneven
     moderation standards across instances
   • Social Proof: Visible toxic content can normalize negative behavior
     (Broken Windows Theory applied to digital spaces)

4. EXTERNAL EVENTS & TEMPORAL DYNAMICS
   ────────────────────────────────────
   
   Observable Patterns:
   • Political Events: Elections, policy announcements trigger heated discourse
   • Platform Migrations: Users fleeing other platforms may bring behavioral
     patterns and grievances
   • Current Events: Breaking news, controversies create engagement spikes
     often accompanied by increased toxicity
   
   Theoretical Basis:
   • Social Movements Theory: Collective action frames drive online mobilization
   • Emotional Contagion: Negative emotions spread rapidly in network structures

5. MECHANISMS EXPLAINING RISING TOXICITY
   ──────────────────────────────────────
   
   Evidence-Based Mechanisms:
   
   a) Disinhibition Effect (Suler, 2004)
      - Online anonymity/pseudonymity reduces social accountability
      - Asynchronous communication allows emotional escalation
   
   b) Outrage Economy
      - Algorithmic amplification of controversial content
      - Attention economy rewards provocative/toxic engagement
   
   c) Polarization Spiral
      - Homophily → echo chambers → radicalization → toxic outgroup behavior
      - Cross-cutting exposure decreases as communities segregate
   
   d) Platform Lifecycle Model
      - Phase 1 (Early): Small, cohesive community, strong norms
      - Phase 2 (Growth): Rapid expansion, norm dilution
      - Phase 3 (Maturity): Competing subcultures, increased conflict

6. IMPLICATIONS & PREDICTIONS
   ──────────────────────────
   
   Based on current trends:
   • Continued growth likely correlates with increased toxicity without
     proportional scaling of moderation resources
   • Decentralized architecture may amplify or mitigate toxicity depending
     on instance-level governance quality
   • Critical intervention points: community guidelines, algorithmic curation,
     user education, and technical features (muting, blocking, filtering)

7. RESEARCH LIMITATIONS
   ────────────────────
   
   • Observational data: Cannot establish causation
   • Selection bias: Top k posts may not represent full platform dynamics
   • Temporal bounds: Analysis limited to {df['createdAt'].min().date()} to {df['createdAt'].max().date()}
   • ML model limitations: Toxicity detection imperfect, culturally situated

8. FUTURE RESEARCH DIRECTIONS
   ──────────────────────────
   
   • Longitudinal user-level analysis tracking behavior changes
   • Network analysis of toxic content spread patterns
   • Cross-platform comparative studies
   • Intervention experiments (A/B testing moderation strategies)
   • Qualitative analysis of toxic content themes and frames

═══════════════════════════════════════════════════════════════════════════════
Key Theoretical References:
• Suler (2004): Online Disinhibition Effect
• Rogers (1962): Diffusion of Innovations
• Tajfel & Turner (1979): Social Identity Theory
• Lessig (1999): Code and Other Laws of Cyberspace
• boyd & Crawford (2012): Critical Questions for Big Data
═══════════════════════════════════════════════════════════════════════════════
"""
            
            # Save report
            output_path = os.path.join(OUTPUT_DIR, 'theoretical_framework.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(report)
            print(f"✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
    
    def export_results(self, df: pd.DataFrame, filename: str = 'toxicity_results.csv'):
        """Export results to CSV"""
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False)
        print(f"✓ Exported results to {output_path}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("Bluesky Toxicity Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BlueskyToxicityAnalyzer(USERNAME, PASSWORD, PERSPECTIVE_API_KEY)
    
    # Authenticate
    if not analyzer.authenticate():
        return
    
    # Load toxic keywords for alternative analysis
    analyzer.load_toxic_keywords()
    
    # Collect posts
    analyzer.collect_posts(START_DATE, END_DATE)
    
    # Load model and classify
    if analyzer.load_toxicity_model():
        analyzer.classify_toxicity()
        
        # Optional: Classify with Perspective API (if key provided)
        if PERSPECTIVE_API_KEY:
            analyzer.classify_with_perspective()
        
        # Optional: Keyword-based toxicity analysis
        keyword_results = analyzer.analyze_keyword_toxicity()
        
        # Create DataFrame and analyze
        df = analyzer.create_dataframe()
        df = analyzer.analyze_toxicity(df)
        
        # Compare methods if Perspective API was used
        if PERSPECTIVE_API_KEY:
            analyzer.compare_toxicity_methods(df)
        
        # Define notable events for annotation (customize as needed)
        events = {
            '2024-11-05': 'US Election',
            '2025-01-20': 'Inauguration',
            # Add more events as needed
        }
        
        # Visualizations
        analyzer.plot_daily_trends(df, events=events)
        analyzer.sensitivity_analysis(df)
        
        # ARIMA time series modeling
        analyzer.arima_analysis(df)
        
        # Advanced research analyses
        analyzer.analyze_platform_growth(df)
        analyzer.analyze_user_cohorts(df)
        analyzer.analyze_event_impact(df, events)
        
        # Generate theoretical framework
        analyzer.generate_theoretical_report(df)
        
        # Export results
        analyzer.export_results(df)
        
        print("\n" + "=" * 60)
        print("✓ Analysis complete!")
        print("=" * 60)
        print("\n📊 Generated outputs:")
        print("   • daily_toxicity_trend.png - Smoothed trend with rolling average & events")
        print("   • sensitivity_analysis.png - Comprehensive k-value comparison (50,75,100)")
        print("   • arima_forecast.png - ARIMA time series with diagnostics & forecast")
        print("   • platform_growth_analysis.png - Network effects & growth correlation")
        print("   • cohort_analysis.png - Early adopters vs late majority patterns")
        print("   • event_impact_analysis.png - External event impact with significance")
        if PERSPECTIVE_API_KEY:
            print("   • method_comparison.png - RoBERTa vs Perspective API comparison")
        print("   • theoretical_framework.txt - Mechanisms & theoretical explanations")
        print("   • toxicity_results.csv - Complete dataset export")


if __name__ == "__main__":
    main()
