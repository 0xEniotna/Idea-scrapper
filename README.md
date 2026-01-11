# Startup Idea Discovery Tool - Complaint Miner

A Python automation tool for discovering startup/SaaS ideas by mining user complaints across multiple platforms. Based on the proven playbook from Om Patel (@om_patel5).

> "Every complaint is someone saying 'I would pay for this to not suck'" - The internet is literally telling you what to build. You just have to listen.

## The Playbook

This tool automates the discovery process outlined in the viral playbook:

| Source | Strategy | What to Look For |
|--------|----------|------------------|
| **G2/Capterra** | Filter 1-2★ B2B reviews | "doesn't have", "wish it could", "missing", "can't" |
| **Reddit** | Search topic + frustration keywords | "frustrating", "hate when", "wish someone would" |
| **Upwork** | Find repetitive paid tasks | "weekly", "monthly", "ongoing", "repeat" |
| **App Stores** | Read 1★ reviews of top apps | Same complaint 20+ times = validated opportunity |

**Validation Formula:** 30+ mentions + willingness to pay = validated idea

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

```bash
# Copy the template
cp config_template.py config.py

# Edit config.py with your credentials
# At minimum, you need Reddit API keys for full functionality
```

**Getting Reddit API Keys:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" type
4. Copy the `client_id` (under app name) and `client_secret`

### 3. Run the Tool

```bash
# Basic usage - mine complaints about golf
python complaint_miner.py --topic "golf" --sources reddit,google_play

# Mine crypto complaints from all sources
python complaint_miner.py --topic "crypto" --sources all --min_mentions 20

# Focus on B2B software (CRM) with G2 reviews
python complaint_miner.py --topic "CRM software" --sources g2,reddit

# Mobile app opportunities
python complaint_miner.py --topic "meditation apps" --sources google_play,app_store
```

## Usage Examples

### Example 1: Finding B2C Mobile App Opportunities

```bash
python complaint_miner.py --topic "fitness" --sources google_play,reddit --min_mentions 10
```

**Sample Output:**
```
📊 RESULTS SUMMARY
Topic: fitness
Total complaints collected: 247
Sources analyzed: 2
Products mentioned: 15

🔑 Top Pain Point Keywords:
  • "doesn't have": 42 mentions
  • "crashes": 38 mentions
  • "no offline": 31 mentions
  • "too expensive": 28 mentions

💡 GENERATED STARTUP IDEAS

IDEA #1: Fitness Solution: Address 'no offline mode'
Pain Point: lack of essential feature - no offline functionality
Mentions: 31
Sources: google_play, reddit
Target Audience: fitness enthusiasts and gym-goers
Potential: Strong potential ($10-50k MRR)

Description:
Build a focused fitness tool that includes the features users are
desperately asking for. Key opportunity: no offline functionality.
```

### Example 2: B2B SaaS Opportunities

```bash
python complaint_miner.py --topic "project management" --sources g2,reddit --min_mentions 15
```

### Example 3: Automation Opportunities

```bash
python complaint_miner.py --topic "podcast" --sources upwork,reddit
```

Finds repetitive tasks people pay for on Upwork that could be automated into a SaaS product.

## Command Line Options

```
usage: complaint_miner.py [-h] --topic TOPIC [--sources SOURCES]
                         [--min_mentions MIN_MENTIONS] [--output OUTPUT]
                         [--no-export] [--verbose] [--list-topics]

Options:
  --topic, -t       Target topic/niche (e.g., "golf", "crypto", "CRM")
  --sources, -s     Comma-separated sources (default: reddit,google_play)
  --min_mentions, -m  Minimum mentions threshold (default: 5)
  --output, -o      Output directory (default: output)
  --no-export       Skip file export
  --verbose, -v     Enable debug logging
  --list-topics     Show preconfigured topic mappings
```

## Available Sources

| Source | Description | Requirements |
|--------|-------------|--------------|
| `reddit` | Reddit posts and comments | Reddit API credentials |
| `google_play` | Google Play 1-2★ reviews | None (uses google-play-scraper) |
| `app_store` | Apple App Store 1-2★ reviews | None (uses app-store-scraper) |
| `g2` | G2 B2B software reviews | None (web scraping) |
| `upwork` | Upwork job patterns | None (search framework) |
| `all` | All available sources | Reddit API for full functionality |

## Output Files

Results are exported to the `output/` directory:

- `complaints_[topic]_[timestamp].csv` - All collected complaints
- `ideas_[topic]_[timestamp].json` - Generated startup ideas
- `analysis_[topic]_[timestamp].json` - Summary statistics

### CSV Columns

| Column | Description |
|--------|-------------|
| `complaint_text` | The full complaint text |
| `source` | Platform (reddit, google_play, g2, etc.) |
| `topic` | Your specified topic/niche |
| `product_name` | The product being complained about |
| `rating` | Star rating (for app reviews) |
| `date` | When the complaint was posted |
| `upvotes` | Engagement score |
| `category` | Complaint type (missing, pricing, performance, etc.) |
| `keywords_found` | Detected pain point keywords |
| `url` | Link to original complaint |

## Preconfigured Topics

The tool comes with optimized configurations for common niches. Run `--list-topics` to see all:

```bash
python complaint_miner.py --list-topics
```

**Included mappings:**
- **Tech:** crypto, finance, saas, crm, productivity, ai
- **Health:** fitness, meditation, nutrition, mental_health
- **Sports:** golf, sports, gaming
- **Business:** marketing, ecommerce, real_estate, freelance
- **Creative:** podcast, video, design, writing

You can use any topic - these are just optimized presets with relevant subreddits and app searches.

## Pattern Detection

The tool automatically detects complaints using these keyword categories:

- **Missing Features:** "doesn't have", "missing", "no way to", "can't"
- **Wishes:** "wish it could", "wish there was", "should have"
- **Frustration:** "frustrating", "annoying", "hate when", "terrible"
- **Complexity:** "too complex", "confusing", "hard to use"
- **Pricing:** "too expensive", "overpriced", "not worth"
- **Performance:** "slow", "crashes", "buggy", "unreliable"

## Idea Generation Logic

Ideas are generated based on:

1. **Complaint Clustering** - Groups similar complaints by category and keywords
2. **Frequency Analysis** - Prioritizes ideas with 30+ mentions
3. **Pattern Extraction** - Uses NLTK n-grams to find common phrases
4. **Revenue Estimation** - Estimates potential based on mention frequency

**Validation Thresholds:**
- 100+ mentions → High potential ($50k+ MRR achievable)
- 50-99 mentions → Strong potential ($10-50k MRR)
- 20-49 mentions → Moderate potential ($1-10k MRR)
- <20 mentions → Early signal - validate further

## Project Structure

```
Idea-scrapper/
├── complaint_miner.py      # Main CLI tool
├── config_template.py      # Configuration template
├── config.py               # Your API credentials (gitignored)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── output/                 # Generated reports
    ├── complaints_*.csv
    ├── ideas_*.json
    └── analysis_*.json
```

## Ethical Usage

This tool is designed for ethical market research:

- Uses official APIs where available (PRAW for Reddit)
- Implements rate limiting and delays between requests
- Respects robots.txt and terms of service
- Focuses on publicly available information
- Does not collect personal user data

## The 4-Week Execution Plan

Once you find a validated idea:

1. **Week 1:** Validate with 10 potential customers from your data
2. **Week 2:** Build MVP focusing on the core pain point
3. **Week 3:** Launch to the complainers you found
4. **Week 4:** Iterate based on feedback

> **Speed is everything.** When you spot a pattern, act fast - others are seeing the same data.

## Troubleshooting

### Reddit API Errors

```
Reddit API not configured. Set credentials in config.py or environment variables.
```

**Solution:** Set up Reddit API credentials. You can also use environment variables:
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="StartupIdeaMiner/1.0"
```

### No Results Found

- Try broader topics (e.g., "fitness" instead of "home workout equipment")
- Lower the `--min_mentions` threshold
- Use more sources with `--sources all`

### Import Errors

```bash
# Install missing dependencies
pip install praw google-play-scraper app-store-scraper nltk rich
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional data sources (Twitter/X, ProductHunt, Hacker News)
- Improved NLP for complaint clustering
- Sentiment analysis integration
- Web dashboard for results

## License

MIT License - Use freely for finding your next startup idea!

---

**Remember:** Negative reviews = features written by future customers. Stop doomscrolling; start reading what people hate.
