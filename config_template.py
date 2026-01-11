"""
Configuration Template for Startup Idea Discovery Tool
=======================================================

SETUP INSTRUCTIONS:
1. Copy this file to 'config.py': cp config_template.py config.py
2. Fill in your API credentials below
3. Never commit config.py to version control (it's in .gitignore)

For Reddit API credentials:
- Go to https://www.reddit.com/prefs/apps
- Click "Create App" or "Create Another App"
- Select "script" type
- Note the client_id (under the app name) and client_secret
"""

# =============================================================================
# REDDIT API CONFIGURATION (Required for Reddit scraping)
# =============================================================================
REDDIT_CONFIG = {
    "client_id": "YOUR_REDDIT_CLIENT_ID",
    "client_secret": "YOUR_REDDIT_CLIENT_SECRET",
    "user_agent": "StartupIdeaMiner/1.0 (by /u/YOUR_USERNAME)",
    "username": "",  # Optional: for authenticated requests
    "password": "",  # Optional: for authenticated requests
}

# =============================================================================
# SCRAPING SETTINGS
# =============================================================================
SCRAPING_CONFIG = {
    # Rate limiting (seconds between requests)
    "request_delay": 2.0,
    "reddit_delay": 1.0,
    "appstore_delay": 1.5,

    # Maximum items to fetch per source
    "max_reddit_posts": 100,
    "max_reddit_comments": 500,
    "max_app_reviews": 200,
    "max_g2_reviews": 100,

    # Request timeout in seconds
    "timeout": 30,

    # Retry settings
    "max_retries": 3,
    "retry_delay": 5,
}

# =============================================================================
# TOPIC-SPECIFIC SUBREDDIT MAPPINGS
# =============================================================================
# Maps topics/niches to relevant subreddits for targeted searching
TOPIC_SUBREDDITS = {
    # Tech & Software
    "crypto": ["cryptocurrency", "bitcoin", "ethereum", "defi", "CryptoMarkets"],
    "finance": ["personalfinance", "investing", "FinancialPlanning", "smallbusiness"],
    "saas": ["SaaS", "startups", "entrepreneur", "smallbusiness"],
    "crm": ["sales", "salesforce", "smallbusiness", "CRM"],
    "productivity": ["productivity", "getdisciplined", "Notion", "ObsidianMD"],
    "ai": ["artificial", "MachineLearning", "ChatGPT", "LocalLLaMA"],

    # Health & Fitness
    "fitness": ["fitness", "loseit", "running", "bodyweightfitness", "gym"],
    "meditation": ["Meditation", "mindfulness", "yoga", "Anxiety"],
    "nutrition": ["nutrition", "EatCheapAndHealthy", "MealPrepSunday"],
    "mental_health": ["mentalhealth", "Anxiety", "depression", "selfimprovement"],

    # Sports & Hobbies
    "golf": ["golf", "GolfSwing", "ProGolf"],
    "sports": ["sports", "fantasyfootball", "sportsbetting"],
    "gaming": ["gaming", "pcgaming", "Games", "IndieGaming"],
    "photography": ["photography", "photocritique", "EditMyRaw"],

    # Business & Marketing
    "marketing": ["marketing", "digital_marketing", "SEO", "PPC"],
    "ecommerce": ["ecommerce", "shopify", "dropship", "FulfillmentByAmazon"],
    "real_estate": ["realestate", "RealEstateInvesting", "FirstTimeHomeBuyer"],
    "freelance": ["freelance", "Upwork", "WorkOnline", "digitalnomad"],

    # Creative
    "podcast": ["podcasting", "podcasts", "audioengineering"],
    "video": ["videography", "YouTubers", "NewTubers", "VideoEditing"],
    "design": ["graphic_design", "web_design", "UXDesign", "UI_Design"],
    "writing": ["writing", "copywriting", "freelanceWriters"],

    # Default fallback subreddits (used if topic not in mapping)
    "_default": ["mildlyinfuriating", "entrepreneur", "Lightbulb", "SomebodyMakeThis"],
}

# =============================================================================
# TOPIC-SPECIFIC APP CATEGORIES (Google Play & App Store)
# =============================================================================
# Maps topics to app store search terms and categories
TOPIC_APP_SEARCHES = {
    "crypto": ["crypto wallet", "bitcoin tracker", "defi", "crypto portfolio"],
    "finance": ["budget tracker", "expense manager", "investment app", "stock trading"],
    "meditation": ["meditation", "mindfulness", "sleep sounds", "breathing exercise"],
    "fitness": ["workout tracker", "fitness app", "gym workout", "home workout"],
    "golf": ["golf gps", "golf swing", "golf scorecard", "golf rangefinder"],
    "productivity": ["todo app", "task manager", "note taking", "habit tracker"],
    "podcast": ["podcast player", "podcast app", "podcast recording"],
    "real_estate": ["real estate", "home buying", "property search", "mortgage calculator"],
    "nutrition": ["calorie counter", "meal planner", "food tracker", "diet app"],
    "photography": ["photo editor", "camera app", "photo filter", "image editing"],
}

# =============================================================================
# G2/CAPTERRA PRODUCT MAPPINGS
# =============================================================================
# Maps topics to popular B2B products to analyze on G2/Capterra
TOPIC_B2B_PRODUCTS = {
    "crm": [
        {"name": "Salesforce", "g2_slug": "salesforce-sales-cloud", "capterra_id": "salesforce"},
        {"name": "HubSpot", "g2_slug": "hubspot-crm", "capterra_id": "hubspot-crm"},
        {"name": "Pipedrive", "g2_slug": "pipedrive", "capterra_id": "pipedrive"},
        {"name": "Zoho CRM", "g2_slug": "zoho-crm", "capterra_id": "zoho-crm"},
    ],
    "productivity": [
        {"name": "Notion", "g2_slug": "notion", "capterra_id": "notion"},
        {"name": "Asana", "g2_slug": "asana", "capterra_id": "asana"},
        {"name": "Monday.com", "g2_slug": "monday-com", "capterra_id": "monday"},
        {"name": "ClickUp", "g2_slug": "clickup", "capterra_id": "clickup"},
    ],
    "marketing": [
        {"name": "Mailchimp", "g2_slug": "mailchimp", "capterra_id": "mailchimp"},
        {"name": "Hootsuite", "g2_slug": "hootsuite", "capterra_id": "hootsuite"},
        {"name": "Buffer", "g2_slug": "buffer", "capterra_id": "buffer"},
        {"name": "Semrush", "g2_slug": "semrush", "capterra_id": "semrush"},
    ],
    "finance": [
        {"name": "QuickBooks", "g2_slug": "quickbooks-online", "capterra_id": "quickbooks"},
        {"name": "Xero", "g2_slug": "xero", "capterra_id": "xero"},
        {"name": "FreshBooks", "g2_slug": "freshbooks", "capterra_id": "freshbooks"},
    ],
    "ecommerce": [
        {"name": "Shopify", "g2_slug": "shopify", "capterra_id": "shopify"},
        {"name": "BigCommerce", "g2_slug": "bigcommerce", "capterra_id": "bigcommerce"},
        {"name": "WooCommerce", "g2_slug": "woocommerce", "capterra_id": "woocommerce"},
    ],
    "ai": [
        {"name": "ChatGPT", "g2_slug": "chatgpt", "capterra_id": "chatgpt"},
        {"name": "Jasper", "g2_slug": "jasper", "capterra_id": "jasper"},
        {"name": "Copy.ai", "g2_slug": "copy-ai", "capterra_id": "copyai"},
    ],
}

# =============================================================================
# UPWORK SEARCH TEMPLATES
# =============================================================================
# Templates for finding repetitive jobs on Upwork by topic
UPWORK_SEARCH_TEMPLATES = {
    "crypto": [
        "crypto analysis weekly",
        "cryptocurrency content monthly",
        "blockchain research ongoing",
        "defi report repeat",
    ],
    "finance": [
        "bookkeeping weekly",
        "financial reporting monthly",
        "invoice processing ongoing",
        "expense tracking repeat",
    ],
    "podcast": [
        "podcast editing weekly",
        "show notes monthly",
        "audio cleanup ongoing",
        "podcast transcription repeat",
    ],
    "marketing": [
        "social media posting weekly",
        "content writing monthly",
        "seo report ongoing",
        "email newsletter repeat",
    ],
    "ecommerce": [
        "product listing weekly",
        "inventory update monthly",
        "order processing ongoing",
        "product photography repeat",
    ],
    "real_estate": [
        "property listing weekly",
        "real estate data entry monthly",
        "mls update ongoing",
        "listing description repeat",
    ],
    "_default": [
        "{topic} weekly",
        "{topic} monthly",
        "{topic} ongoing",
        "{topic} repeat",
    ],
}

# =============================================================================
# COMPLAINT DETECTION PATTERNS
# =============================================================================
# Keywords and phrases that indicate complaints/frustrations
COMPLAINT_KEYWORDS = {
    # Missing features
    "missing": ["doesn't have", "missing", "no way to", "can't", "cannot", "unable to",
                "doesn't support", "lack of", "doesn't include", "no option"],

    # Wishes and desires
    "wishes": ["wish it could", "wish there was", "wish they would", "would be nice if",
               "should have", "needs to have", "hoping for", "want to be able to"],

    # Frustrations
    "frustration": ["frustrating", "annoying", "hate when", "hate that", "drives me crazy",
                    "so annoying", "terrible", "awful", "horrible", "worst"],

    # Complexity/usability issues
    "complexity": ["too complex", "too complicated", "confusing", "hard to use",
                   "not intuitive", "steep learning curve", "overwhelming"],

    # Pricing issues
    "pricing": ["too expensive", "overpriced", "not worth", "costs too much",
                "pricing is", "price increased", "hidden fees"],

    # Performance issues
    "performance": ["slow", "laggy", "crashes", "buggy", "unreliable", "keeps crashing",
                    "doesn't work", "broken", "glitchy"],

    # Support issues
    "support": ["terrible support", "no response", "unhelpful", "bad customer service",
                "never replied", "ignored"],
}

# =============================================================================
# IDEA GENERATION TEMPLATES
# =============================================================================
# Templates for generating startup ideas from complaints
IDEA_TEMPLATES = [
    "Build a {solution_type} that solves '{pain_point}' for {target_audience}",
    "Create a focused tool for {use_case} without the complexity of {existing_product}",
    "Develop an integration layer connecting {product_a} with {product_b}",
    "Launch a {pricing_model} alternative to {expensive_tool} for {target_segment}",
    "Build an offline-first {app_type} with {missing_feature}",
]

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT_CONFIG = {
    # Minimum mentions to consider a complaint significant
    "min_mentions_threshold": 5,

    # Minimum mentions for idea generation
    "idea_generation_threshold": 10,

    # Maximum ideas to generate per run
    "max_ideas": 10,

    # Output file formats
    "export_csv": True,
    "export_json": True,

    # Output directory
    "output_dir": "output",
}
