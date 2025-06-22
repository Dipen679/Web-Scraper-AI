# Web-Scraper

# Web Content Analyzer with Gemini AI

A Python-based web scraping and content analysis tool that uses Google's Gemini AI to automatically categorize website content, identify site types, and extract relevant information from web pages.

## Features

- **Automated Web Scraping**: Uses Playwright for robust, JavaScript-enabled web scraping
- **AI-Powered Content Analysis**: Leverages Google Gemini AI to identify content categories and site types
- **Rate Limiting**: Built-in API rate limiting to prevent quota exhaustion
- **Batch Processing**: Analyze multiple URLs in sequence with proper delays
- **Site Type Classification**: Identifies website types (educational, medical, news, e-commerce, etc.)
- **Internal Link Extraction**: Finds and associates relevant internal links with content categories

## Installation

### 1. Install Python Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following dependencies:

```txt
langchain-google-genai>=1.0.0
playwright>=1.40.0
aiohttp>=3.9.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

### 2. Install Playwright Browser

After installing the Python packages, you need to install the browser engines:

```bash
# Install Playwright browsers (required for web scraping)
playwright install

# Or install only Chromium (used by this script)
playwright install chromium
```

### 3. Set Up Environment Variables

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env
```

Add your Gemini API key to the `.env` file:

```env
GEMINI_API_KEY=your_actual_api_key_here
```
## Usage

### Basic Usage (Default URLs)

Run the script with default hardcoded URLs:

```bash
python web_analyzer.py
```

### Custom URLs from File
Run with custom URL file:

```bash
python web_analyzer.py --file urls.txt
```

### Advanced Options

```bash
# Custom output file and rate limiting
python web_analyzer.py --file urls.txt --output my_results.json --rate-limit 3
```

### Command Line Arguments

- `--file, -f`: Path to file containing URLs (one per line)
- `--output, -o`: Output JSON file name (default: `analysis_results.json`)
- `--rate-limit`: Maximum API requests per minute (default: 5)

## Expected Output Structure

### Console Output

```
2024-01-15 10:30:45 - INFO - Analyzer initialized successfully
2024-01-15 10:30:45 - INFO - Starting analysis of 7 URLs...
2024-01-15 10:30:45 - INFO - Processing URL 1/7: https://www.bbc.com/news
2024-01-15 10:30:48 - INFO - Successfully scraped content from https://www.bbc.com/news
2024-01-15 10:30:52 - INFO - Processing URL 2/7: https://www.coursera.org
...
================================================================================
ANALYSIS RESULTS
================================================================================
[
  {
    "URL": "https://www.bbc.com/news",
    "site_type": "news",
    "extracted_web_content": "Breaking news from around the world...",
    "content": [
      {
        "News": {
          "links": [
            "https://www.bbc.com/news/world",
            "https://www.bbc.com/news/uk"
          ],
          "text": "Latest breaking news and headlines from BBC News..."
        }
      }
    ],
    "errors": null
  }
]
```

### JSON File Output Structure

```json
[
  {
    "URL": "https://www.example.com",
    "site_type": "educational",
    "extracted_web_content": "Full extracted text content from the webpage...",
    "content": [
      {
        "About Us": {
          "links": [
            "https://www.example.com/about",
            "https://www.example.com/company"
          ],
          "text": "Relevant text content related to About Us section..."
        }
      },
      {
        "Products & Services": {
          "links": [
            "https://www.example.com/products",
            "https://www.example.com/services"
          ],
          "text": "Text describing products and services..."
        }
      }
    ],
    "errors": null
  }
]
```

## Content Categories

The analyzer can identify the following predefined categories:

- About Us
- Products & Services
- Team
- Blog/Press
- Contact
- Careers
- Privacy Policy
- Terms of Service
- FAQ
- Support
- Documentation
- Pricing
- News
- Events
- Resources
- Portfolio
- Case Studies
- Testimonials
- Partners
- Investors

## Site Type Classifications

- **educational**: Learning platforms, universities, training sites
- **medical/health**: Medical journals, health information, healthcare providers
- **research/academic**: Academic databases, research institutions
- **news**: News outlets, newspapers, magazines
- **blog**: Personal blogs, opinion sites, lifestyle content
- **e-commerce**: Online stores, shopping sites
- **government**: Official government sites, public services
- **social media**: Social networks, community platforms
- **forum**: Discussion boards, Q&A sites
- **portfolio**: Personal/professional portfolios
- **non-profit**: Charitable organizations, foundations

## Content Analysis Heuristics

### Content Categorization Method

The tool uses Google Gemini AI with carefully crafted prompts to:

1. **Category Identification**: Analyzes the first 3,000 characters of webpage content
2. **Quality Thresholds**: Only selects categories with substantial, clear evidence
3. **Keyword Matching**: Looks for specific topics and themes directly related to categories
4. **Contextual Analysis**: Considers the overall purpose and focus of the content

**Limitations**:
- Limited to first 3,000 characters of content
- May miss categories buried deep in long pages
- Relies on AI interpretation which can be subjective
- Categories are predefined and may not cover all website types

## Cost Considerations

### API Usage Costs

This tool makes **2 API calls per URL** to Google Gemini:

*  One for **content categorization**
*  One for **site type identification**

---

### Cost Management Tips

* **Monitor Usage**: Use [Google AI Studio](https://makersuite.google.com/) or [Google Cloud Console](https://console.cloud.google.com/) to track your API consumption.
* **Adjust Rate Limits**: Use the `--rate-limit` parameter to control API calls per minute.
* **Batch Processing**: Process URLs in smaller chunks to manage request volume.
* **Content Limits**: Each API call is limited to **3,000 characters** to optimize cost.

---

### Example Cost Calculation (Gemini 2.0 Flash Pricing)

As of 2024:

* **Free Tier**: 1,500 requests/day
  → \~750 URLs/day (since each URL = 2 API calls)
* **Beyond Free Tier**:

  * \~\$0.075 per **1M characters** (input)
  * Each URL = 2 × 3,000 characters = **6,000 characters**
  * **Paid Cost per URL** ≈ **\$0.00045**

| URLs Analyzed | Free Tier | Paid Tier Cost |
| ------------- | --------- | -------------- |
| 100 URLs      |  Free    | \~\$0.045      |
| 500 URLs      |  Free    | \~\$0.225      |
| 750 URLs      |  Free    | \~\$0.3375     |
| 1000 URLs     |          | \~\$0.45       |
| 2000 URLs     |          | \~\$0.90       |

---

### Free Tier Notes

* Resets daily at **midnight Pacific Time (PT)**
* Applies to both **input and output characters**
* Limited to **15 requests per minute**
* No billing setup required to use the free tier

> **Disclaimer**: Pricing and limits are subject to change. Always refer to the [official Google AI pricing page](https://cloud.google.com/vertex-ai/pricing) for the most up-to-date information.

### Content Extraction Process

1. **HTML Parsing**: Uses BeautifulSoup to parse webpage HTML
2. **Content Cleaning**: Removes navigation, ads, scripts, and other non-content elements
3. **Main Content Focus**: Prioritizes `<main>`, `<article>`, and content-specific elements
4. **Text Normalization**: Removes excessive whitespace and limits content to 10,000 characters

**Limitations**:
- May miss content in non-standard HTML structures
- JavaScript-generated content might be missed despite Playwright usage
- Content truncation at 10,000 characters may lose important information

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**
   - Ensure `.env` file exists with correct API key
   - Check file is in same directory as script

2. **Browser Installation Issues**
   - Run `playwright install` after pip installation
   - On Linux, may need additional dependencies: `playwright install-deps`

3. **Rate Limiting Errors**
   - Reduce `--rate-limit` parameter
   - Wait for quota reset (usually 1 hour)
   - Check Google Cloud Console for quota limits
