import os
import json
import re
import argparse
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from playwright.async_api import async_playwright, Page, Browser
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
import platform

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON FIELDS THAT ARE NEEDED IN THE FINAL OUTPUT


@dataclass
class AnalysisResult:
    """Data class for storing analysis results"""
    url: str
    site_type: str
    extracted_web_content: str
    content: List[Dict[str, Any]]
    errors: Optional[str] = None

# AVOIDS THE GEMINI API KEY TO GET EXHAUSTED


class RateLimitHandler:
    """Handle API rate limiting and quota management"""

    def __init__(self, max_requests_per_minute=10):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests_timestamps = []
        self.quota_exceeded = False
        self.quota_reset_time = None

    async def wait_if_needed(self):
        """Wait if rate limit is exceeded"""
        current_time = datetime.now()

        if self.quota_exceeded and self.quota_reset_time:
            if current_time < self.quota_reset_time:
                wait_time = (self.quota_reset_time -
                             current_time).total_seconds()
                logger.warning(
                    f"Quota exceeded. Waiting {wait_time:.0f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                self.quota_exceeded = False
                self.quota_reset_time = None

        cutoff_time = current_time - timedelta(minutes=1)
        self.requests_timestamps = [
            ts for ts in self.requests_timestamps if ts > cutoff_time
        ]

        if len(self.requests_timestamps) >= self.max_requests_per_minute:
            oldest_request = min(self.requests_timestamps)
            wait_time = 60 - (current_time - oldest_request).total_seconds()
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
                await asyncio.sleep(wait_time)

    def record_request(self):
        """Record a new API request"""
        self.requests_timestamps.append(datetime.now())

    def handle_quota_error(self, retry_delay_seconds=None):
        """Handle quota exceeded error"""
        self.quota_exceeded = True
        if retry_delay_seconds:
            self.quota_reset_time = datetime.now() + timedelta(seconds=retry_delay_seconds)
        else:
            self.quota_reset_time = datetime.now() + timedelta(hours=1)
        logger.error(
            f"API quota exceeded. Will retry after {self.quota_reset_time}")


class ContentAnalyzer:
    """Main class for web content analysis using Gemini AI"""

    CONTENT_CATEGORIES = [
        "About Us", "Products & Services", "Team", "Blog/Press",
        "Contact", "Careers", "Privacy Policy", "Terms of Service",
        "FAQ", "Support", "Documentation", "Pricing", "News",
        "Events", "Resources", "Portfolio", "Case Studies",
        "Testimonials", "Partners", "Investors"
    ]

    def __init__(self, max_requests_per_minute=5):
        """Initialize the analyzer with rate limiting"""
        self.rate_limiter = RateLimitHandler(max_requests_per_minute)
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.context = None
        self.setup_gemini()

    def setup_gemini(self):
        """Setup Google Gemini AI using LangChain"""
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. Please add it to your .env file or set it as an environment variable.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        logger.info("Gemini AI initialized successfully with LangChain")

    async def validate_url(self, url: str) -> bool:
        """Validate if URL is reachable"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return False

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.head(url) as response:
                        return response.status < 400
                except aiohttp.ClientError:
                    async with session.get(url, headers={'Range': 'bytes=0-1023'}) as response:
                        return response.status < 400
        except Exception as e:
            logger.warning(f"URL validation failed for {url}: {e}")
            return False

    async def start_browser(self):
        """Start Playwright browser with proper cleanup handling"""
        if not self.browser:
            self.playwright = await async_playwright().start()

            browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
            if platform.system() == "Windows":
                browser_args.extend([
                    '--disable-features=VizDisplayCompositor',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ])

            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=browser_args
            )

            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            logger.info("Browser started successfully")

    async def close_browser(self):
        """Close Playwright browser with proper cleanup"""
        error_occurred = False

        try:
            if self.context:
                await self.context.close()
                self.context = None
                logger.info("Browser context closed")
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
            error_occurred = True

        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
                logger.info("Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
            error_occurred = True

        try:
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
                logger.info("Playwright stopped")
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
            error_occurred = True

        if not error_occurred:
            await asyncio.sleep(0.1)

    async def scrape_content(self, url: str) -> Tuple[str, str, List[str]]:
        """
        Scrape content from URL using Playwright
        Returns: (final_url, cleaned_content, internal_links)
        """
        page = None
        try:
            if not self.browser or not self.context:
                await self.start_browser()

            page = await self.context.new_page()

            try:
                response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                if response and response.status >= 400:
                    raise Exception(f"HTTP {response.status}")
            except Exception as e:
                logger.error(f"Failed to navigate to {url}: {e}")
                raise

            final_url = page.url

            await page.wait_for_timeout(2000)

            html_content = await page.content()

            cleaned_content = self.extract_main_content(
                html_content, final_url)
            internal_links = await self.extract_internal_links(page, final_url)

            logger.info(f"Successfully scraped content from {url}")
            return final_url, cleaned_content, internal_links

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise
        finally:
            if page:
                try:
                    await page.close()
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

    def extract_main_content(self, html: str, base_url: str) -> str:
        """Extract main readable content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')

        unwanted_tags = [
            'nav', 'header', 'footer', 'aside', 'script', 'style',
            'advertisement', 'ads', 'popup', 'modal', 'banner', 'noscript'
        ]

        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

        unwanted_selectors = [
            '[class*="ad"]', '[class*="banner"]', '[class*="popup"]',
            '[class*="modal"]', '[id*="ad"]', '[class*="nav"]',
            '[class*="menu"]', '[class*="sidebar"]', '[class*="cookie"]'
        ]

        for selector in unwanted_selectors:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                continue

        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body') or soup

        text_content = main_content.get_text(separator=' ', strip=True)

        text_content = re.sub(r'\s+', ' ', text_content)
        text_content = text_content.strip()

        if len(text_content) > 10000:
            text_content = text_content[:10000] + "..."

        return text_content

    async def extract_internal_links(self, page: Page, base_url: str) -> List[str]:
        """Extract internal links from the page"""
        try:
            links = await page.query_selector_all('a[href]')
            internal_links = []
            base_domain = urlparse(base_url).netloc

            for link in links:
                href = await link.get_attribute('href')
                if href:
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)

                    if (parsed_url.netloc == base_domain or not parsed_url.netloc) and \
                       parsed_url.scheme in ['http', 'https', '']:
                        if parsed_url.netloc:
                            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                        else:
                            clean_url = urljoin(base_url, parsed_url.path)

                        if clean_url not in internal_links and clean_url != base_url:
                            internal_links.append(clean_url)

            return internal_links[:50]

        except Exception as e:
            logger.warning(f"Error extracting internal links: {e}")
            return []

    ####### PROMPT FOR IDENTIFYING CATEGORIES #######
    async def identify_categories(self, content: str) -> List[Dict[str, str]]:
        """Use Gemini via LangChain to identify content categories"""
        if self.rate_limiter.quota_exceeded:
            logger.warning(
                "API quota exceeded, skipping category identification")
            return []

        try:
            await self.rate_limiter.wait_if_needed()

            categories_str = ", ".join(self.CONTENT_CATEGORIES)

            prompt = f"""
            Analyze the following website content and identify which of these predefined categories are clearly present and relevant.
        
            AVAILABLE CATEGORIES: {categories_str}
            
            IDENTIFICATION CRITERIA:
            - Only select categories that have substantial, clear evidence in the content
            - Look for specific keywords, topics, or themes that directly relate to each category
            - Avoid selecting categories based on vague associations or single mentions
            - Categories should represent meaningful content sections, not just passing references
            - If unsure about a category, do not include it
            
            QUALITY THRESHOLDS:
            - The category content should be informative and substantial (not just navigation links)
            - Look for dedicated sections, detailed information, or primary focus areas
            - Avoid categories that are only mentioned in headers, footers, or brief mentions
            
            EXAMPLES OF WHAT TO LOOK FOR:
            - Technology: Detailed tech articles, product reviews, software tutorials, technical specifications
            - Health: Medical advice, health conditions, treatments, wellness information, symptoms
            - Business: Company news, financial information, industry analysis, business strategies
            - Education: Course content, learning materials, academic information, tutorials
            - Entertainment: Reviews, celebrity news, movies, games, recreational content
            
            For each category that meets these criteria, return it in this exact JSON format:
            [{{"category_name": "exact category name from list", "text": ""}}]
            
            IMPORTANT:
            - Use the exact category names from the provided list
            - Only include categories with clear, substantial presence in the content
            - The "text" field should remain empty as specified
            - Return an empty array [] if no categories clearly match
            - Return only the JSON array, no additional text or explanation
            
            Website content:
            {content[:3000]}
            """

            message = HumanMessage(content=prompt)
            self.rate_limiter.record_request()
            response = await asyncio.to_thread(self.llm.invoke, [message])

            response_text = response.content.strip()

            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                categories_json = json.loads(json_match.group())
                return categories_json
            else:
                logger.warning(
                    "Could not extract valid JSON from Gemini response")
                return []

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                retry_delay = None
                if "retry_delay" in error_str:
                    try:
                        delay_match = re.search(r'seconds: (\d+)', error_str)
                        if delay_match:
                            retry_delay = int(delay_match.group(1))
                    except:
                        pass

                self.rate_limiter.handle_quota_error(retry_delay)
                logger.warning(
                    "API quota exceeded, skipping category identification")
                return []
            else:
                logger.error(f"Error identifying categories: {e}")
                return []

    ###### PROMPT FOR IDENTIFYING WHAT KIND OF SITE IT IS ######

    async def identify_site_type(self, content: str) -> str:
        """Use Gemini via LangChain to identify site type"""
        if self.rate_limiter.quota_exceeded:
            logger.warning("API quota exceeded, returning unknown site type")
            return "unknown"

        try:
            await self.rate_limiter.wait_if_needed()

            prompt = f"""
            Analyze the following website content and determine the primary type of website based on its main purpose and functionality.
        
            Return your answer in this exact JSON format:
            {{"site_type": "type here"}}
            
            CLASSIFICATION GUIDELINES:
            
            - **educational**: Online learning platforms (Coursera, Khan Academy, Udemy), universities, schools, training sites
            - **medical/health**: Medical journals, health information sites, medical databases, healthcare providers
            - **research/academic**: Academic databases (PubMed, arXiv), research institutions, scientific journals
            - **news**: News outlets, newspapers, magazines, current events sites
            - **blog**: Personal blogs, opinion sites, lifestyle blogs, individual content creators
            - **e-commerce**: Online stores, shopping sites, product catalogs with purchase functionality
            - **company**: Corporate websites, business homepages, company information sites
            - **government**: Official government sites, public services, regulatory agencies
            - **social media**: Social networks, community platforms, user-generated content hubs
            - **forum**: Discussion boards, Q&A sites, community forums
            - **portfolio**: Personal/professional portfolios, showcase sites, creative work displays
            - **non-profit**: Charitable organizations, foundations, advocacy groups
            
            ANALYSIS CRITERIA:
            1. DO NOT refer any site as "company". ALWAYS use ONLY the classifications listed above - no other categories are permitted
            2. Look for key indicators in the content (course listings, medical terms, research papers, product catalogs, etc.)
            3. Consider the site's primary function and target audience
            4. If multiple types apply, choose the most dominant/primary purpose
            5. Pay attention to domain patterns (.edu, .gov, .org) but prioritize content over domain
            
            EXAMPLES:
            - Coursera → educational (online courses and certifications)
            - Medical News Today → medical/health (health information and medical news)
            - PubMed → medical/health (medical research database)
            
            Return only the JSON object, no additional text.
            
            Website content:
            {content[:3000]}
            """

            message = HumanMessage(content=prompt)
            self.rate_limiter.record_request()
            response = await asyncio.to_thread(self.llm.invoke, [message])

            response_text = response.content.strip()

            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                site_type_json = json.loads(json_match.group())
                return site_type_json.get('site_type', 'unknown')
            else:
                logger.warning(
                    "Could not extract valid JSON from site type response")
                return "unknown"

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                retry_delay = None
                if "retry_delay" in error_str:
                    try:
                        delay_match = re.search(r'seconds: (\d+)', error_str)
                        if delay_match:
                            retry_delay = int(delay_match.group(1))
                    except:
                        pass

                self.rate_limiter.handle_quota_error(retry_delay)
                logger.warning(
                    "API quota exceeded, returning unknown site type")
                return "unknown"
            else:
                logger.error(f"Error identifying site type: {e}")
                return "unknown"

    def extract_category_content(self, content: str, category: str) -> str:
        """Extract content using synonyms for better accuracy."""
        import re
        CATEGORY_KEYWORDS = {
            "About Us": ["about us", "who we are", "our mission", "our story", "what we do", "company overview"],
            "Products & Services": ["products", "services", "what we offer", "solutions", "offerings"],
            "Team": ["our team", "meet the team", "leadership", "our people", "team members", "staff"],
            "Blog/Press": ["blog", "press", "news", "press release", "latest articles", "media"],
            "Contact": ["contact", "get in touch", "reach us", "contact us", "support email", "call us"],
            "Careers": ["careers", "jobs", "we're hiring", "join our team", "work with us", "open positions"],
            "Privacy Policy": ["privacy policy", "your privacy", "data protection", "data privacy", "information policy"],
            "Terms of Service": ["terms of service", "terms and conditions", "legal", "user agreement", "website terms"],
            "FAQ": ["faq", "frequently asked questions", "common questions", "help center", "questions and answers"],
            "Support": ["support", "help", "customer service", "assistance", "technical support"],
            "Documentation": ["documentation", "docs", "developer guide", "user manual", "api reference", "product manual"],
            "Pricing": ["pricing", "plans", "cost", "subscription", "fees", "payment options"],
            "News": ["news", "latest news", "company updates", "press", "announcements"],
            "Events": ["events", "upcoming events", "webinars", "conferences", "meetups", "workshops"],
            "Resources": ["resources", "downloads", "ebooks", "whitepapers", "tools", "guides"],
            "Portfolio": ["portfolio", "our work", "projects", "case studies", "examples"],
            "Case Studies": ["case studies", "customer stories", "success stories", "client stories"],
            "Testimonials": ["testimonials", "reviews", "what our customers say", "feedback", "client testimonials"],
            "Partners": ["partners", "partnerships", "affiliates", "collaborators", "alliances"],
            "Investors": ["investors", "investor relations", "shareholders", "financial reports", "stock info"]
        }

        keywords = CATEGORY_KEYWORDS.get(category, [category.lower()])

        sentences = re.split(r'[.!?]+', content)
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]) + '.'
        return ""

    def match_links_to_category(self, links: List[str], category: str) -> List[str]:
        """Match internal links to categories using basic URL pattern matching"""
        category_lower = category.lower().replace(' & ', '-').replace(' ', '-')
        relevant_links = []

        url_keywords = {
            'about-us': ['about', 'company', 'story'],
            'products-&-services': ['product', 'service', 'solution'],
            'team': ['team', 'staff', 'people'],
            'blog/press': ['blog', 'news', 'press'],
            'contact': ['contact', 'reach'],
            'careers': ['career', 'job', 'hiring'],
            'pricing': ['price', 'pricing', 'plan'],
            'support': ['support', 'help', 'faq'],
        }

        keywords = url_keywords.get(
            category_lower, [category_lower.split('-')[0]])

        for link in links:
            link_lower = link.lower()
            if any(keyword in link_lower for keyword in keywords):
                relevant_links.append(link)

        return relevant_links[:5]

    async def analyze_url(self, url: str) -> AnalysisResult:
        """Analyze a single URL and return structured results"""
        try:
            logger.info(f"Starting analysis of: {url}")

            if not await self.validate_url(url):
                return AnalysisResult(
                    url=url,
                    site_type="unknown",
                    extracted_web_content="",
                    content=[],
                    errors="URL validation failed"
                )

            final_url, content, internal_links = await self.scrape_content(url)

            if not content:
                return AnalysisResult(
                    url=final_url,
                    site_type="unknown",
                    extracted_web_content="",
                    content=[],
                    errors="No content extracted"
                )

            categories = await self.identify_categories(content)
            site_type = await self.identify_site_type(content)

            processed_content = []
            for category_info in categories:
                category_name = category_info.get('category_name', '')
                if category_name:
                    category_text = self.extract_category_content(
                        content, category_name)
                    category_links = self.match_links_to_category(
                        internal_links, category_name)

                    processed_content.append({
                        category_name: {
                            "links": category_links,
                            "text": category_text
                        }
                    })

            return AnalysisResult(
                url=final_url,
                site_type=site_type,
                extracted_web_content=content,
                content=processed_content,
                errors=None
            )

        except Exception as e:
            logger.error(f"Error analyzing {url}: {e}")
            return AnalysisResult(
                url=url,
                site_type="unknown",
                extracted_web_content="",
                content=[],
                errors=str(e)
            )

    async def analyze_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple URLs with proper cleanup"""
        results = []

        try:
            await self.start_browser()

            for i, url in enumerate(urls):
                logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

                result = await self.analyze_url(url)

                result_dict = {
                    "URL": result.url,
                    "site_type": result.site_type,
                    "extracted_web_content": result.extracted_web_content,
                    "content": result.content,
                    "errors": result.errors
                }

                results.append(result_dict)

                if i < len(urls) - 1:
                    await asyncio.sleep(3)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
        finally:
            await self.close_browser()

        return results


def load_urls_from_file(filepath: str) -> List[str]:
    """Load URLs from a file"""
    try:
        with open(filepath, 'r') as f:
            urls = [line.strip() for line in f if line.strip()
                    and not line.strip().startswith('#')]
        return urls
    except Exception as e:
        logger.error(f"Error loading URLs from file: {e}")
        return []

# FOR AVOIDING A RUNTIME ERROR CAUSED IN WINDOWS WHEN THE PROGRAM ENDS BEFORE THE BROWSER GETS CLOSED


def set_event_loop_policy():
    """Set appropriate event loop policy for Windows"""
    if platform.system() == "Windows":
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy())
        import warnings
        warnings.filterwarnings(
            "ignore", message=".*ProactorBasePipeTransport.*")


async def main():
    """Main function with proper event loop handling"""
    set_event_loop_policy()

    parser = argparse.ArgumentParser(
        description='Web Content Analyzer with Gemini AI')
    parser.add_argument('--file', '-f', help='File containing URLs to analyze')
    parser.add_argument('--output', '-o', default='analysis_results.json',
                        help='Output file for results')
    parser.add_argument('--rate-limit', type=int, default=5,
                        help='Max API requests per minute (default: 5)')

    args = parser.parse_args()

    default_urls = [
        # News and Magazine Sites
        "https://www.bbc.com/news",
        "https://www.theguardian.com",

        # Educational and Reference Sites
        "https://www.coursera.org",

        # Technology and Development Blogs
        "https://medium.com",

        # Popular Content-Rich Blogs
        "https://www.searchenginejournal.com",

        # Health & Medical
        "https://www.medicalnewstoday.com",
        "https://pubmed.ncbi.nlm.nih.gov",
    ]

    if args.file:
        urls = load_urls_from_file(args.file)
        if not urls:
            logger.error("No valid URLs found in file")
            return
    else:
        urls = default_urls
        logger.info("Using default URLs for analysis")

    try:
        analyzer = ContentAnalyzer(max_requests_per_minute=args.rate_limit)
        logger.info("Analyzer initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    logger.info(f"Starting analysis of {len(urls)} URLs...")
    results = await analyzer.analyze_urls(urls)

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    logger.info("Analysis complete.")


if __name__ == "__main__":
    set_event_loop_policy()
    asyncio.run(main())
