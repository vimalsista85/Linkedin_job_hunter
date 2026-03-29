import os
import random
import time
import re
import json
import smtplib
import ssl
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from curl_cffi import requests as stealth_requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load API keys
load_dotenv()

def get_job_domains():
    """Reads job board domains from websites.txt."""
    try:
        with open("websites.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("websites.txt not found. Using default domains.")
        return ['greenhouse.io', 'lever.co', 'myworkdayjobs.com']

def get_search_queries():
    """Reads configuration from search_queries.txt.
    Returns config dictionary."""
    config = {
        "location": "Dallas, TX",
        "email": None,
        "distance": "25",
        "search_term": "pega",
        "max_jobs": "20",
        "keywords": "pega, pega lead, pega lsa"
    }
    try:
        with open("search_queries.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = line.split(":", 1)
                config[key.strip().lower()] = val.strip()
    except FileNotFoundError:
        print("search_queries.txt not found. Using defaults.")
    
    return config

def load_job_urls():
    """Loads jobs from job_urls.json and prunes those older than 7 days."""
    try:
        if os.path.exists("job_urls.json"):
            with open("job_urls.json", "r", encoding="utf-8") as f:
                jobs = json.load(f)
        else:
            # Migration from old txt if it exists, otherwise empty
            return []
            
        # Pruning logic
        today = date.today()
        seven_days_ago = today - timedelta(days=7)
        
        fresh_jobs = []
        pruned_count = 0
        for job in jobs:
            found_date = datetime.strptime(job['date_found'], '%Y-%m-%d').date()
            if found_date >= seven_days_ago and "/jobs/view/" in job['url']:
                fresh_jobs.append(job)
            else:
                pruned_count += 1
        
        if pruned_count > 0:
            print(f"  [PRUNE] Removed {pruned_count} jobs older than 7 days.")
            save_job_urls(fresh_jobs)
            
        return fresh_jobs
    except Exception as e:
        print(f"  [!] Error loading/pruning job_urls.json: {e}")
        return []

def save_job_urls(jobs):
    """Saves the job list to job_urls.json."""
    with open("job_urls.json", "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=4)

# --- 1. Define the State ---
class JobSearchState(TypedDict):
    search_queries: List[str]
    job_urls: List[str]
    scraped_jobs: List[Dict[str, str]] # {url: ..., text: ...}
    analyzed_jobs: List[Dict[str, str]] # {url: ..., grade: pass/fail, reasoning: ...}
    final_report: str
    errors: List[str]
    search_index: int
    search_exhausted: bool
    location: str
    config: Dict[str, str]

# --- 2. Initialize the LLM ---
# gemini-2.5-flash works well on free tier
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def call_llm_with_retry(messages, max_retries=3):
    """Self-healing LLM call: retries on quota/rate-limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s - safer for free tier
                print(f"  [SELF-HEAL] Rate limit hit. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded for LLM call.")

# --- 3. Define the Nodes (Agents) ---

def scout_node(state: JobSearchState):
    print("--- SCOUT AGENT: Finding LinkedIn jobs via Playwright ---")
    
    try:
        with open("blacklist.json", "r") as f:
            blacklist = json.load(f)
    except FileNotFoundError:
        blacklist = []

    existing_jobs = load_job_urls()
    valid_existing = [j for j in existing_jobs if j['url'] not in blacklist]
    if len(valid_existing) < len(existing_jobs):
        existing_jobs = valid_existing
        save_job_urls(existing_jobs)

    urls = [j['url'] for j in existing_jobs]
    if urls:
        print(f"  [SOURCE] Loaded {len(urls)} fresh jobs from storage.")
        
    search_exhausted = state.get("search_exhausted", False)
    config = state.get('config', {})
    max_jobs = int(config.get('max_jobs', 20))
    target_limit = max_jobs * 2 # Gather double the target to ensure we have enough to filter
    
    print("  [SOURCE] Launching Playwright to scrape LinkedIn Jobs directly...")
    config = state.get('config', {})
    location_query = config.get('location', 'Dallas, Texas').replace(' ', '%20').replace(',', '%2C')
    distance = config.get('distance', '25')
    search_term = config.get('search_term', 'pega').replace(' ', '%20')
    search_url = f"https://www.linkedin.com/jobs/search/?keywords={search_term}&location={location_query}&distance={distance}&f_TPR=r604800"
    
    new_found = 0
    try:
        if len(urls) < target_limit and not search_exhausted:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                print(f"Navigating to {search_url}")
                page.goto(search_url, timeout=45000)
                page.wait_for_timeout(3000)
                
                # Scroll multiple times to load job cards
                for _ in range(8):
                    page.keyboard.press("PageDown")
                    page.wait_for_timeout(1000)
                
                job_links = page.locator('a.base-card__full-link').all()
                if not job_links:
                    job_links = page.locator('a[data-tracking-control-name="public_jobs_jserp-result_search-card"]').all()

                print(f"  [PLAYWRIGHT] Found {len(job_links)} job cards on page.")
                
                for link_elem in job_links:
                    href = link_elem.get_attribute('href')
                    if not href: continue
                    clean_url = href.split('?')[0] 
                    
                    if "/jobs/view/" not in clean_url:
                        continue
                        
                    if clean_url not in urls and clean_url not in blacklist:
                        urls.append(clean_url)
                        title_text = link_elem.text_content().strip() if link_elem.text_content() else "LinkedIn Job"
                        existing_jobs.append({
                            "url": clean_url,
                            "title": title_text,
                            "date_found": str(date.today())
                        })
                        new_found += 1
                        if len(urls) >= target_limit:
                            break
                            
                browser.close()
                
        if new_found > 0:
            print(f"  [MATCH] Added {new_found} new LinkedIn jobs.")
            save_job_urls(existing_jobs)
        else:
            search_exhausted = True
            
    except Exception as e:
        print(f"  [!] Scout error: {e}")
        state.setdefault("errors", []).append(f"Scout error: {e}")

    if len(urls) == 0:
        print(f"  [!] No URLs found on LinkedIn.")
         
    return {"job_urls": urls, "search_index": 1, "search_exhausted": search_exhausted}


def _process_text(url, text, scraped, blacklist, invalid_this_round, method="unknown"):
    """Shared logic to validate scraped text and append or blacklist."""
    # Check multiple patterns LinkedIn uses for expired/closed jobs
    expired_patterns = [
        "No longer accepting applications",
        "No longer accepting",
        "no longer accepting applications",
        "This job is no longer available",
        "this job is no longer available",
        "Applications are closed",
        "applications closed",
    ]
    for pattern in expired_patterns:
        if pattern in text:
            print(f"  [!] Job expired (matched: '{pattern}'). Discarding and Blacklisting.")
            blacklist.append(url)
            invalid_this_round.append(url)
            return False
    scraped.append({"url": url, "text": text[:3000], "method": method})
    print(f"  Success! Extracted {len(text)} chars. (via {method})")
    return True


def _scrape_with_playwright(urls_batch, scraped, blacklist, invalid_this_round):
    """Scrape a batch of URLs using Playwright headless Chromium."""
    print(f"  [PLAYWRIGHT] Scraping {len(urls_batch)} URLs with headless browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for url in urls_batch:
            print(f"Scraping (Playwright) {url}...")
            try:
                page.goto(url, timeout=15000)
                page.wait_for_timeout(2000)  # Let JS render
                text = page.inner_text('body')
                if not text or len(text.strip()) < 50:
                    print("  [!] Page returned minimal content. Blacklisting.")
                    blacklist.append(url)
                    invalid_this_round.append(url)
                    continue
                _process_text(url, text, scraped, blacklist, invalid_this_round, method="Playwright")
            except Exception as e:
                print(f"  [X] Playwright failed: {e}")
                blacklist.append(url)
                invalid_this_round.append(url)
            time.sleep(random.uniform(2, 4))
        browser.close()


def _scrape_with_curl_cffi(urls_batch, scraped, blacklist, invalid_this_round):
    """Scrape a batch of URLs using curl_cffi with Chrome TLS impersonation."""
    print(f"  [CURL_CFFI] Scraping {len(urls_batch)} URLs with stealth TLS...")
    for url in urls_batch:
        print(f"Scraping (curl_cffi) {url}...")
        try:
            response = stealth_requests.get(url, impersonate="chrome", timeout=10)
            if response.status_code != 200:
                print(f"  [!] Encountered {response.status_code}. Blacklisting.")
                blacklist.append(url)
                invalid_this_round.append(url)
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            _process_text(url, text, scraped, blacklist, invalid_this_round, method="curl_cffi")
        except Exception as e:
            print(f"  [X] curl_cffi failed: {e}")
            blacklist.append(url)
            invalid_this_round.append(url)
        time.sleep(random.uniform(2, 5))


def scraper_node(state: JobSearchState):
    print("--- SCRAPER AGENT: Extracting job descriptions (Hybrid Mode) ---")
    urls = state.get("job_urls", [])
    scraped = state.get("scraped_jobs", [])
    already_scraped_urls = {job["url"] for job in scraped}
    
    try:
        with open("blacklist.json", "r") as f:
            blacklist = json.load(f)
    except FileNotFoundError:
        blacklist = []
        
    invalid_this_round = []
    
    # Filter to only URLs we haven't processed yet
    pending_urls = [u for u in urls if u not in already_scraped_urls and u not in blacklist]
    
    if not pending_urls:
        print("  [INFO] No new URLs to scrape.")
    else:
        # Use Playwright for ALL URLs to ensure JS-rendered content
        # (like 'No longer accepting applications') is always detected
        _scrape_with_playwright(pending_urls, scraped, blacklist, invalid_this_round)

    # Save updated blacklist
    with open("blacklist.json", "w") as f:
        json.dump(blacklist, f)
        
    # Remove invalidated jobs from our main storage so Scout can find replacements later
    if invalid_this_round:
        jobs = load_job_urls()
        valid_jobs = [j for j in jobs if j['url'] not in invalid_this_round]
        save_job_urls(valid_jobs)

    return {"scraped_jobs": scraped}


def analyst_node(state: JobSearchState):
    print("--- ANALYST AGENT: Evaluating criteria (Keyword Filter) ---")
    scraped_jobs = state.get("scraped_jobs", [])
    analyzed = []
    
    # Load existing jobs to get their stored titles for Company matching
    all_jobs = load_job_urls()
    url_to_title = {j['url']: j.get('title', 'Unknown Title') for j in all_jobs}
    
    # Required keywords from config
    config = state.get('config', {})
    keywords_str = config.get('keywords', 'pega, pega lead, pega lsa')
    keywords = [k.strip().lower() for k in keywords_str.split(',')]
    location_target = config.get("location", "Not Specified")

    # LLM evaluation commented out per user request
    # system_prompt = """..."""

    for job in scraped_jobs:
        text = job.get('text', '').lower()
        url = job["url"]
        
        if not text:
             continue
             
        # 0. Expired Check
        if "no longer accepting applications" in text or "applications are closed" in text:
             print(f"  [X] Expired Application {url[:50]}")
             analyzed.append({"url": url, "grade": "FAIL", "reasoning": "Job expired.", "method": job.get("method", "unknown")})
             continue
             
        # 1. Check for Pega Keywords
        if not any(kw in text for kw in keywords):
            print(f"  [X] Filtered {url[:50]}... missing Pega keywords")
            analyzed.append({
                "url": url,
                "grade": "FAIL",
                "reasoning": "Missing required Pega keywords.",
                "method": job.get("method", "unknown")
            })
            continue
            
        # 2. Extract Data (No LLM)
        title = url_to_title.get(url, 'Unknown Title')
        
        # Company Extraction heuristic from Title
        company = "Not Found"
        if ' at ' in title:
            company = title.split(' at ')[-1].split(' | ')[0].split(' - ')[0].strip()
        elif ' - ' in title:
            parts = title.split(' - ')
            if len(parts) > 1:
                company = parts[1].strip()
        
        # Salary Extraction ($100k, $100,000, etc.)
        salary = "Not Specified"
        salary_match = re.search(r'\$[0-9,]+(?:[kK]|\.[0-9]{2})?(?:\s*(?:to|-)\s*\$[0-9,]+(?:[kK]|\.[0-9]{2})?)?', text)
        if salary_match:
            salary = salary_match.group(0)

        print(f"  [+] Passed {url[:50]}... (Company: {company}, Salary: {salary})")
        analyzed.append({
            "url": url,
            "grade": "PASS",
            "reasoning": "Pega keywords found.",
            "company": company,
            "salary": salary,
            "location": location_target,
            "method": job.get("method", "unknown")
        })
            
    return {"analyzed_jobs": analyzed}


def manager_node(state: JobSearchState):
    print("--- MANAGER AGENT: QA and Formatting ---")
    analyzed = state.get("analyzed_jobs", [])
    config = state.get('config', {})
    max_jobs = int(config.get('max_jobs', 20))
    
    # The manager compiles only passed jobs into the report per user request.
    passed_jobs = [j for j in analyzed if j["grade"] == "PASS"][:max_jobs]
    
    report = f"# Pega Job Search Results ({date.today()})\n\n"
    report += f"**Jobs Meeting Criteria:** {len(passed_jobs)}\n\n"
    
    report += "## ✅ Approved Jobs (Correct Location)\n\n"
    if not passed_jobs:
        report += "No jobs met the criteria in this run.\n"
    else:
        for i, job in enumerate(passed_jobs, 1):
            report += f"### {i}. [Job Post]({job['url']})\n"
            report += f"> **Company:** {job.get('company', 'Not specified')} | **Location:** {job.get('location', 'Not specified')} | **Salary:** {job.get('salary', 'Not specified')}\n"
            report += f"> **Analyst Reasoning:** {job['reasoning']}\n\n"
            
    if state.get("errors"):
         report += f"\n## ⚠️ Run Errors (Self-Healed):\n"
         for e in state["errors"]:
             report += f"- {e}\n"
    
    # Save to file
    with open("matching_jobs.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Generated matching_jobs.md with {len(passed_jobs)} passed jobs.")
    return {"final_report": report}

def router(state: JobSearchState):
    scraped_count = len(state.get("scraped_jobs", []))
    exhausted = state.get("search_exhausted", False)
    
    if scraped_count < 40 and not exhausted:
        print(f"\n  [ROUTER] Only {scraped_count}/40 valid jobs scraped. Sending back to Scout...")
        return "Scout"
    
    print(f"\n  [ROUTER] Proceeding to Analyst with {scraped_count} jobs.")
    return "Analyst"

# --- 4. Build the Graph ---
workflow = StateGraph(JobSearchState)

workflow.add_node("Scout", scout_node)
workflow.add_node("Scraper", scraper_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Manager", manager_node)

# Define edges
workflow.set_entry_point("Scout")
workflow.add_edge("Scout", "Scraper")
workflow.add_conditional_edges("Scraper", router, {"Scout": "Scout", "Analyst": "Analyst"})
workflow.add_edge("Analyst", "Manager")
workflow.add_edge("Manager", END)

# Compile
app = workflow.compile()

def send_email_notification(to_email, report_path):
    """Sends the job report via email using SMTP settings from environment variables."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        print("  [!] Email skipped: SMTP_USER or SMTP_PASSWORD not set in .env")
        return

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = f"Pega Job Alert: {date.today()}"
        
        # Simple HTML wrapper for better readability in email
        html_body = f"<h2>Daily Pega Job Search Results</h2><pre>{report_content}</pre>"
        msg.attach(MIMEText(html_body, 'html'))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        
        print(f"  [EMAIL] Report sent successfully to {to_email}")
    except Exception as e:
        print(f"  [!] Failed to send email: {e}")

# Execution helper
def run_job_finder():
    config = get_search_queries()
    domains = get_job_domains()
    
    email_to = config.get('email')
    location = config.get('location', 'Dallas, TX')
    
    initial_state = {
        "search_queries": [config.get('search_term', 'pega')], 
        "errors": [],
        "search_index": 0,
        "search_exhausted": False,
        "scraped_jobs": [],
        "location": location,
        "config": config
    }
    
    print(f"\n--- STARTING MULTI-AGENT RUN ---")
    print(f"Location: {location}")
    if email_to:
        print(f"Notification Email: {email_to}")
    print(f"Target Websites: {domains}\n")
    
    for output in app.stream(initial_state):
        pass
    
    print("\n--- RUN COMPLETE ---!")
    
    if email_to:
        print(f"--- SENDING NOTIFICATION ---")
        send_email_notification(email_to, "matching_jobs.md")

if __name__ == "__main__":
    run_job_finder()
