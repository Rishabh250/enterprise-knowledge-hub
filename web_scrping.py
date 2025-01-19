# Standard library imports
import logging
import os
from functools import lru_cache
from typing import Optional

# Third party imports
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define constants
DEFAULT_URL = "https://www.linkedin.com/feed/update/urn:li:activity:7286308453000654848/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@lru_cache()
def get_llm():
    """Initialize and cache LLM instance"""
    return OllamaLLM(model=os.getenv("MODEL"), temperature=0.0)

def fetch_website_content(url: str = DEFAULT_URL) -> Optional[str]:
    """
    Fetch website content with error handling and logging
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data from {url}: {str(e)}")
        return None

def process_content(html_content: str) -> Optional[str]:
    """
    Process and format website content using BeautifulSoup and LLM
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
            
        # Extract and clean text
        text = ' '.join(
            line.strip() 
            for line in soup.get_text().splitlines() 
            if line.strip()
        )

        # Format using LLM
        prompt = """Please format and structure the following website content in a clear and organized way:
        {text}
        """.format(text=text[:4000])  # Limit text length to avoid token limits

        return get_llm().invoke(prompt)

    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        return None

def main():
    """Main execution function"""
    logger.info(f"Fetching content from {DEFAULT_URL}")
    
    if html_content := fetch_website_content():
        if formatted_content := process_content(html_content):
            print(formatted_content)
            return True
    return False

if __name__ == "__main__":
    main()