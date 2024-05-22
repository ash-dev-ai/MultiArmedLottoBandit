# settings.py
import logging

# Configuration settings for the application

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_ENDPOINT = "data.ny.gov"
API_LIMIT = 5000

# Log configuration settings
logging.info(f"API endpoint: {API_ENDPOINT}")
logging.info(f"API limit: {API_LIMIT}")
