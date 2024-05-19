#Config.py
import os

# Database Configuration
DATABASE_NAME = os.getenv('DATABASE_NAME', './data/numbers.db')

# API Configuration
API_ENDPOINT = "data.ny.gov"
API_LIMIT = 5000

# API Endpoints for different datasets
API_ENDPOINT_MM = "5xaw-6ayf"
API_ENDPOINT_PB = "d6yy-54nr"
