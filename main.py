import sys
from dotenv import load_dotenv

# Load environment variables if any .env file exists
load_dotenv()

from app.main import main

if __name__ == "__main__":
    main()
