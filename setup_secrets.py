"""
Setup script for Prefect Secret blocks.
Run once locally to initialize secrets in Prefect.

Usage:
    1. Create a .env file with your secrets (add .env to .gitignore!)
    2. Run: python setup_secrets.py
"""

import os

from dotenv import load_dotenv
from prefect.blocks.system import Secret

# Load environment variables from .env file
load_dotenv()


def create_secrets():
    """Create all required Prefect Secret blocks from environment variables."""

    secrets = {
        "job-searcher--database-host": os.getenv("DATABASE_HOST"),
        "job-searcher--database-port": os.getenv("DATABASE_PORT"),
        "job-searcher--database-name": os.getenv("DATABASE_NAME"),
        "job-searcher--database-user": os.getenv("DATABSE_USER"),
        "job-searcher--database-password": os.getenv("DATABSE_PASSWORD"),
        "job-searcher--telegram-bot-token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "job-searcher--telegram-chat-id": os.getenv("TELEGRAM_CHAT_ID"),
        "job-searcher--anthropic-api-key": os.getenv("ANTHROPIC_API_KEY"),
    }

    # Check for missing secrets
    missing = [name for name, value in secrets.items() if not value]
    if missing:
        print("❌ Error: Missing environment variables for:")
        for name in missing:
            print(f"   - {name}")
        print("\nPlease set these in your .env file")
        return

    # Create all secrets
    print("Creating Prefect Secret blocks...\n")
    for name, value in secrets.items():
        try:
            secret = Secret(value=value)
            secret.save(name, overwrite=True)
            print(f"✓ Created secret: {name}")
        except Exception as e:
            print(f"✗ Failed to create {name}: {e}")

    print("\n✅ All secrets created successfully!")
    print("⚠️  Remember: Never commit your .env file to version control")


if __name__ == "__main__":
    create_secrets()
