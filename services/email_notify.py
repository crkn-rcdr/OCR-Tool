"""Send OCR run logs by email using SMTP settings from the environment."""

import os
from pathlib import Path
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Log directory
LOG_DIR = BASE_DIR / "logs"

# Environment variables
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 25))
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")
TO_EMAIL = [email.strip() for email in os.getenv("TO_EMAIL").split(",") if email.strip()]

def send_log_email(log_dir, smtp_host, smtp_port, from_email, to_email):
    """ 
    Traverse the log directory and send summary/error log files as email attachments.
    """

    msg = EmailMessage()
    msg["Subject"] = "PaddleOCR Report"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content("The OCR process has completed successfully. Please review the attached summary and error logs.")
    # Walk through log folders
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith(("summary", "error")):
                path = os.path.join(root, f)

                with open(path, "rb") as file:
                    msg.add_attachment(
                        file.read(),
                        maintype="application",
                        subtype="octet-stream",
                        filename=os.path.basename(path),
                    )

    # Send email
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.send_message(msg)


def main():
    """Load configured SMTP settings and send the current OCR log bundle."""
    send_log_email(
        log_dir=LOG_DIR,
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        from_email=FROM_EMAIL,
        to_email=TO_EMAIL,
    )

if __name__ == "__main__":
    main()
