"""
Email Service for Pakistan Travel RAG System.
Handles sending detailed travel documents via email.
"""

import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import json

logger = logging.getLogger(__name__)

@dataclass
class TravelDocument:
    """Travel document structure."""
    trip_summary: str
    detailed_itinerary: str
    accommodation_info: str
    transport_info: str
    budget_breakdown: str
    contact_info: str
    emergency_contacts: str

class EmailService:
    """Service for sending travel documents via email."""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL", "")
        self.sender_password = os.getenv("SENDER_PASSWORD", "")
        
    def send_travel_document(self, recipient_email: str, 
                           travel_document: TravelDocument,
                           trip_details: Dict[str, Any]) -> bool:
        """
        Send complete travel document to user email.
        
        Args:
            recipient_email: User's email address
            travel_document: Formatted travel document
            trip_details: Trip planning details
            
        Returns:
            True if email sent successfully
        """
        
        try:
            # Validate email configuration
            if not self._validate_config():
                logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Your Pakistan Travel Itinerary - {trip_details.get('destination', 'Multi-City Trip')}"
            
            # Create HTML email body
            html_body = self._generate_email_body(travel_document, trip_details)
            msg.attach(MimeText(html_body, 'html'))
            
            # Create PDF attachment (optional)
            pdf_path = self._generate_pdf_document(travel_document, trip_details)
            if pdf_path:
                self._attach_pdf(msg, pdf_path, trip_details.get('destination', 'Pakistan-Trip'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Travel document sent successfully to {recipient_email}")
            
            # Cleanup temp files
            if pdf_path and Path(pdf_path).exists():
                Path(pdf_path).unlink()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate email configuration."""
        return bool(self.sender_email and self.sender_password)
    
    def _generate_email_body(self, document: TravelDocument, details: Dict[str, Any]) -> str:
        """Generate HTML email body."""
        
        destination = details.get('destination', 'Pakistan')
        duration = details.get('duration', 'Multi-day')
        travelers = details.get('travelers', 1)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #2c5530, #4a7c59); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; max-width: 800px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4a7c59; background: #f9f9f9; }}
                .section h3 {{ color: #2c5530; margin-top: 0; }}
                .highlight {{ background: #e8f4ea; padding: 10px; border-radius: 5px; }}
                .footer {{ background: #f0f0f0; padding: 15px; text-align: center; font-size: 0.9em; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🇵🇰 Your Pakistan Travel Itinerary</h1>
                <p>Destination: {destination} | Duration: {duration} | Travelers: {travelers}</p>
            </div>
            
            <div class="content">
                <div class="highlight">
                    <h2>📋 Trip Summary</h2>
                    <p>{document.trip_summary}</p>
                </div>
                
                <div class="section">
                    <h3>📅 Detailed Itinerary</h3>
                    <div style="white-space: pre-line;">{document.detailed_itinerary}</div>
                </div>
                
                <div class="section">
                    <h3>🏨 Accommodation Information</h3>
                    <div style="white-space: pre-line;">{document.accommodation_info}</div>
                </div>
                
                <div class="section">
                    <h3>🚗 Transport Information</h3>
                    <div style="white-space: pre-line;">{document.transport_info}</div>
                </div>
                
                <div class="section">
                    <h3>💰 Budget Breakdown</h3>
                    <div style="white-space: pre-line;">{document.budget_breakdown}</div>
                </div>
                
                <div class="section">
                    <h3>📞 Important Contacts</h3>
                    <div style="white-space: pre-line;">{document.contact_info}</div>
                </div>
                
                <div class="section">
                    <h3>🚨 Emergency Information</h3>
                    <div style="white-space: pre-line;">{document.emergency_contacts}</div>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Pakistan Travel Intelligence RAG System</p>
                <p>Safe travels! 🧳</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_pdf_document(self, document: TravelDocument, details: Dict[str, Any]) -> Optional[str]:
        """Generate PDF version of travel document."""
        
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            
            # Write text content (simplified version for now)
            content = f"""
Pakistan Travel Itinerary

Destination: {details.get('destination', 'Pakistan')}
Duration: {details.get('duration', 'Multi-day')}
Travelers: {details.get('travelers', 1)}

TRIP SUMMARY
{document.trip_summary}

DETAILED ITINERARY
{document.detailed_itinerary}

ACCOMMODATION INFORMATION
{document.accommodation_info}

TRANSPORT INFORMATION  
{document.transport_info}

BUDGET BREAKDOWN
{document.budget_breakdown}

IMPORTANT CONTACTS
{document.contact_info}

EMERGENCY INFORMATION
{document.emergency_contacts}

Generated by Pakistan Travel Intelligence RAG System
            """
            
            temp_file.write(content)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return None
    
    def _attach_pdf(self, msg: MimeMultipart, pdf_path: str, filename: str) -> None:
        """Attach PDF file to email message."""
        
        try:
            with open(pdf_path, "rb") as attachment:
                part = MimeBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}-Itinerary.txt'
            )
            
            msg.attach(part)
            
        except Exception as e:
            logger.error(f"Failed to attach PDF: {e}")
    
    def send_notification(self, recipient_email: str, message: str, subject: str = "Pakistan Travel Update") -> bool:
        """Send simple notification email."""
        
        try:
            if not self._validate_config():
                return False
                
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls() 
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

# Global service instance
email_service = EmailService()