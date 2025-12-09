#!/usr/bin/env python3
"""
n8n Webhook Integration
Sends voicebot data to existing n8n workflow for Atera ticket creation
"""

import requests
import logging
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/create-ticket"


def create_ticket_via_n8n_blocking(
    caller_id: str,
    transcript: str,
    severity: str,
    customer_name: str = "",
    product_family: str = "General Support",
    conversation_completeness: float = 0.0,
    conversation_duration: float = 0.0
) -> Optional[str]:
    """
    Send ticket data to n8n webhook (synchronous, blocking)
    This is the internal implementation that does the actual work.
    """
    try:
        # Enhanced payload with conversation quality data (customer lookup via database)
        payload = {
            # Basic ticket information
            'caller_id': caller_id,
            'transcript': transcript,
            'severity': severity,
            'product_family': product_family,
            'customer_name': customer_name,
            'timestamp': datetime.now().isoformat(),
            'source': 'voicebot',

            # Conversation quality metrics (enhanced feature)
            'conversation_completeness': conversation_completeness,
            'conversation_duration_seconds': conversation_duration,
            'information_gathering_quality': 'High' if conversation_completeness >= 0.7 else 'Medium' if conversation_completeness >= 0.4 else 'Low',

            # Enhanced internal notes with structured information
            'internal_notes': _build_enhanced_internal_notes(
                caller_id, customer_name, product_family, severity, transcript,
                conversation_completeness, conversation_duration
            ),

            # NETOVO required fields (customer/contract lookup via database in n8n)
            'field_technician': 'UNASSIGNED',

            # Additional metadata
            'voicebot_version': '2.0_enhanced',
            'ticket_creation_method': 'conversation_state_managed'
        }

        logger.info(f"Creating ticket via n8n: severity={severity}, product={product_family}, caller={caller_id}")

        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=10.0
        )

        if response.status_code == 200:
            result = response.json()

            # Check if ticket was created successfully (for both verified and unknown customers)
            if result.get('success', True):
                ticket_id = result.get('ticket_id') or result.get('ticket_number')
                verification_status = result.get('verification_status', 'unknown')

                if verification_status == 'verified':
                    logger.info(f"✅ Ticket created for verified customer: {ticket_id}")
                    return {
                        'status': 'success',
                        'ticket_id': ticket_id,
                        'ticket_created': True,
                        'customer_verified': True,
                        'verification_status': 'verified'
                    }
                else:
                    logger.info(f"✅ Ticket created for unknown customer: {ticket_id}")
                    return {
                        'status': 'success',
                        'ticket_id': ticket_id,
                        'ticket_created': True,
                        'customer_verified': False,
                        'verification_status': 'unknown'
                    }

            else:
                logger.error(f"❌ Ticket creation failed: {result}")
                return {
                    'status': 'failed',
                    'message': result.get('message', 'Ticket creation failed'),
                    'ticket_created': False
                }

        else:
            logger.error(f"❌ n8n webhook failed: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        logger.error("❌ n8n webhook timeout (>10 seconds)")
        return None
    except Exception as e:
        logger.error(f"❌ n8n webhook error: {e}")
        return None


def create_ticket_via_n8n(
    caller_id: str,
    transcript: str,
    severity: str,
    customer_name: str = "",
    product_family: str = "General Support",
    conversation_completeness: float = 0.0,
    conversation_duration: float = 0.0
) -> None:
    """
    Send ticket data to n8n webhook (NON-BLOCKING via background thread)

    This function immediately returns and creates the ticket in the background.
    The conversation can continue without waiting for the webhook response.

    Args:
        caller_id: Phone number
        transcript: Full conversation transcript
        severity: critical/high/medium/low
        customer_name: Extracted customer name (if any)

    Returns:
        None (runs in background)
    """
    def _background_ticket_creation():
        """Background thread function for ticket creation"""
        try:
            logger.info(f"🎫 Background ticket creation started for caller: {caller_id}")
            ticket_id = create_ticket_via_n8n_blocking(
                caller_id=caller_id,
                transcript=transcript,
                severity=severity,
                customer_name=customer_name,
                product_family=product_family,
                conversation_completeness=conversation_completeness,
                conversation_duration=conversation_duration
            )

            if ticket_id:
                logger.info(f"✅ Background ticket creation successful: {ticket_id}")
            else:
                logger.warning(f"⚠️ Background ticket creation failed for caller: {caller_id}")

        except Exception as e:
            logger.error(f"❌ Background ticket creation error: {e}")

    # Start ticket creation in background thread (non-blocking)
    thread = threading.Thread(
        target=_background_ticket_creation,
        name=f"TicketCreation-{caller_id}",
        daemon=True  # Dies with main process
    )
    thread.start()
    logger.info(f"🚀 Ticket creation started in background for {caller_id}")

    # Return immediately - conversation can continue


def format_transcript(messages: list) -> str:
    """Format conversation messages into readable transcript"""
    transcript = ""
    for msg in messages:
        role = "Customer" if msg.get('role') == 'user' else "VoiceBot"
        content = msg.get('content', '')
        transcript += f"{role}: {content}\n\n"
    return transcript.strip()


def detect_product_family(messages: list) -> str:
    """
    Detect product family from conversation
    Based on common NETOVO service categories
    """
    full_text = " ".join([msg.get('content', '') for msg in messages]).lower()

    # Email-related issues
    if any(word in full_text for word in ['email', 'outlook', 'mail', 'exchange', 'smtp', 'imap']):
        return "Email Services"

    # Printer/printing issues
    if any(word in full_text for word in ['printer', 'print', 'printing', 'paper', 'toner', 'cartridge']):
        return "Printing Services"

    # Network/connectivity issues
    if any(word in full_text for word in ['network', 'internet', 'wifi', 'connection', 'router', 'switch']):
        return "Network Services"

    # Software issues
    if any(word in full_text for word in ['software', 'application', 'program', 'app', 'system', 'windows', 'microsoft']):
        return "Software Support"

    # Hardware issues
    if any(word in full_text for word in ['computer', 'laptop', 'desktop', 'hardware', 'device', 'machine']):
        return "Hardware Support"

    # Security issues
    if any(word in full_text for word in ['password', 'login', 'access', 'security', 'account', 'virus']):
        return "Security Services"

    # Default fallback
    return "General Support"


def extract_customer_name(messages: list) -> str:
    """
    Extract customer name from conversation
    Looks for patterns: "I'm John", "This is Mary", etc.
    """
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()

            # Common introduction patterns
            if "i'm " in content or "i am " in content:
                words = content.replace("i'm", "").replace("i am", "").split()
                if words and len(words[0]) > 2:
                    return words[0].strip('.,!?').title()

            if "this is " in content:
                words = content.split("this is ")[1].split()
                if words and len(words[0]) > 2:
                    return words[0].strip('.,!?').title()

            if "my name is " in content:
                words = content.split("my name is ")[1].split()
                if words and len(words[0]) > 2:
                    return words[0].strip('.,!?').title()

    return "Unknown Customer"


def _build_enhanced_internal_notes(
    caller_id: str,
    customer_name: str,
    product_family: str,
    severity: str,
    transcript: str,
    conversation_completeness: float,
    conversation_duration: float
) -> str:
    """
    Build enhanced internal notes with structured information for NETOVO technicians
    """

    # Build comprehensive internal notes
    notes = f"""=== VOICEBOT GENERATED TICKET (Enhanced v2.0) ===

CUSTOMER INFORMATION:
• Name: {customer_name}
• Phone: {caller_id}
• Customer Lookup: Database verification in n8n workflow

CALL DETAILS:
• Duration: {conversation_duration:.1f} seconds ({conversation_duration/60:.1f} minutes)
• Information Quality: {conversation_completeness:.1%} complete
• Category: {product_family}
• Severity: {severity.upper()}

CONVERSATION ANALYSIS:
"""

    # Add conversation quality indicators
    if conversation_completeness >= 0.7:
        notes += "• ✅ HIGH QUALITY: Comprehensive information gathered (120+ seconds)\n"
    elif conversation_completeness >= 0.4:
        notes += "• ⚠️ MEDIUM QUALITY: Adequate information gathered\n"
    else:
        notes += "• ❌ LOW QUALITY: Limited information gathered - may need follow-up\n"

    # Add timing information
    if conversation_duration >= 120:
        notes += "• ✅ THOROUGH: Sufficient time spent gathering details\n"
    elif conversation_duration >= 60:
        notes += "• ⚠️ MODERATE: Reasonable conversation duration\n"
    else:
        notes += "• ❌ BRIEF: Short conversation - may need additional details\n"

    # Add technical notes based on category
    category_notes = {
        'PC - Printing': 'Check: Printer drivers, network connectivity, print queue, toner/paper',
        'PC - Software': 'Check: Version compatibility, installation logs, user permissions',
        'Office365 - Email': 'Check: Email client settings, server connectivity, account status',
        'Office365 - Login': 'Check: Password reset, MFA settings, account lockout status',
        'Network - VPN': 'Check: VPN client version, network settings, authentication',
        'Network - Wireless': 'Check: WiFi settings, signal strength, network adapter',
        'VoIP - General': 'Check: Phone registration, network quality, 3CX settings'
    }

    if product_family in category_notes:
        notes += f"\nTECHNICIAN CHECKLIST:\n• {category_notes[product_family]}\n"

    notes += f"""
=== FULL CONVERSATION TRANSCRIPT ===
{transcript}

=== TICKET METADATA ===
• Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Source: VoiceBot Enhanced v2.0
• Processing: State-aware conversation management
• Escalation: Automatic (if unassigned >15min)
"""

    return notes