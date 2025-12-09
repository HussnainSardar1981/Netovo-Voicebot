#!/usr/bin/python3
"""
Zabbix Alert Server - Complete replacement for n8n workflow
Runs directly on server, eliminates Docker networking issues
"""

import os
import sys
import time
import json
import logging
import requests
import base64
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify

# Add project path
sys.path.insert(0, "/home/aiadmin/netovo_voicebot/kokora")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ZabbixAlertProcessor:
    """Processes Zabbix alerts with business logic"""

    def __init__(self):
        self.technicians = [
            {'name': 'Vitaliy', 'extension': '1680'},    # Week 1
            {'name': 'Sasha', 'extension': '1689'},      # Week 2
            {'name': 'Bogdan', 'extension': '1687'},     # Week 3
            {'name': 'Aleksey', 'extension': '1111'}     # Week 4 - TESTING: Dedicated test extension
        ]
        # Manager for escalation
        self.manager = {'name': 'Artem', 'extension': '1691'}

        # Track alert states for escalation
        self.active_alerts = {}

        # Track phone call responses for escalation
        self.call_responses = {}

        # Track call attempts and failures for backup calling
        self.call_attempts = {}

        # Track escalation attempts to prevent infinite loops
        self.escalation_attempts = {}

        # Track all processed alerts for duplicate detection
        self.processed_alerts = {}

        # NEW: Store alert data by call_id for AudioSocket voicebot queries
        self.alert_data_by_call_id = {}

        # Thread safety
        import threading
        self.lock = threading.RLock()

    def cleanup_old_data(self):
        """Clean up old alerts and call responses to prevent memory leaks"""
        try:
            with self.lock:
                cutoff_time = datetime.now() - timedelta(hours=24)

                # Clean old active alerts (older than 24 hours)
                old_alert_count = len(self.active_alerts)
                self.active_alerts = {
                    alert_id: alert_info
                    for alert_id, alert_info in self.active_alerts.items()
                    if alert_info.get('first_seen', datetime.min) > cutoff_time
                }

                # Clean old call responses (older than 24 hours)
                old_call_count = len(self.call_responses)
                self.call_responses = {
                    call_id: call_info
                    for call_id, call_info in self.call_responses.items()
                    if call_info.get('call_time', datetime.min) > cutoff_time
                }

                # Clean old escalation attempts (older than 24 hours)
                old_escalation_count = len(self.escalation_attempts)
                self.escalation_attempts = {
                    alert_id: escalation_info
                    for alert_id, escalation_info in self.escalation_attempts.items()
                    if escalation_info.get('first_escalation', datetime.min) > cutoff_time
                }

                # Clean old call attempts (older than 24 hours)
                old_call_attempts_count = len(self.call_attempts)
                self.call_attempts = {
                    call_id: call_info
                    for call_id, call_info in self.call_attempts.items()
                    if call_info.get('attempt_time', datetime.min) > cutoff_time
                }

                # Clean old processed alerts (older than 24 hours)
                old_processed_count = len(self.processed_alerts)
                self.processed_alerts = {
                    alert_id: process_info
                    for alert_id, process_info in self.processed_alerts.items()
                    if process_info.get('processed_time', datetime.min) > cutoff_time
                }

                # Clean old alert data by call_id (older than 24 hours)
                old_alert_data_count = len(self.alert_data_by_call_id)
                self.alert_data_by_call_id = {
                    call_id: alert_info
                    for call_id, alert_info in self.alert_data_by_call_id.items()
                    if alert_info.get('created_time', datetime.min) > cutoff_time
                }

                cleaned_alerts = old_alert_count - len(self.active_alerts)
                cleaned_calls = old_call_count - len(self.call_responses)
                cleaned_escalations = old_escalation_count - len(self.escalation_attempts)
                cleaned_attempts = old_call_attempts_count - len(self.call_attempts)
                cleaned_processed = old_processed_count - len(self.processed_alerts)
                cleaned_alert_data = old_alert_data_count - len(self.alert_data_by_call_id)

                if cleaned_alerts > 0 or cleaned_calls > 0 or cleaned_escalations > 0 or cleaned_attempts > 0 or cleaned_processed > 0 or cleaned_alert_data > 0:
                    logger.info(f"🧹 Memory cleanup: Removed {cleaned_alerts} old alerts, {cleaned_calls} old call responses, {cleaned_escalations} old escalations, {cleaned_attempts} old call attempts, {cleaned_processed} old processed alerts, {cleaned_alert_data} old alert data")

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def get_est_time(self):
        """Get current EST/EDT time with proper DST handling"""
        try:
            import pytz
            est = pytz.timezone('US/Eastern')
            return datetime.now(est)
        except ImportError:
            # Fallback if pytz not available - basic EST/EDT logic
            now = datetime.now(timezone.utc)

            # Simple DST detection (not perfect but better than fixed offset)
            # DST typically runs from second Sunday in March to first Sunday in November
            year = now.year
            dst_start = datetime(year, 3, 8 + (6 - datetime(year, 3, 8).weekday()) % 7)
            dst_end = datetime(year, 11, 1 + (6 - datetime(year, 11, 1).weekday()) % 7)

            # Check if we're in DST period
            if dst_start <= now.replace(tzinfo=None) < dst_end:
                est_offset = timedelta(hours=-4)  # EDT
                logger.debug("Using EDT (UTC-4)")
            else:
                est_offset = timedelta(hours=-5)  # EST
                logger.debug("Using EST (UTC-5)")

            return now + est_offset

    def is_business_hours(self):
        """Check if current time is business hours (9 AM - 6 PM EST, Mon-Fri)"""
        est_time = self.get_est_time()
        hour = est_time.hour
        day = est_time.weekday()  # 0 = Monday, 6 = Sunday

        # Monday to Friday (0-4) and between 9 AM and 6 PM
        is_weekday = day <= 4
        is_business_time = 9 <= hour <= 18

        return is_weekday and is_business_time

    def get_week_number(self, date):
        """Get ISO week number"""
        return date.isocalendar()[1]

    def get_current_week_technician(self):
        """Get technician for current week with rotation"""
        est_time = self.get_est_time()
        current_week = self.get_week_number(est_time)

        # Adjust calculation (same logic as n8n)
        technician_index = (current_week - 1) % 4
        technician_index = (technician_index + 3) % 4  # Manual adjustment

        selected_tech = self.technicians[technician_index]

        logger.info(f"EST Time: {est_time.isoformat()}")
        logger.info(f"Current Week: {current_week}")
        logger.info(f"Technician Index: {technician_index}")
        logger.info(f"Selected Technician: {selected_tech['name']} (Ext: {selected_tech['extension']})")

        return selected_tech

    def get_backup_technicians(self, exclude_extension=None):
        """Get backup technicians list excluding the primary one"""
        backup_list = [tech for tech in self.technicians
                      if tech['extension'] != exclude_extension]
        return backup_list

    def trigger_cascading_calls(self, alert_data, primary_technician, reason="Primary technician unavailable"):
        """Trigger calls to backup technicians if primary fails"""
        logger.info(f"🔄 Starting cascading calls - Primary {primary_technician['name']} failed: {reason}")

        backup_technicians = self.get_backup_technicians(primary_technician['extension'])

        for i, backup_tech in enumerate(backup_technicians[:2]):  # Try 2 backups max
            logger.info(f"Trying backup technician {i+1}: {backup_tech['name']} ({backup_tech['extension']})")

            backup_call_result = self.trigger_voicebot_call(backup_tech['extension'], alert_data)

            if backup_call_result.get('success'):
                logger.info(f"  Backup call successful to {backup_tech['name']}")
                return {
                    'success': True,
                    'technician': backup_tech,
                    'call_result': backup_call_result,
                    'backup_level': i + 1,
                    'reason': f"Escalated to backup after {reason}"
                }
            else:
                logger.warning(f" Backup call failed to {backup_tech['name']}: {backup_call_result.get('error')}")

        # If all backups fail, escalate to manager
        logger.error("🚨 All backup technicians failed - escalating to manager")
        manager_result = self.escalate_to_manager(alert_data, "All technicians unavailable")

        return {
            'success': False,
            'all_technicians_failed': True,
            'manager_escalation': manager_result,
            'reason': "All technicians unavailable - escalated to manager"
        }

    def requires_phone_call(self, alert_data):
        """Check if phone call is required based on trigger keywords"""
        trigger_name = (alert_data.get('trigger_name', '') or '').lower()
        description = (alert_data.get('description', '') or '').lower()

        logger.info('🔍 DEBUG - Phone call check:')
        logger.info(f'  trigger_name: {repr(alert_data.get("trigger_name"))}')
        logger.info(f'  triggerName (lowercase): {repr(trigger_name)}')
        logger.info(f'  description (lowercase): {repr(description)}')

        phone_keywords = ['unavailable', 'not available']

        for keyword in phone_keywords:
            logger.info(f'  Checking for keyword: "{keyword}"')
            logger.info(f'  triggerName.includes("{keyword}"): {keyword in trigger_name}')
            logger.info(f'  description.includes("{keyword}"): {keyword in description}')

            if keyword in trigger_name or keyword in description:
                logger.info(f'🔥 Phone call required - found keyword \'{keyword}\' in alert')
                return True

        logger.info(' No phone call required - keywords not found')
        return False

    def trigger_voicebot_call(self, extension, alert_data):
        """Create Asterisk call file directly (no HTTP API needed)"""
        try:
            logger.info(f"🚨 Creating direct VoiceBot call to {extension}")

            # Check spool directory first
            spool_dir = "/var/spool/asterisk/outgoing"
            if not os.path.exists(spool_dir):
                logger.error(f"Asterisk spool directory does not exist: {spool_dir}")
                return {'success': False, 'error': 'Asterisk spool directory not found'}

            # Note: Skip write permission check as it may fail even when we can actually write

            # Create unique call ID
            call_id = f"zabbix_alert_{int(time.time())}"

            # Clean data for Asterisk variables
            def clean_for_asterisk(text):
                if not text:
                    return "Unknown"
                return str(text).replace('"', '').replace("'", "").replace('\\', '').replace('\n', ' ')[:100]

            alert_id = clean_for_asterisk(alert_data.get('alert_id', ''))
            hostname = clean_for_asterisk(alert_data.get('hostname', ''))
            trigger_name = clean_for_asterisk(alert_data.get('trigger_name', ''))
            severity = clean_for_asterisk(alert_data.get('severity', 'medium'))

            # Use context detection - this definitely works for alert routing!
            call_content = f"""Channel: PJSIP/{extension}@3cx-trunk
Context: zabbix-alert-voicebot
Extension: s
Priority: 1
CallerID: Zabbix Alert <1100>
Set: ALERT_MODE=zabbix
Set: ALERT_ID={alert_id}
Set: HOSTNAME={hostname}
Set: TRIGGER={trigger_name}
Set: SEVERITY={severity}
Set: CALL_ID={call_id}
MaxRetries: 2
RetryTime: 60
WaitTime: 30
Archive: yes
"""

            # Write to Asterisk spool directory with error checking
            spool_path = f"{spool_dir}/{call_id}.call"

            try:
                with open(spool_path, 'w') as f:
                    f.write(call_content)

                # Fix ownership for Asterisk processing (this was the root cause!)
                import subprocess
                try:
                    subprocess.run(['chown', 'asterisk:asterisk', spool_path], check=True)
                    logger.info(f"  Call file ownership changed to asterisk:asterisk: {spool_path}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to change ownership (but will try anyway): {e}")
                    # Set permissions as fallback
                    os.chmod(spool_path, 0o666)

            except PermissionError as pe:
                logger.error(f"Permission denied writing call file: {pe}")
                return {'success': False, 'error': f'Permission denied: {str(pe)}'}
            except OSError as oe:
                logger.error(f"OS error writing call file: {oe}")
                return {'success': False, 'error': f'OS error: {str(oe)}'}

            logger.info(f"  Call file created: {spool_path}")
            logger.info(f"Calling {extension} for alert {alert_id} on {hostname}")

            # Track call attempt for failure monitoring
            self.call_attempts[call_id] = {
                'extension': extension,
                'alert_data': alert_data,
                'attempt_time': datetime.now(),
                'spool_path': spool_path,
                'status': 'initiated'
            }

            # NEW: Store alert data for AudioSocket voicebot to query
            self.alert_data_by_call_id[call_id] = {
                'alert_id': alert_id,
                'hostname': hostname,
                'trigger': trigger_name,
                'severity': severity,
                'call_id': call_id,
                'extension': extension,
                'created_time': datetime.now(),
                'full_alert_data': alert_data
            }
            logger.info(f"📋 Stored alert data for call_id: {call_id}")

            return {
                'success': True,
                'call_id': call_id,
                'target_extension': extension,
                'alert_id': alert_data.get('alert_id'),
                'message': f'Zabbix alert call initiated to extension {extension}'
            }

        except Exception as error:
            logger.error(f" Call file creation failed: {error}")
            return {
                'success': False,
                'error': f'Call file creation failed: {str(error)}'
            }

    def check_dtmf_response(self, call_id):
        """Check if technician has responded via DTMF (1=acknowledge, 2=not available)"""
        try:
            # Check Asterisk logs for DTMF responses
            # This is a simple file-based approach that doesn't break existing functionality
            dtmf_log_path = f"/tmp/zabbix_dtmf_{call_id}.log"

            if os.path.exists(dtmf_log_path):
                with open(dtmf_log_path, 'r') as f:
                    response = f.read().strip()
                    if response in ['1', '2']:
                        logger.info(f"DTMF response received for {call_id}: {response}")
                        return response

            return None
        except Exception as e:
            logger.error(f"Error checking DTMF response: {e}")
            return None

    def check_escalation_timeouts(self):
        """Check if any calls need escalation due to timeout or technician unavailable"""
        current_time = datetime.now()
        escalations_needed = []

        for call_id, call_info in self.call_responses.items():
            if call_info.get('escalated'):
                continue  # Already escalated

            call_time = call_info.get('call_time')
            response = call_info.get('response')
            alert_id = call_info.get('alert_id')

            if not call_time:
                continue

            elapsed = current_time - call_time

            # Business hours: escalate after 5 minutes if no response
            # After hours: escalate immediately if technician says not available (response '2')
            business_hours = self.is_business_hours()

            should_escalate = False
            escalation_reason = ""

            if response == '2':  # Technician pressed 2 (not available)
                should_escalate = True
                escalation_reason = "Technician reported not available"
            elif business_hours and elapsed.total_seconds() >= 300:  # 5 minutes for business hours
                should_escalate = True
                escalation_reason = "No response within 5 minutes (business hours)"

            if should_escalate:
                escalations_needed.append({
                    'call_id': call_id,
                    'alert_id': alert_id,
                    'reason': escalation_reason,
                    'call_info': call_info
                })

        return escalations_needed

    def check_failed_calls(self):
        """Check for calls that failed to connect and trigger backup calls"""
        current_time = datetime.now()
        failed_calls = []

        for call_id, call_attempt in self.call_attempts.items():
            if call_attempt.get('status') != 'initiated':
                continue  # Already processed

            attempt_time = call_attempt.get('attempt_time')
            spool_path = call_attempt.get('spool_path')

            if not attempt_time:
                continue

            # Check if call file has been processed (5 minutes timeout)
            elapsed = current_time - attempt_time
            if elapsed.total_seconds() >= 300:  # 5 minutes

                # Check if call file still exists (indicates failure)
                if os.path.exists(spool_path):
                    logger.warning(f" Call {call_id} failed - call file still exists after 5 minutes")

                    failed_calls.append({
                        'call_id': call_id,
                        'extension': call_attempt['extension'],
                        'alert_data': call_attempt['alert_data'],
                        'reason': 'Call file not processed - likely busy/unavailable'
                    })

                    # Mark as failed
                    self.call_attempts[call_id]['status'] = 'failed'

                # Check if we have a DTMF response
                elif call_id in self.call_responses:
                    response = self.call_responses[call_id].get('response')
                    if response == '1':
                        # Successfully acknowledged
                        self.call_attempts[call_id]['status'] = 'acknowledged'
                        logger.info(f" Call {call_id} acknowledged by technician")
                    elif response == '2':
                        # Technician reported unavailable
                        self.call_attempts[call_id]['status'] = 'unavailable'
                        logger.info(f" Call {call_id} - technician reported unavailable")
                    # If no response, it will be handled by timeout escalation
                else:
                    # Call processed but no DTMF response yet - keep monitoring
                    logger.debug(f" Call {call_id} processed, waiting for DTMF response")

        return failed_calls

    def escalate_to_manager(self, alert_data, reason):
        """Escalate alert to manager (Artem 1691) with infinite loop prevention"""
        try:
            alert_id = alert_data.get('alert_id', 'unknown')

            # Check if we've already escalated this alert multiple times
            if alert_id in self.escalation_attempts:
                escalation_info = self.escalation_attempts[alert_id]
                escalation_count = escalation_info.get('count', 0)
                last_escalation = escalation_info.get('last_escalation', datetime.min)

                # Prevent spam escalations - max 3 escalations per alert, minimum 30 minutes between escalations
                time_since_last = datetime.now() - last_escalation
                if escalation_count >= 3:
                    logger.warning(f"  Escalation blocked: Alert {alert_id} has already been escalated {escalation_count} times")
                    return {'success': False, 'error': 'Maximum escalations reached', 'escalation_blocked': True}

                if time_since_last.total_seconds() < 1800:  # 30 minutes
                    remaining_minutes = (1800 - time_since_last.total_seconds()) / 60
                    logger.warning(f"  Escalation blocked: Alert {alert_id} escalated too recently, {remaining_minutes:.1f} minutes remaining")
                    return {'success': False, 'error': f'Must wait {remaining_minutes:.1f} minutes between escalations', 'escalation_blocked': True}

            logger.info(f" ESCALATING to manager {self.manager['name']} ({self.manager['extension']}) - Reason: {reason}")

            # Track escalation attempt
            current_time = datetime.now()
            if alert_id not in self.escalation_attempts:
                self.escalation_attempts[alert_id] = {
                    'first_escalation': current_time,
                    'count': 0,
                    'reasons': []
                }

            self.escalation_attempts[alert_id]['count'] += 1
            self.escalation_attempts[alert_id]['last_escalation'] = current_time
            self.escalation_attempts[alert_id]['reasons'].append(reason)

            # Create manager call with escalation context
            escalation_data = {
                **alert_data,
                'escalation_reason': reason,
                'original_alert_id': alert_id,
                'escalation_count': self.escalation_attempts[alert_id]['count']
            }

            manager_call = self.trigger_voicebot_call(self.manager['extension'], escalation_data)

            logger.info(f"Manager escalation call initiated (escalation #{self.escalation_attempts[alert_id]['count']}): {manager_call}")
            return manager_call

        except Exception as e:
            logger.error(f"Manager escalation failed: {e}")
            return {'success': False, 'error': str(e)}

    def create_ticket_with_fallback(self, alert_data):
        """Create support ticket with fallback logging if n8n fails"""
        ticket_data = {
            'caller_id': 'ZABBIX_SYSTEM',
            'transcript': f"""ZABBIX SYSTEM ALERT
Alert ID: {alert_data.get('alert_id', '')}
Host: {alert_data.get('hostname', '')}
Trigger: {alert_data.get('trigger_name', '')}
Severity: {(alert_data.get('severity', '') or '').upper()}
Time: {datetime.now().isoformat()}
Description: {alert_data.get('description', '')}
This alert was automatically generated by the Zabbix monitoring system.
Please investigate and resolve the issue promptly.""",
            'severity': 'high' if alert_data.get('severity') in ['high', 'disaster'] else 'medium',
            'customer_name': 'System Monitor',
            'product_family': 'Monitoring'
        }

        # Try primary ticket creation
        try:
            response = requests.post(
                'http://localhost:5678/webhook/create-ticket',
                json=ticket_data,
                timeout=30
            )
            logger.info(f"Ticket created successfully: {response.json()}")
            return response.json()
        except Exception as primary_error:
            logger.error(f"Primary ticket creation failed: {primary_error}")

            # FALLBACK: Log to file for manual processing
            try:
                fallback_log_path = "/var/log/zabbix_alerts_fallback.log"
                with open(fallback_log_path, 'a') as f:
                    f.write(f"\n{datetime.now().isoformat()} - FAILED TICKET CREATION\n")
                    f.write(f"Alert ID: {alert_data.get('alert_id', '')}\n")
                    f.write(f"Host: {alert_data.get('hostname', '')}\n")
                    f.write(f"Trigger: {alert_data.get('trigger_name', '')}\n")
                    f.write(f"Severity: {alert_data.get('severity', '')}\n")
                    f.write(f"Error: {primary_error}\n")
                    f.write("=" * 50 + "\n")

                logger.warning(f"Ticket creation failed - logged to {fallback_log_path}")
                return {'message': 'Logged to fallback file', 'status': 'fallback'}

            except Exception as fallback_error:
                logger.error(f"Both ticket creation and fallback logging failed: {fallback_error}")
                return {'message': 'All ticket creation methods failed', 'status': 'error'}

    def create_ticket(self, alert_data):
        """Wrapper for backward compatibility"""
        return self.create_ticket_with_fallback(alert_data)

    def process_alert(self, alert_data):
        """Main alert processing logic (same as n8n workflow)"""
        # Extract basic info first
        alert_id = alert_data.get('alert_id', '')
        hostname = alert_data.get('hostname', '')
        trigger_name = alert_data.get('trigger_name', '')
        severity = alert_data.get('severity', 'medium')

        logger.info(f"Processing alert: {alert_id} - {trigger_name}")

        # Thread-safe duplicate detection
        with self.lock:
            # Basic duplicate alert handling - ignore alerts processed in last 5 minutes
            current_time = datetime.now()
            if alert_id in self.processed_alerts:
                last_processed = self.processed_alerts[alert_id].get('processed_time', datetime.min)
                time_since_last = current_time - last_processed

                if time_since_last.total_seconds() < 300:  # 5 minutes
                    logger.info(f"⏭️ Duplicate alert {alert_id} ignored - processed {time_since_last.total_seconds():.0f} seconds ago")
                    return {
                        'alert_id': alert_id,
                        'action': 'duplicate_ignored',
                        'message': f'Alert processed {time_since_last.total_seconds():.0f} seconds ago',
                        'time_since_last': time_since_last.total_seconds()
                    }

            # Track this alert as processed
            self.processed_alerts[alert_id] = {
                'processed_time': current_time,
                'alert_data': alert_data
            }

        # Business logic (outside lock for performance)
        business_hours = self.is_business_hours()
        needs_phone_call = self.requires_phone_call(alert_data)

        logger.info(f"Business Hours: {business_hours}")
        logger.info(f"Needs Phone Call: {needs_phone_call}")

        result = {
            'alert_id': alert_id,
            'hostname': hostname,
            'trigger_name': trigger_name,
            'severity': severity,
            'business_hours': business_hours,
            'needs_phone_call': needs_phone_call,
            'timestamp': datetime.now().isoformat()
        }

        if business_hours:
            # Business Hours Processing
            logger.info('Business hours processing')

            # Create ticket immediately
            ticket_response = self.create_ticket(alert_data)
            result['ticket_created'] = True
            result['ticket_response'] = ticket_response

            if needs_phone_call:
                # Get current technician
                technician = self.get_current_week_technician()
                result['technician'] = technician

                # Make VoiceBot call with backup calling support
                call_result = self.trigger_voicebot_call(technician['extension'], alert_data)
                result['voicebot_call'] = call_result
                result['action'] = 'ticket_created_and_voicebot_call_triggered'

                # Store call for 5-minute timeout tracking (business hours)
                if call_result.get('success'):
                    call_id = call_result.get('call_id')
                    self.call_responses[call_id] = {
                        'call_time': datetime.now(),
                        'alert_id': alert_id,
                        'technician': technician,
                        'response': None,  # Will be updated by DTMF
                        'escalated': False
                    }
                else:
                    # Primary call failed - try backup technicians
                    logger.warning(f"Primary call to {technician['name']} failed: {call_result.get('error')}")
                    backup_result = self.trigger_cascading_calls(alert_data, technician, f"Primary call failed: {call_result.get('error')}")
                    result['backup_calls'] = backup_result

                    if backup_result.get('success'):
                        result['action'] = 'ticket_created_and_backup_voicebot_call_triggered'
                        result['technician'] = backup_result['technician']
                        result['backup_level'] = backup_result['backup_level']

                        # Store backup call for tracking
                        backup_call_result = backup_result['call_result']
                        if backup_call_result.get('success'):
                            call_id = backup_call_result.get('call_id')
                            self.call_responses[call_id] = {
                                'call_time': datetime.now(),
                                'alert_id': alert_id,
                                'technician': backup_result['technician'],
                                'response': None,
                                'escalated': False
                            }
                    else:
                        result['action'] = 'ticket_created_but_all_calls_failed_escalated_to_manager'

                logger.info(f"Business hours: Ticket created, VoiceBot call made to {technician['name']} ({technician['extension']})")
            else:
                result['action'] = 'immediate_ticket_created'
                logger.info('Business hours: Ticket created, no escalation needed')

        else:
            # After Hours Processing
            logger.info('After hours processing')

            if needs_phone_call:
                # NETOVO PROCEDURE: Wait 15 minutes for auto-remediation
                alert_id = alert_data.get('alert_id', 'unknown')

                if alert_id not in self.active_alerts:
                    # First time seeing this alert - start timer
                    logger.info(f"After-hours alert {alert_id} - waiting 15 minutes for auto-remediation")
                    self.active_alerts[alert_id] = {
                        'first_seen': datetime.now(),
                        'data': alert_data
                    }

                    # Create ticket immediately but delay phone call
                    ticket_response = self.create_ticket(alert_data)
                    result['ticket_created'] = True
                    result['ticket_response'] = ticket_response
                    result['action'] = 'after_hours_ticket_created_waiting_remediation'
                    result['remediation_wait'] = '15 minutes'

                    logger.info(f"After hours: Ticket created for {alert_id}, waiting 15 minutes before phone call")

                else:
                    # Check if 15 minutes have passed
                    first_seen = self.active_alerts[alert_id]['first_seen']
                    elapsed = datetime.now() - first_seen

                    if elapsed.total_seconds() >= 900:  # 15 minutes
                        logger.info(f"Alert {alert_id} still active after 15 minutes - triggering phone call")

                        # Get current technician
                        technician = self.get_current_week_technician()
                        result['technician'] = technician

                        # Make VoiceBot call with backup calling support
                        call_result = self.trigger_voicebot_call(technician['extension'], alert_data)
                        result['voicebot_call'] = call_result
                        result['action'] = 'after_hours_voicebot_call_after_wait'

                        # Store call for escalation tracking (after-hours)
                        if call_result.get('success'):
                            call_id = call_result.get('call_id')
                            self.call_responses[call_id] = {
                                'call_time': datetime.now(),
                                'alert_id': alert_id,
                                'technician': technician,
                                'response': None,  # Will be updated by DTMF
                                'escalated': False
                            }
                        else:
                            # Primary call failed - try backup technicians
                            logger.warning(f"After-hours primary call to {technician['name']} failed: {call_result.get('error')}")
                            backup_result = self.trigger_cascading_calls(alert_data, technician, f"After-hours primary call failed: {call_result.get('error')}")
                            result['backup_calls'] = backup_result

                            if backup_result.get('success'):
                                result['action'] = 'after_hours_backup_voicebot_call_after_wait'
                                result['technician'] = backup_result['technician']
                                result['backup_level'] = backup_result['backup_level']

                                # Store backup call for tracking
                                backup_call_result = backup_result['call_result']
                                if backup_call_result.get('success'):
                                    call_id = backup_call_result.get('call_id')
                                    self.call_responses[call_id] = {
                                        'call_time': datetime.now(),
                                        'alert_id': alert_id,
                                        'technician': backup_result['technician'],
                                        'response': None,
                                        'escalated': False
                                    }
                            else:
                                result['action'] = 'after_hours_all_calls_failed_escalated_to_manager'

                        logger.info(f"After hours: VoiceBot call made to {technician['name']} ({technician['extension']}) after 15-minute wait")

                    else:
                        remaining = 900 - elapsed.total_seconds()
                        logger.info(f"Alert {alert_id} still in remediation period - {remaining:.0f} seconds remaining")
                        result['action'] = 'after_hours_still_waiting_remediation'
                        result['remaining_wait'] = f"{remaining:.0f} seconds"
            else:
                # Create ticket only, no phone call
                ticket_response = self.create_ticket(alert_data)
                result['ticket_created'] = True
                result['ticket_response'] = ticket_response
                result['action'] = 'ticket_created_only'

                logger.info('After hours: Ticket created only, no phone call needed')

        return result

# Initialize processor
processor = ZabbixAlertProcessor()

@app.route('/zabbix-alert', methods=['POST'])
def handle_zabbix_alert():
    """Handle Zabbix webhook - direct replacement for n8n webhook"""
    try:
        # Extract alert data (same as n8n workflow)
        input_data = request.json
        logger.info(f'🔍 Raw input data: {json.dumps(input_data, indent=2)}')

        # Extract alert data from webhook body
        alert_data = input_data.get('body', input_data)
        logger.info(f'🔍 Extracted alert data: {json.dumps(alert_data, indent=2)}')

        # Process alert with same logic as n8n
        processing_result = processor.process_alert(alert_data)

        # Handle duplicate detection responses (don't have business_hours key)
        if processing_result.get('action') == 'duplicate_ignored':
            return jsonify({
                'status': 'success',
                'mode': 'duplicate_detection',
                **processing_result
            })

        return jsonify({
            'status': 'success',
            'mode': 'business_hours' if processing_result['business_hours'] else 'after_hours',
            **processing_result
        })

    except Exception as e:
        logger.error(f" Error processing Zabbix alert: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'zabbix-alert-server',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/dtmf-response', methods=['POST'])
def handle_dtmf_response():
    """Handle DTMF response from VoiceBot"""
    try:
        data = request.json
        call_id = data.get('call_id')
        dtmf_response = data.get('dtmf_response')
        alert_id = data.get('alert_id')

        logger.info(f"DTMF response received: Call {call_id}, Response {dtmf_response}, Alert {alert_id}")

        if call_id and dtmf_response:
            # Update existing call record with DTMF response
            if call_id in processor.call_responses:
                processor.call_responses[call_id]['response'] = dtmf_response
                processor.call_responses[call_id]['response_time'] = datetime.now()
            else:
                # Store DTMF response for escalation tracking
                processor.call_responses[call_id] = {
                    'response': dtmf_response,
                    'response_time': datetime.now(),
                    'alert_id': alert_id,
                    'escalated': False
                }

            response_text = {
                '1': 'Alert acknowledged by technician',
                '2': 'Technician not available - will escalate'
            }.get(dtmf_response, 'Unknown response')

            # If technician pressed '2' (not available), trigger immediate escalation
            if dtmf_response == '2' and call_id in processor.call_responses:
                call_info = processor.call_responses[call_id]
                alert_id = call_info.get('alert_id', 'unknown')

                # Get alert data for escalation
                if alert_id in processor.active_alerts:
                    alert_data = processor.active_alerts[alert_id]['data']
                    escalation_result = processor.escalate_to_manager(alert_data, "Technician reported not available")

                    # Mark as escalated
                    processor.call_responses[call_id]['escalated'] = True
                    processor.call_responses[call_id]['escalation_result'] = escalation_result

                    logger.info(f"Immediate escalation triggered for call {call_id}")

            return jsonify({
                'status': 'success',
                'message': response_text,
                'call_id': call_id,
                'dtmf_response': dtmf_response
            })

        return jsonify({
            'status': 'error',
            'error': 'Missing call_id or dtmf_response'
        }), 400

    except Exception as e:
        logger.error(f" DTMF response error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/alert-data/<call_id>', methods=['GET'])
def get_alert_data(call_id):
    """NEW: Get alert data by call_id for AudioSocket voicebot"""
    try:
        logger.info(f"📞 AudioSocket voicebot requesting alert data for call_id: {call_id}")

        if call_id in processor.alert_data_by_call_id:
            alert_data = processor.alert_data_by_call_id[call_id]
            logger.info(f"✅ Alert data found: {alert_data['alert_id']} - {alert_data['hostname']}")

            return jsonify({
                'status': 'success',
                'alert_data': {
                    'alert_id': alert_data['alert_id'],
                    'hostname': alert_data['hostname'],
                    'trigger': alert_data['trigger'],
                    'severity': alert_data['severity'],
                    'call_id': alert_data['call_id'],
                    'extension': alert_data['extension']
                }
            })
        else:
            logger.warning(f"⚠️ Alert data not found for call_id: {call_id}")
            return jsonify({
                'status': 'error',
                'error': 'Alert data not found for this call_id'
            }), 404

    except Exception as e:
        logger.error(f"❌ Error retrieving alert data: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/check-escalation', methods=['POST', 'GET'])
def check_escalation():
    """Check if alerts need manager escalation"""
    try:
        # Check all pending escalations
        escalations_needed = processor.check_escalation_timeouts()
        escalation_results = []

        if escalations_needed:
            logger.info(f"Found {len(escalations_needed)} alerts requiring escalation")

            for escalation in escalations_needed:
                call_id = escalation['call_id']
                alert_id = escalation['alert_id']
                reason = escalation['reason']
                call_info = escalation['call_info']

                # Get alert data for escalation
                if alert_id in processor.active_alerts:
                    alert_data = processor.active_alerts[alert_id]['data']
                    escalation_result = processor.escalate_to_manager(alert_data, reason)

                    # Mark as escalated
                    processor.call_responses[call_id]['escalated'] = True
                    processor.call_responses[call_id]['escalation_result'] = escalation_result

                    escalation_results.append({
                        'call_id': call_id,
                        'alert_id': alert_id,
                        'reason': reason,
                        'escalation_result': escalation_result
                    })

                    logger.info(f"Escalated call {call_id} (alert {alert_id}) to manager: {reason}")

        return jsonify({
            'status': 'success',
            'escalations_processed': len(escalation_results),
            'escalations': escalation_results,
            'message': f'Processed {len(escalation_results)} escalations'
        })

    except Exception as e:
        logger.error(f" Escalation check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test-alert', methods=['POST'])
def test_alert():
    """Test endpoint for debugging"""
    try:
        test_data = request.json or {}

        test_alert = {
            'alert_id': 'TEST_' + str(int(time.time())),
            'hostname': test_data.get('hostname', 'TEST-SERVER'),
            'trigger_name': test_data.get('trigger_name', 'Test Service Unavailable'),
            'severity': test_data.get('severity', 'high'),
            'description': 'This is a test alert'
        }

        logger.info(f"Test alert: {test_alert}")

        # Process test alert
        result = processor.process_alert(test_alert)

        return jsonify({
            'status': 'success',
            'mode': 'business_hours' if result['business_hours'] else 'after_hours',
            **result
        })

    except Exception as e:
        logger.error(f" Test alert error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Background escalation monitor
import threading

def escalation_monitor():
    """Background thread to monitor for escalation timeouts and failed calls"""
    import time
    cleanup_counter = 0
    while True:
        try:
            # Check for escalations and failed calls every 30 seconds
            time.sleep(30)
            cleanup_counter += 1

            # Memory cleanup every 10 minutes (20 cycles of 30s)
            if cleanup_counter >= 20:
                processor.cleanup_old_data()
                cleanup_counter = 0

            # Check for failed calls and trigger backup calls
            failed_calls = processor.check_failed_calls()
            if failed_calls:
                logger.info(f"Background call monitor: Found {len(failed_calls)} failed calls")

                for failed_call in failed_calls:
                    call_id = failed_call['call_id']
                    extension = failed_call['extension']
                    alert_data = failed_call['alert_data']
                    reason = failed_call['reason']

                    # Find the original technician
                    original_technician = None
                    for tech in processor.technicians:
                        if tech['extension'] == extension:
                            original_technician = tech
                            break

                    if original_technician:
                        logger.info(f"Triggering backup calls for failed call {call_id} to {original_technician['name']}")
                        backup_result = processor.trigger_cascading_calls(alert_data, original_technician, reason)

                        if backup_result.get('success'):
                            logger.info(f"  Backup call successful to {backup_result['technician']['name']}")
                        else:
                            logger.warning(f" All backup calls failed for alert {alert_data.get('alert_id')}")

            # Check for escalation timeouts
            escalations_needed = processor.check_escalation_timeouts()

            if escalations_needed:
                logger.info(f"Background escalation check: Found {len(escalations_needed)} alerts requiring escalation")

                for escalation in escalations_needed:
                    call_id = escalation['call_id']
                    alert_id = escalation['alert_id']
                    reason = escalation['reason']

                    # Get alert data for escalation
                    if alert_id in processor.active_alerts:
                        alert_data = processor.active_alerts[alert_id]['data']
                        escalation_result = processor.escalate_to_manager(alert_data, reason)

                        # Mark as escalated
                        processor.call_responses[call_id]['escalated'] = True
                        processor.call_responses[call_id]['escalation_result'] = escalation_result

                        logger.info(f"Background escalation: Call {call_id} (alert {alert_id}) escalated to manager - {reason}")

        except Exception as e:
            logger.error(f"Background escalation monitor error: {e}")

if __name__ == '__main__':
    logger.info("Starting Zabbix Alert Server - Complete n8n replacement")
    logger.info("Server running on port 9001 - Direct call file creation!")

    # Start background escalation monitor
    escalation_thread = threading.Thread(target=escalation_monitor, daemon=True)
    escalation_thread.start()
    logger.info("Background escalation monitor started")

    # Run on port 9001 (HTTP API functionality moved here)
    app.run(host='0.0.0.0', port=9001, debug=True)