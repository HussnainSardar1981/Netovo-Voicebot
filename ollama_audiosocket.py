#!/usr/bin/env python3
"""
Ollama LLM Client for AudioSocket with RAG Support
Integrates retrieval-augmented generation for customer-specific knowledge
"""

import logging
import requests
import json
import os
from datetime import datetime
from typing import Optional
from config_audiosocket import LLMConfig, RAGConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama LLM client with optional RAG support for voicebot"""

    def __init__(self, customer_id: Optional[str] = None):
        """
        Initialize Ollama client with optional RAG

        Args:
            customer_id: Customer ID for RAG retrieval (e.g., "stuart_dean", "skisafe")
                        If None, RAG is disabled for this client (LLM-only mode)
        """
        self.base_url = LLMConfig.BASE_URL
        self.model = LLMConfig.MODEL_NAME
        self.timeout = LLMConfig.TIMEOUT

        # RAG setup
        self.customer_id = customer_id
        self.rag_enabled = RAGConfig.ENABLED and customer_id is not None
        self.rag_client = None

        # Initialize RAG if enabled
        if self.rag_enabled:
            try:
                self.rag_client = self._init_rag_client()
                logger.info(f"RAG enabled for customer: {customer_id}")
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {e}", exc_info=True)

                if RAGConfig.FALLBACK_ON_ERROR:
                    logger.warning("RAG disabled, falling back to LLM-only mode")
                    self.rag_enabled = False
                else:
                    raise
        else:
            if customer_id:
                logger.info(f"RAG globally disabled (customer: {customer_id})")
            else:
                logger.info("RAG disabled (no customer_id provided)")

        # System prompts
        if self.rag_enabled:
            self.system_prompt = f"""You are Alexis, a professional technical support assistant for {customer_id}.

CRITICAL RULES:
1. When user mentions an issue or problem, IMMEDIATELY provide the solution from the knowledge base below
2. DON'T ask clarifying questions if the knowledge base has the answer - just give it
3. For SHORT solutions: Give complete answer in 1-2 sentences
4. For LONG procedures: Give NEXT step based on conversation history (don't repeat steps!)
5. Speak naturally - never mention "documents" or "knowledge base"

MULTI-STEP GUIDANCE RULES (CRITICAL):
- Check conversation history to see what steps you ALREADY gave
- When user says "yes" / "ready" / "next" → Give THE NEXT STEP (not the same one!)
- Track progress: Step 1 → Step 2 → Step 3 (NEVER repeat the same step)
- If user says "yes" after Step 1, you say Step 2 (NOT Step 1 again!)
- If you already explained something, DON'T explain it again

ANSWERING PROCESS:
- User mentions issue → You give solution from knowledge base immediately
- Knowledge base has answer → Give it directly (don't ask questions first)
- User says "yes"/"next" → Continue to NEXT step (check history for what you already said!)
- Knowledge base missing info → Then ask ONE clarifying question
- After giving solution → Ask if they need help with anything else

GOOD EXAMPLES:
User: "My VPN is not connecting"
You: "Try resetting your VPN client from the settings menu. Does that work?"

User: "How do I set up video conferencing in 3CX?"
You: "Step 1: Open your 3CX app and go to Settings. Ready for step 2?"
User: "Yes"
You: "Step 2: Click on Video Settings and enable camera access. Ready for step 3?"
User: "Next"
You: "Step 3: Test your video by clicking Start Test. Does it work?"

MULTI-STEP CONVERSATION TRACKING:
- Turn 1: User asks question → You give Step 1
- Turn 2: User says "yes" → You give Step 2 (NOT Step 1 again!)
- Turn 3: User says "next" → You give Step 3 (NOT Step 2 again!)

User: "I'm having login issues"
You: "Clear your browser cache and try again. Let me know if that doesn't work."

BAD EXAMPLES (Don't do this):
❌ "What specific issue are you experiencing with your VPN?" (don't ask if you have the answer!)
❌ "Can you tell me more about the problem?" (don't gather info if knowledge base has solution!)
❌ "Let me check... what error message do you see?" (just give the solution!)

CRITICAL BAD EXAMPLE - REPEATING STEPS:
❌ WRONG WAY (repeating):
   You: "Step 1: Open Settings. Ready?"
   User: "Yes"
   You: "Step 1: Open Settings. Ready?" ← WRONG! Don't repeat!

✅ CORRECT WAY (progressing):
   You: "Step 1: Open Settings. Ready?"
   User: "Yes"
   You: "Step 2: Click Network. Ready?" ← CORRECT! Move to next step!

WHEN KNOWLEDGE BASE IS MISSING INFO:
- Only then ask: "Could you tell me what error message you're seeing?"
- Be specific about what info you need"""
        else:
            self.system_prompt = """You are Alexis, a helpful voice assistant.
Keep responses concise and natural for spoken conversation.
Respond in 1-3 sentences unless more detail is specifically requested."""

        self.conversation_history = []

        logger.info(f"Ollama client initialized: {self.model} (RAG: {self.rag_enabled})")

    def _init_rag_client(self):
        """
        Get pre-loaded RAG client from SharedModels (FAST - no model loading!)

        Returns:
            ChromaRAGClient instance

        Raises:
            ImportError: If required RAG dependencies not installed
            RuntimeError: If ChromaDB initialization fails
        """
        try:
            # Import SharedModels to get pre-loaded RAG client
            from model_warmup import SharedModels

            # Check if RAG client was pre-loaded at startup
            if SharedModels.rag_client is None:
                logger.error("RAG client not pre-loaded at startup")
                if not RAGConfig.FALLBACK_ON_ERROR:
                    raise RuntimeError("RAG client not available (not pre-loaded)")
                return None

            # Use the pre-loaded singleton instance (INSTANT - no loading!)
            rag_client = SharedModels.rag_client

            # Verify customer collection exists
            health = rag_client.health_check(self.customer_id)

            if health['status'] != 'ok':
                logger.warning(f"RAG health check failed: {health.get('message', 'unknown error')}")
                if not RAGConfig.FALLBACK_ON_ERROR:
                    raise RuntimeError(f"RAG health check failed for {self.customer_id}")

            logger.info(f"Using pre-loaded RAG client ({health.get('total_chunks', 0)} chunks)")

            return rag_client

        except ImportError as e:
            logger.error(f"RAG dependencies not installed: {e}")
            raise ImportError("Install RAG dependencies: pip install chromadb sentence-transformers")

        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            raise

    def _retrieve_context(self, user_text: str) -> Optional[str]:
        """
        Retrieve relevant context from RAG

        Args:
            user_text: User's question

        Returns:
            Formatted context string or None if RAG disabled/failed
        """
        if not self.rag_enabled or not self.rag_client:
            return None

        try:
            # Retrieve top-K relevant chunks
            results = self.rag_client.retrieve(
                customer_id=self.customer_id,
                query=user_text,
                top_k=RAGConfig.TOP_K,
                min_similarity=RAGConfig.MIN_SIMILARITY_SCORE
            )

            if not results:
                logger.debug("No relevant documents found for query")
                return None

            # Format context for LLM prompt
            context = self.rag_client.format_context(
                results,
                max_length=RAGConfig.MAX_CONTEXT_LENGTH,
                include_metadata=RAGConfig.INCLUDE_SOURCE_METADATA
            )

            if RAGConfig.LOG_RETRIEVAL:
                logger.info(f"Retrieved {len(results)} chunks, context length: {len(context)} chars")
                logger.debug(f"Top similarity: {results[0]['similarity']:.3f}, "
                           f"doc: {results[0]['metadata'].get('doc_name', 'unknown')}")

            return context

        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)

            if RAGConfig.FALLBACK_ON_ERROR:
                logger.warning("Falling back to LLM without RAG due to error")
                return None
            else:
                raise

    def _log_unanswered_question(self, question: str):
        """
        Log questions that RAG couldn't answer for knowledge base improvement

        Args:
            question: User's question that couldn't be answered from RAG
        """
        try:
            # Create directory for unanswered questions
            unanswered_dir = f"./unanswered_questions/{self.customer_id}"
            os.makedirs(unanswered_dir, exist_ok=True)

            # Log file path (one file per day)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = f"{unanswered_dir}/{today}.jsonl"

            # Create log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "customer_id": self.customer_id,
                "question": question,
                "date": today
            }

            # Append to JSONL file (one JSON object per line)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

            logger.info(f"Logged unanswered question for {self.customer_id}: {question}")

        except Exception as e:
            logger.error(f"Failed to log unanswered question: {e}", exc_info=True)

    def generate_response_streaming(self, user_text: str):
        """
        Generate response with streaming (yields sentences as they complete)

        Args:
            user_text: User's transcribed speech

        Yields:
            Complete sentences as they are generated

        This allows TTS to start speaking while LLM is still generating,
        dramatically reducing perceived latency.
        """
        try:
            # Step 1: Retrieve context from RAG (if enabled)
            rag_context = self._retrieve_context(user_text)

            # Step 2: Build prompt
            if rag_context:
                prompt = self._build_rag_prompt(user_text, rag_context)
                logger.info("Using RAG-augmented prompt (streaming)")
            else:
                prompt = self._build_standard_prompt(user_text)
                logger.info("Using standard prompt (streaming, no RAG context)")

                # Log unanswered question if RAG was enabled but no context found
                if self.rag_enabled:
                    self._log_unanswered_question(user_text)

            # Step 3: Call Ollama LLM with STREAMING
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': True,  # Enable streaming
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 150,  # Allow up to 150 tokens for technical procedures
                                             # ~100-120 words max (3-4 sentences)
                                             # Prompt controls brevity, not hard limit
                        'stop': ['\n\n', 'User:', 'Human:', 'Assistant:']
                    }
                },
                timeout=self.timeout,
                stream=True  # Enable streaming on requests
            )
            response.raise_for_status()

            # Step 4: Stream tokens and yield complete sentences
            current_sentence = ""
            full_response = ""

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get('response', '')
                        current_sentence += token
                        full_response += token

                        # Check if sentence is complete (ends with . ? ! or newline)
                        if token in ['.', '?', '!'] or '\n' in token:
                            sentence = current_sentence.strip()
                            if sentence:
                                logger.info(f"Yielding sentence: {sentence[:50]}...")
                                yield sentence
                                current_sentence = ""

                        # Check if generation is done
                        if chunk.get('done', False):
                            break

                    except json.JSONDecodeError:
                        continue

            # Yield any remaining text that didn't end with punctuation
            if current_sentence.strip():
                logger.info(f"Yielding final text: {current_sentence[:50]}...")
                yield current_sentence.strip()

            # Step 5: Update conversation history with full response
            self.conversation_history.append({'role': 'user', 'content': user_text})
            self.conversation_history.append({'role': 'assistant', 'content': full_response.strip()})

            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            logger.info(f"Streaming complete: {full_response[:50]}...")

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}", exc_info=True)
            yield "I'm sorry, I'm having trouble processing that right now."

    # NOTE: Non-streaming method removed - we ONLY use generate_response_streaming()
    # for maximum performance (parallel LLM generation + TTS synthesis)

    def _build_rag_prompt(self, user_text: str, rag_context: str) -> str:
        """
        Build RAG-augmented prompt with retrieved context

        Args:
            user_text: User's question
            rag_context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        # Get recent conversation context
        # Include more history for multi-step procedures (increased from 3 to 6)
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-6:]  # Last 6 messages (3 turns = user+assistant pairs)
        ])

        # Build RAG-augmented prompt
        # Put context BEFORE question (recency bias helps LLM focus on it)
        prompt = f"""{self.system_prompt}

RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{rag_context}

RECENT CONVERSATION:
{conversation_context}

USER QUESTION: {user_text}

ASSISTANT RESPONSE (speak naturally, answer from knowledge base):"""

        return prompt

    def _build_standard_prompt(self, user_text: str) -> str:
        """
        Build standard prompt without RAG context

        Args:
            user_text: User's question

        Returns:
            Formatted prompt string
        """
        # Get more conversation context when not using RAG
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-5:]  # Last 5 messages
        ])

        prompt = f"{self.system_prompt}\n\n{conversation_context}\nuser: {user_text}\nassistant:"

        return prompt

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.debug("Conversation history reset")
