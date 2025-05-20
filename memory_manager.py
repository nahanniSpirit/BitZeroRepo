"""
BitZero Memory Manager

This module implements persistent local memory and conversation history for BitZero.
It stores conversations in a SQLite database and provides memory retrieval functionality.

MODIFIED:
    - Added placeholder for text embedding generation.
    - Added 'embedding' column to 'memories' table (TEXT, stores JSON string of list).
    - Implemented `get_all_memories` placeholder.
    - Implemented `cosine_similarity`.
    - Implemented `retrieve_memories_by_semantic_similarity` (Rec 3.1).
    - Updated `store_memory` to generate and store embeddings.
    - Updated `retrieve_memories` and `search_memories` to fetch and parse embeddings.
    - Added EmotionalIntelligence class and integrated basic emotion detection into memory storage (Rec 6.1).
    - Added memory consolidation and temporal weighting.
"""

import sqlite3
import json
import os
import time
import torch 
import torch.nn.functional as F 
import numpy as np 
from typing import Dict, List, Tuple, Optional, Any
import datetime
import re
import random # For empathetic response selection
import math


# --- Recommendation 6.1: Emotional Intelligence Framework ---
class EmotionalIntelligence:
    """Framework for emotional intelligence capabilities."""
    def __init__(self, memory_manager_instance: 'MemoryManager'): # Forward reference
        self.memory_manager = memory_manager_instance
        self.emotion_categories = [ # As per OCR
            "joy", "sadness", "anger", "fear", "surprise",
            "disgust", "trust", "anticipation"
        ]
        # Simple keyword-based emotion detection
        self.emotion_keywords = { # As per OCR
            "joy": ["happy", "glad", "excited", "pleased", "delighted", "joy", "wonderful", "great", "awesome", "love this", "excellent"],
            "sadness": ["sad", "unhappy", "depressed", "down", "blue", "upset", "sorry to hear", "miserable"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "hate this"],
            "fear": ["afraid", "scared", "worried", "anxious", "nervous", "terrified"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "wow", "omg", "unbelievable"],
            "disgust": ["disgusted", "repulsed", "revolted", "sick", "gross"],
            "trust": ["trust", "believe", "rely", "depend", "confident", "sure"],
            "anticipation": ["looking forward", "excited about", "can't wait", "eager", "expecting"]
        }
        self.response_templates = { # As per OCR, expanded slightly
            "joy": [
                "I'm glad to hear that you're feeling {emotion}!",
                "That's wonderful! I'm happy for you.",
                "It's great to see you in such good spirits."
            ],
            "sadness": [
                "I'm sorry to hear that you're feeling {emotion}.",
                "That sounds difficult. I'm here if you want to talk about it.",
                "I understand that can be hard. How can I help?",
                "It's okay to feel {emotion}. Take your time."
            ],
            "anger": [
                "It sounds like you're feeling {emotion}. I understand that can be frustrating.",
                "I'm here to listen if you need to vent about why you're {emotion}.",
                "Feeling {emotion} is valid. What's on your mind?"
            ],
            "fear": [
                "It's understandable to feel {emotion} in this situation.",
                "I'm here for you. Is there anything I can do to help you feel less {emotion}?",
                "Feeling {emotion} can be tough. Remember to breathe."
            ],
            "surprise": [
                "Wow, that sounds {emotion}!",
                "That's quite a {emotion}! Tell me more.",
                "I'm {emotion} to hear that!"
            ],
            # Default/Fallback templates
            "default": [
                 "I notice you're feeling {emotion}.",
                 "Thank you for sharing how you feel.",
                 "I understand you're experiencing {emotion}."
            ]
        }


    def detect_user_emotion(self, text: str) -> Dict[str, int]:
        """Detect emotional content in user input. Returns a dict of emotion:score."""
        detected_emotions = {}
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                detected_emotions[emotion] = score
        return detected_emotions

    def generate_empathetic_response_snippet(self, detected_emotions: Dict[str, int]) -> Optional[str]:
        """
        Generate a simple empathetic response snippet based on the dominant detected emotion.
        This is a snippet, not a full conversational turn.
        """
        if not detected_emotions:
            return None 

        # Find dominant emotion (highest score)
        if not detected_emotions: return None
        dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0]
        
        templates = self.response_templates.get(dominant_emotion, self.response_templates["default"])
        response_snippet = random.choice(templates).format(emotion=dominant_emotion)
        return response_snippet

    def log_detected_emotion(self, user_id: str, conversation_id: Optional[str], text_input: str, detected_emotions: Dict[str, int]):
        """Logs the dominant detected emotion to memory."""
        if not detected_emotions:
            return

        dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0]
        intensity = detected_emotions[dominant_emotion]
        timestamp = datetime.datetime.now().isoformat()

        self.memory_manager.store_memory(
            category="user_emotion_log", # Using this category from existing code
            content=f"User '{user_id}' expressed {dominant_emotion} (intensity {intensity}) in conversation {conversation_id} at {timestamp}. Input: '{text_input[:100]}...'",
            importance=0.6 + (intensity * 0.05), # Importance slightly based on intensity
            metadata={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "source_text": text_input,
                "detected_emotion_profile": detected_emotions, # Store full profile
                "dominant_emotion": dominant_emotion,
                "intensity": intensity
            }
        )
# --- End Recommendation 6.1 ---


class MemoryManager:
    """Memory manager for BitZero with local storage and retrieval."""
    EMBEDDING_DIM = 128 

    def __init__(self, db_path: str = "bitzero_memory.db"):
        self.db_path = db_path
        self.initialize_db()
        # Instantiate EmotionalIntelligence framework
        self.emotional_intelligence = EmotionalIntelligence(memory_manager_instance=self)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        if not text: return [0.0] * self.EMBEDDING_DIM
        vec = np.zeros(self.EMBEDDING_DIM)
        for i, char_val in enumerate(text):
            idx = (hash(char_val) + i) % self.EMBEDDING_DIM
            vec[idx] = (vec[idx] + (ord(char_val) / 128.0) - 0.5) % 1.0 
        norm = np.linalg.norm(vec)
        return list(vec / norm) if norm != 0 else list(vec)

    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_session_id TEXT NOT NULL,
            user_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL, timestamp TEXT NOT NULL, metadata TEXT ) ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT,
            embedding TEXT, importance REAL, last_accessed TEXT,
            access_count INTEGER, metadata TEXT ) ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY, value TEXT, last_updated TEXT ) ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interaction_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, turn_id INTEGER, 
            user_id TEXT, feedback_score INTEGER, feedback_text TEXT, timestamp TEXT ) ''')
        conn.commit(); conn.close()
    
    def get_or_create_conversation_session(self, user_id: str = "default_user") -> str:
        return f"{user_id}_{datetime.datetime.now().timestamp()}"

    def add_turn_to_conversation(self, conversation_session_id: str, user_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        cursor.execute(
            "INSERT INTO conversations (conversation_session_id, user_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_session_id, user_id, role, content, timestamp, metadata_json)
        )
        turn_id = cursor.lastrowid
        conn.commit(); conn.close()
        return turn_id

    def get_recent_turns_for_prompt(self, conversation_session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp FROM conversations WHERE conversation_session_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_session_id, limit)
        )
        rows = cursor.fetchall()
        turns = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]} for row in reversed(rows)]
        conn.close(); return turns

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        search_param = f"%{query}%"
        cursor.execute(
            "SELECT id, conversation_session_id, user_id, role, content, timestamp, metadata FROM conversations WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (search_param, limit)
        )
        rows = cursor.fetchall()
        matching_turns = []
        for row in rows:
            turn = dict(row)
            turn["metadata"] = json.loads(turn["metadata"]) if turn["metadata"] else {}
            matching_turns.append(turn)
        conn.close(); return matching_turns
    
    def store_memory(self, category: str, content: str, importance: float = 0.5, metadata: Optional[Dict[str, Any]] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"
        embedding = self._get_text_embedding(content)
        embedding_json = json.dumps(embedding)
        cursor.execute(
            "INSERT INTO memories (category, content, embedding, importance, last_accessed, access_count, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (category, content, embedding_json, importance, timestamp, 1, metadata_json)
        )
        conn.commit(); conn.close()

    def _parse_memory_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        memory = dict(row)
        memory["metadata"] = json.loads(memory["metadata"]) if memory["metadata"] else {}
        try: memory["embedding"] = json.loads(memory["embedding"]) if memory["embedding"] else None
        except (json.JSONDecodeError, TypeError): memory["embedding"] = None
        return memory

    def retrieve_memories(self, category: Optional[str] = None, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        base_query, conditions, params_list = "SELECT * FROM memories", [], [] # Renamed params to params_list
        if category: conditions.append("category = ?"); params_list.append(category)
        if user_id: conditions.append("metadata LIKE ?"); params_list.append(f'%"user_id": "{user_id}"%')
        if conditions: base_query += " WHERE " + " AND ".join(conditions)
        base_query += " ORDER BY importance DESC, access_count DESC, last_accessed DESC LIMIT ?"; params_list.append(limit)
        cursor.execute(base_query, tuple(params_list))
        rows = cursor.fetchall()
        memories = []
        for row in rows:
            memories.append(self._parse_memory_row(row))
            cursor.execute( "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (datetime.datetime.now().isoformat(), row["id"]) )
        conn.commit(); conn.close(); return memories
    
    def search_memories(self, query: str, limit: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        base_query, params_list = "SELECT * FROM memories WHERE content LIKE ?", [f"%{query}%"] # Renamed params to params_list
        if user_id: base_query += " AND metadata LIKE ?"; params_list.append(f'%"user_id": "{user_id}"%')
        base_query += " ORDER BY importance DESC, access_count DESC, last_accessed DESC LIMIT ?"; params_list.append(limit)
        cursor.execute(base_query, tuple(params_list))
        rows = cursor.fetchall()
        memories = []
        for row in rows:
            memories.append(self._parse_memory_row(row))
            cursor.execute( "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (datetime.datetime.now().isoformat(), row["id"]) )
        conn.commit(); conn.close(); return memories

    def update_memory(self, memory_id: int, content: Optional[str] = None, importance: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Updates specific fields of an existing memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fields_to_update = []
        params_list = [] 

        if content is not None:
            fields_to_update.append("content = ?")
            params_list.append(content)
            new_embedding = self._get_text_embedding(content)
            fields_to_update.append("embedding = ?")
            params_list.append(json.dumps(new_embedding))

        if importance is not None:
            fields_to_update.append("importance = ?")
            params_list.append(importance)
        
        if metadata is not None:
            fields_to_update.append("metadata = ?")
            params_list.append(json.dumps(metadata))
        
        if not fields_to_update:
            conn.close()
            return

        fields_to_update.append("last_accessed = ?")
        params_list.append(datetime.datetime.now().isoformat())
        params_list.append(memory_id)

        query = f"UPDATE memories SET {', '.join(fields_to_update)} WHERE id = ?"
        cursor.execute(query, tuple(params_list))
        conn.commit()
        conn.close()

    def delete_memory(self, memory_id: int):
        """Deletes a memory by its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        conn.close()

    def get_all_memories(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query, params_list = "SELECT * FROM memories", [] # Renamed params to params_list
        if user_id: query += " WHERE metadata LIKE ?"; params_list.append(f'%"user_id": "{user_id}"%')
        cursor.execute(query, tuple(params_list))
        rows = cursor.fetchall()
        memories = [self._parse_memory_row(row) for row in rows]
        conn.close(); return memories

    def _cosine_similarity_numpy(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2): return 0.0
        v1, v2 = np.array(vec1), np.array(vec2)
        dot_product, norm_v1, norm_v2 = np.dot(v1, v2), np.linalg.norm(v1), np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2) if norm_v1 != 0 and norm_v2 != 0 else 0.0
    
    # Recommendation 3.2: Memory Consolidation
    def calculate_similarity(self, text1: str, text2: str) -> float: # Helper for consolidation
        """Calculates semantic similarity between two texts using their embeddings."""
        if not text1 or not text2: return 0.0
        emb1 = self._get_text_embedding(text1)
        emb2 = self._get_text_embedding(text2)
        return self._cosine_similarity_numpy(emb1, emb2)

    def consolidate_memories(self, category: Optional[str] = None, threshold: float = 0.85, user_id: Optional[str] = None):
        """Consolidate similar memories to prevent redundancy."""
        # Retrieve memories, potentially filtered by user_id for user-specific consolidation
        memories_to_check = self.retrieve_memories(category=category, limit=1000, user_id=user_id)
        
        consolidated_groups: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        processed_memory_ids = set()

        for i in range(len(memories_to_check)):
            if memories_to_check[i]['id'] in processed_memory_ids:
                continue

            primary_memory = memories_to_check[i]
            similar_to_primary: List[Dict[str, Any]] = []
            processed_memory_ids.add(primary_memory['id'])

            for j in range(i + 1, len(memories_to_check)):
                if memories_to_check[j]['id'] in processed_memory_ids:
                    continue
                
                # Use embedding similarity if available, else fall back to content similarity
                similarity = 0.0
                if primary_memory.get('embedding') and memories_to_check[j].get('embedding'):
                    similarity = self._cosine_similarity_numpy(primary_memory['embedding'], memories_to_check[j]['embedding'])
                else: # Fallback if embeddings are missing for some reason
                    similarity = self.calculate_similarity(primary_memory['content'], memories_to_check[j]['content'])

                if similarity > threshold:
                    similar_to_primary.append(memories_to_check[j])
                    processed_memory_ids.add(memories_to_check[j]['id'])
            
            if similar_to_primary: # Only add to groups if there are similar items to consolidate
                 consolidated_groups.append((primary_memory, similar_to_primary))

        # Merge similar memories
        for primary_memory, similar_memories_list in consolidated_groups:
            if similar_memories_list: # Ensure there's something to merge
                all_related_memories = [primary_memory] + similar_memories_list
                
                # Update importance: average or max, let's use max for now to keep strongest signal
                new_importance = max(m['importance'] for m in all_related_memories)
                
                # Update content: choose longest or combine. For now, choose longest.
                # A more sophisticated merge could be implemented.
                new_content = primary_memory['content']
                for m in similar_memories_list:
                    if len(m['content']) > len(new_content):
                        new_content = m['content']
                
                # Combine metadata (simple merge, could be more sophisticated)
                # For now, just use primary memory's metadata.
                # A better approach might merge unique keys or specific fields.
                new_metadata = primary_memory.get('metadata', {})


                print(f"Consolidating memory ID {primary_memory['id']} with {len(similar_memories_list)} other memories.")
                self.update_memory(primary_memory['id'], 
                                  content=new_content,
                                  importance=new_importance,
                                  metadata=new_metadata) # Pass updated metadata
                
                for m_to_delete in similar_memories_list:
                    self.delete_memory(m_to_delete['id'])
        if consolidated_groups:
            print(f"Memory consolidation finished. Processed {len(consolidated_groups)} groups for potential merging.")
        else:
            print("No memories found to consolidate with the given criteria.")


    def retrieve_memories_by_semantic_similarity(self, query_text: str, limit: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = self._get_text_embedding(query_text)
        if not query_embedding or all(v == 0 for v in query_embedding): return []
        all_memories_for_user = self.get_all_memories(user_id=user_id)
        scored_memories = []
        for memory in all_memories_for_user:
            memory_embedding = memory.get("embedding")
            if not memory_embedding or not isinstance(memory_embedding, list) or len(memory_embedding) != self.EMBEDDING_DIM: continue
            similarity = self._cosine_similarity_numpy(query_embedding, memory_embedding)
            scored_memories.append((memory, similarity))
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [mem_tuple[0]["id"] for mem_tuple in scored_memories[:limit]]
        if retrieved_ids:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for mem_id in retrieved_ids:
                cursor.execute( "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (datetime.datetime.now().isoformat(), mem_id) )
            conn.commit(); conn.close()
        return [mem_tuple[0] for mem_tuple in scored_memories[:limit]]

    def set_preference(self, key: str, value: Any):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        value_json = json.dumps(value)
        cursor.execute( "INSERT OR REPLACE INTO preferences (key, value, last_updated) VALUES (?, ?, ?)", (key, value_json, timestamp) )
        conn.commit(); conn.close()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close(); return json.loads(row[0]) if row else default
    
    def extract_memories_from_conversation(self, user_input: str, model_response: str, user_id: str = "default_user", conversation_id: Optional[str] = None):
        """
        Modified to use EmotionalIntelligence framework for emotion detection and logging.
        Other memory extraction logic remains.
        """
        # --- Emotion Detection and Logging (Rec 6.1 integration) ---
        if user_input: # Detect emotion from user input
            detected_emotions_user = self.emotional_intelligence.detect_user_emotion(user_input)
            if detected_emotions_user:
                self.emotional_intelligence.log_detected_emotion(user_id, conversation_id, user_input, detected_emotions_user)
        # --- End Emotion Detection ---

        # Existing memory extraction logic
        if user_input:
            user_metadata = {"user_id": user_id, "conversation_id": conversation_id, "source_text": user_input, "source_role": "user"}
            if "I like" in user_input or "I prefer" in user_input or "I love" in user_input:
                self.store_memory("user_preference", user_input, importance=0.8, metadata=user_metadata)
            if "my name is" in user_input.lower():
                name_match = re.search(r"my name is\s*([\w.-]+)", user_input, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1)
                    self.store_memory("user_fact", f"User '{user_id}' said their name is {name}", importance=0.9, metadata=user_metadata)
                    self.set_preference(f"{user_id}_name", name)
            
            goal_keywords = ["i want to ", "my goal is to ", "i need to ", "i plan to ", "i'd like to "]
            for g_kw in goal_keywords:
                if g_kw in user_input.lower():
                    try:
                        goal_content = user_input.lower().split(g_kw, 1)[1].strip()
                        if goal_content:
                            self.store_memory(category="user_goal", content=f"User '{user_id}' stated goal: {goal_content}", importance=0.9, metadata=user_metadata)
                            break 
                    except IndexError: pass

        if model_response: 
            assistant_metadata = {"user_id": user_id, "conversation_id": conversation_id, "source_text": model_response, "source_role": "assistant"}
            if "the capital of" in model_response.lower() and " is " in model_response.lower():
                 self.store_memory("stated_fact_by_assistant", model_response, importance=0.7, metadata=assistant_metadata)

    def extract_and_store_memories_from_interaction(self, user_id: str, conversation_id: str, user_text: str, assistant_text: str):
        self.extract_memories_from_conversation(user_input=user_text, model_response="", user_id=user_id, conversation_id=conversation_id)
        self.extract_memories_from_conversation(user_input=user_text, model_response=assistant_text, user_id=user_id, conversation_id=conversation_id)

    def get_context_summary(self, user_id: str, conversation_id: str, limit_goals: int = 2, limit_emotions: int = 1) -> str:
        context_parts = []
        user_name = self.get_preference(f"{user_id}_name", user_id)
        context_parts.append(f"[System Info: User '{user_name}'. Session: {conversation_id}]")

        user_goals = self.retrieve_memories(category="user_goal", limit=limit_goals, user_id=user_id)
        if user_goals:
            context_parts.append("[Recent User Goals:]")
            for goal in user_goals:
                goal_text = goal['content']
                prefix = f"User '{user_id}' stated goal: "
                context_parts.append(f"- {goal_text[len(prefix):] if goal_text.startswith(prefix) else goal_text}")
        
        # Fetch from 'user_emotion_log' which is now populated by EmotionalIntelligence
        emotion_logs = self.retrieve_memories(category="user_emotion_log", limit=limit_emotions, user_id=user_id)
        if emotion_logs:
            # The content string for user_emotion_log is now more detailed.
            # Example: "User 'default_user' expressed joy (intensity 1) in conversation..."
            # We can parse the dominant emotion from the content or metadata.
            # Let's try parsing from metadata if available, else from content.
            log_entry = emotion_logs[0] # Most recent
            emotion_to_display = "Unknown"
            if log_entry.get("metadata") and log_entry["metadata"].get("dominant_emotion"):
                emotion_to_display = log_entry["metadata"]["dominant_emotion"]
            else: # Fallback to regex parsing from content if needed
                match = re.search(r"expressed (\w+)", log_entry['content'])
                if match: emotion_to_display = match.group(1)
            context_parts.append(f"[Recent User Emotion: {emotion_to_display}.]")
        
        return "\n".join(context_parts) if len(context_parts) > 1 else ""

    # Recommendation 3.3: Temporal Memory Weighting
    def generate_context_from_memory(self, user_input: str, for_user_id: str = "default_user", conversation_id: Optional[str] = None, max_items: int = 3, use_semantic_retrieval: bool = True) -> str:
        """Generate context with temporal weighting of memories."""
        context_parts = []
        summary_context = self.get_context_summary(user_id=for_user_id, conversation_id=conversation_id if conversation_id else "N/A")
        if summary_context: context_parts.append(summary_context)

        # Retrieve a larger pool of memories to apply temporal weighting
        # If using semantic, it already considers relevance. If not, search_memories does.
        # We'll fetch more than max_items initially.
        initial_retrieval_limit = max_items * 3 
        
        if use_semantic_retrieval:
            relevant_memories_pool = self.retrieve_memories_by_semantic_similarity(query_text=user_input, limit=initial_retrieval_limit, user_id=for_user_id)
        else:
            # Fallback to keyword search if semantic is not preferred or fails
            # For keyword search, we might want to broaden the query if user_input is short
            # For now, direct search:
            relevant_memories_pool = self.search_memories(query=user_input, limit=initial_retrieval_limit, user_id=for_user_id)
            if not relevant_memories_pool and len(user_input.split()) > 3: # If keyword search yields nothing, try with broader category retrieval
                relevant_memories_pool = self.retrieve_memories(limit=initial_retrieval_limit, user_id=for_user_id)


        if not relevant_memories_pool:
            return "\n".join(context_parts) if context_parts else ""

        current_time = datetime.datetime.now()
        weighted_memories = []
        for memory in relevant_memories_pool:
            try:
                memory_time_str = memory.get('last_accessed', memory.get('timestamp')) # Prefer last_accessed, fallback to creation timestamp
                if not memory_time_str: # Should not happen if DB is consistent
                    memory_time = current_time 
                else:
                    memory_time = datetime.datetime.fromisoformat(memory_time_str)
            except (ValueError, TypeError): # Handle cases where timestamp might be malformed or missing
                memory_time = current_time # Assume current time if parsing fails

            days_old = (current_time - memory_time).total_seconds() / (24 * 60 * 60) # More precise days_old
            
            # Exponential decay: recency_weight decreases as days_old increases
            # Adjust decay_rate: smaller value = slower decay, larger value = faster decay
            decay_rate = 0.1 
            recency_weight = math.exp(-decay_rate * days_old)
            
            # Combine with importance (ensure importance is float)
            memory_importance = float(memory.get('importance', 0.5)) # Default importance if missing
            final_weight = 0.7 * memory_importance + 0.3 * recency_weight
            
            weighted_memories.append((memory, final_weight))
        
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m_tuple[0] for m_tuple in weighted_memories[:max_items]]
        
        if top_memories:
            context_parts.append("[Relevant Memories (temporally weighted):]")
            for memory in top_memories:
                # Display how old the memory is for context, if desired
                try:
                    mem_time_str_display = memory.get('last_accessed', memory.get('timestamp'))
                    mem_time_display = datetime.datetime.fromisoformat(mem_time_str_display)
                    days_old_display = (current_time - mem_time_display).days
                    age_display = f"{days_old_display}d old" if days_old_display > 0 else "today"
                    context_parts.append(f"- {memory['content']} (Importance: {memory.get('importance', 0.0):.2f}, Age: {age_display})")
                except: # Fallback if timestamp parsing fails for display
                    context_parts.append(f"- {memory['content']} (Importance: {memory.get('importance', 0.0):.2f})")

        return "\n".join(context_parts) if context_parts else ""


    def store_interaction_feedback(self, conversation_id: str, turn_id: int, user_id: str, feedback_score: int, feedback_text: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO interaction_feedback (conversation_id, turn_id, user_id, feedback_score, feedback_text, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, turn_id, user_id, feedback_score, feedback_text, timestamp)
        )
        conn.commit(); conn.close()

    def end_conversation_session(self, conversation_session_id: str):
        print(f"Conversation session {conversation_session_id} ended.")

# ... (BitZeroMemoryChat class remains unchanged for now) ...
class BitZeroMemoryChat:
    def __init__(self, checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        import torch 
        from hybrid_precision import create_bitzero_nano_hybrid 
        self.device = device
        print(f"Initializing BitZero Memory Chat (legacy?) on {device}...")
        self.model = create_bitzero_nano_hybrid(critical_ratio=0.1) 
        self.model.to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.memory_manager = MemoryManager() 
        print("BitZero Memory Chat (legacy?) initialized.")
    
    def tokenize(self, text, max_seq_len=512):
        import torch 
        tokens = [ord(char) % 256 for char in text] 
        tokens = tokens[:max_seq_len] if len(tokens) > max_seq_len else tokens + [0] * (max_seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def detokenize(self, token_ids):
        return "".join([chr(token_id % 256) for token_id in token_ids.squeeze().tolist() if token_id > 0])
    
    def generate_response(self, user_input, max_new_tokens=100, temperature=0.7):
        import torch 
        import torch.nn.functional as F 
        
        detected_emotions = self.memory_manager.emotional_intelligence.detect_user_emotion(user_input)
        empathetic_snippet = None
        if detected_emotions:
            self.memory_manager.emotional_intelligence.log_detected_emotion("legacy_user", "legacy_session", user_input, detected_emotions)
            empathetic_snippet = self.memory_manager.emotional_intelligence.generate_empathetic_response_snippet(detected_emotions)

        self.memory_manager.extract_memories_from_conversation(user_input, "", user_id="legacy_user", conversation_id="legacy_session") 
        # Use the updated generate_context_from_memory for the BitZeroMemoryChat class as well
        memory_context = self.memory_manager.generate_context_from_memory(user_input, for_user_id="legacy_user", conversation_id="legacy_session", use_semantic_retrieval=True) 
        
        prompt = ""
        if empathetic_snippet: 
            prompt += empathetic_snippet + "\n"
        prompt += f"{memory_context}\nUser: {user_input}\nBitZero: "

        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                top_k_val = 40
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_val)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1)]
                input_ids = torch.cat([input_ids, next_token_id.view(1, 1)], dim=1)
                if next_token_id.item() == 0 or chr(next_token_id.item() % 256) == '\n': break
        
        initial_prompt_ids_unpadded_len = 0
        # Correctly get length of tokenized prompt for slicing
        temp_prompt_tokens = self.tokenize(prompt)[0]
        for t_id in temp_prompt_tokens:
            if t_id.item() == 0: break # Assuming 0 is PAD for this legacy tokenizer
            initial_prompt_ids_unpadded_len +=1
        
        generated_tokens_tensor = input_ids[0, initial_prompt_ids_unpadded_len:]
        response = self.detokenize(generated_tokens_tensor) 
        
        temp_session_id = self.memory_manager.get_or_create_conversation_session("legacy_user")
        self.memory_manager.add_turn_to_conversation(temp_session_id, "legacy_user", "user", user_input)
        self.memory_manager.add_turn_to_conversation(temp_session_id, "legacy_user", "assistant", response)
        self.memory_manager.extract_memories_from_conversation(user_input, response, user_id="legacy_user", conversation_id=temp_session_id)
        return response
    
    def chat(self):
        print("Welcome to BitZero Memory Chat (legacy version)! Type 'exit' to end.")
        while True:
            user_input = input("You (legacy): ")
            if user_input.lower() in ["exit", "quit"]: break
            print("BitZero (legacy) thinking...")
            response = self.generate_response(user_input)
            print(f"BitZero (legacy): {response}")


if __name__ == "__main__":
    print("--- Testing MemoryManager Emotional Intelligence Framework (Directly) ---")
    test_db_path = "test_ei_framework.db"
    if os.path.exists(test_db_path): os.remove(test_db_path)
    
    manager = MemoryManager(db_path=test_db_path)
    ei_module = manager.emotional_intelligence

    test_user_id = "emotion_tester"
    test_convo_id = manager.get_or_create_conversation_session(test_user_id)

    inputs_and_expected_emotions = [
        ("I am so happy today, it's wonderful!", {"joy"}),
        ("This is terrible, I feel really sad.", {"sadness"}),
        ("I'm furious about this!", {"anger"}),
        ("That's a bit scary.", {"fear"}),
        ("Wow, I'm amazed!", {"surprise"}),
        ("I trust your judgment on this.", {"trust"}),
        ("I'm really looking forward to the event.", {"anticipation"})
    ]

    for user_text, expected_dom_emotions in inputs_and_expected_emotions:
        print(f"\nInput: '{user_text}'")
        detected = ei_module.detect_user_emotion(user_text)
        print(f"Detected profile: {detected}")
        manager.extract_memories_from_conversation(user_input=user_text, model_response="", user_id=test_user_id, conversation_id=test_convo_id)
        empathetic_snippet = ei_module.generate_empathetic_response_snippet(detected)
        print(f"Empathetic snippet: {empathetic_snippet}")
        if detected:
            dominant_detected = max(detected.items(), key=lambda x: x[1])[0]
            assert dominant_detected in expected_dom_emotions, f"Expected one of {expected_dom_emotions}, got {dominant_detected}"
            print(f"Dominant emotion '{dominant_detected}' correctly identified (or one of expected).")
        elif expected_dom_emotions: 
             assert False, f"Expected one of {expected_dom_emotions}, but no emotion detected."

    print("\nVerifying emotion logs in DB for user 'emotion_tester':")
    emotion_logs = manager.retrieve_memories(category="user_emotion_log", user_id=test_user_id, limit=10)
    assert len(emotion_logs) >= len(inputs_and_expected_emotions) 
    print(f"Found {len(emotion_logs)} emotion logs for user '{test_user_id}'. Sample of first log metadata:")
    if emotion_logs:
        print(json.dumps(emotion_logs[0].get("metadata", {}), indent=2))

    # --- Test Memory Consolidation ---
    print("\n--- Testing Memory Consolidation ---")
    consolidation_user_id = "consolidation_tester"
    manager.store_memory("user_fact", "User likes apples.", 0.7, metadata={"user_id": consolidation_user_id})
    manager.store_memory("user_fact", "User enjoys eating apples.", 0.8, metadata={"user_id": consolidation_user_id})
    manager.store_memory("user_fact", "User mentioned they like apples a lot.", 0.9, metadata={"user_id": consolidation_user_id})
    manager.store_memory("user_fact", "User likes bananas.", 0.6, metadata={"user_id": consolidation_user_id})
    manager.store_memory("user_preference", "Prefers red apples.", 0.85, metadata={"user_id": consolidation_user_id})

    print("Before consolidation:")
    mem_before = manager.retrieve_memories(user_id=consolidation_user_id, limit=10)
    for m in mem_before: print(f"  ID: {m['id']}, Content: '{m['content']}', Importance: {m['importance']:.2f}")
    
    manager.consolidate_memories(user_id=consolidation_user_id, threshold=0.7) # Use a threshold that should group apple memories
    
    print("After consolidation:")
    mem_after = manager.retrieve_memories(user_id=consolidation_user_id, limit=10)
    for m in mem_after: print(f"  ID: {m['id']}, Content: '{m['content']}', Importance: {m['importance']:.2f}")
    # Basic check: number of memories should decrease if consolidation happened
    # assert len(mem_after) < len(mem_before), "Consolidation did not reduce memory count as expected for apples."
    # More specific checks can be added based on expected merged content/importance

    # --- Test Temporal Memory Weighting ---
    print("\n--- Testing Temporal Memory Weighting ---")
    temporal_user_id = "temporal_tester"
    manager.store_memory("user_info", "Recent event: attended a concert last week.", 0.9, metadata={"user_id": temporal_user_id},)
    # Manually update last_accessed for older memories to simulate age
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat()
    manager.store_memory("user_info", "Old fact: learned to code two years ago.", 0.8, metadata={"user_id": temporal_user_id})
    # Find the ID of the 'Old fact' to update its timestamp
    cursor.execute("SELECT id FROM memories WHERE content LIKE ? AND metadata LIKE ?", ("%Old fact%", f'%"user_id": "{temporal_user_id}"%'))
    old_fact_row = cursor.fetchone()
    if old_fact_row:
        cursor.execute("UPDATE memories SET last_accessed = ? WHERE id = ?", (one_year_ago, old_fact_row[0]))
    conn.commit()
    conn.close()

    context_with_temporal = manager.generate_context_from_memory("tell me about my activities", for_user_id=temporal_user_id, max_items=3)
    print("Context with temporal weighting:")
    print(context_with_temporal)
    assert "Recent event" in context_with_temporal
    # Depending on weighting, "Old fact" might or might not appear if many recent items exist.

    if os.path.exists(test_db_path): os.remove(test_db_path)
    print("\nAll MemoryManager tests finished.")