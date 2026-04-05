"""
rag_engine.py
─────────────
Core RAG engine:
  - Pulls relevant context from Neo4j KG given a topic/question
  - Calls Ollama (local LLaMA) with and without that context
  - Returns both answers so they can be compared
"""

import re
import json
import requests
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j+s://8b949258.databases.neo4j.io"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "d0NbdbmArKSLJNcCF5rnBjT-cLpYb0NStSgheyZHzdU"

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen2.5:7b"   # Qwen 2.5 7B

# Proficiency levels
LEVEL_BEGINNER     = "beginner"      # score 0-3
LEVEL_INTERMEDIATE = "intermediate"  # score 4-6
LEVEL_ADVANCED     = "advanced"      # score 7-10


# ── Neo4j helpers ─────────────────────────────────────────────────────────────
class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_concept_context(self, topic: str, depth: int = 2) -> dict:
        """
        Given a topic string, find the closest matching Concept node and return:
          - its definition
          - its prerequisites  (REQUIRES edges going in)
          - what it leads to   (REQUIRES edges going out)
          - subtypes / supertypes
          - misconceptions
        """
        with self.driver.session() as s:
            # Fuzzy match on name
            result = s.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($topic)
                   OR toLower(c.id)   CONTAINS toLower($topic)
                RETURN c.id AS id, c.name AS name, c.definition AS definition,
                       c.section AS section
                LIMIT 5
            """, topic=topic).data()

            if not result:
                return {}

            # Use best match (first result)
            concept = result[0]
            cid = concept["id"]

            # Prerequisites (what you need to know BEFORE this)
            prereqs = s.run("""
                MATCH (c:Concept {id: $id})-[:REQUIRES]->(pre:Concept)
                RETURN pre.name AS name, pre.definition AS definition
                LIMIT 10
            """, id=cid).data()

            # What this concept unlocks (what comes AFTER)
            unlocks = s.run("""
                MATCH (next:Concept)-[:REQUIRES]->(c:Concept {id: $id})
                RETURN next.name AS name, next.definition AS definition
                LIMIT 10
            """, id=cid).data()

            # Subtypes
            subtypes = s.run("""
                MATCH (sub:Concept)-[:SUBTYPE_OF]->(c:Concept {id: $id})
                RETURN sub.name AS name, sub.definition AS definition
                LIMIT 10
            """, id=cid).data()

            # Uses relationships
            uses = s.run("""
                MATCH (c:Concept {id: $id})-[:USES]->(u:Concept)
                RETURN u.name AS name, u.definition AS definition
                LIMIT 10
            """, id=cid).data()

            # Misconceptions
            misconceptions = s.run("""
                MATCH (c:Concept {id: $id})-[:HAS_MISCONCEPTION]->(m:Misconception)
                RETURN m.description AS description
                LIMIT 5
            """, id=cid).data()

            return {
                "concept":        concept,
                "prerequisites":  prereqs,
                "unlocks":        unlocks,
                "subtypes":       subtypes,
                "uses":           uses,
                "misconceptions": misconceptions,
                "all_matches":    result,
            }

    def get_prerequisites_chain(self, topic: str) -> list[dict]:
        """Returns full prerequisite chain (recursive, up to 3 hops)."""
        with self.driver.session() as s:
            return s.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($topic)
                MATCH path = (c)-[:REQUIRES*1..3]->(pre:Concept)
                RETURN [node in nodes(path) | node.name] AS chain
                LIMIT 10
            """, topic=topic).data()


# ── Context builder ───────────────────────────────────────────────────────────
def build_rag_context(kg_data: dict, user_profile: dict) -> str:
    """
    Converts raw KG data + user proficiency profile into a focused
    context string to inject into the LLM prompt.
    """
    if not kg_data:
        return ""

    concept     = kg_data.get("concept", {})
    prereqs     = kg_data.get("prerequisites", [])
    unlocks     = kg_data.get("unlocks", [])
    subtypes    = kg_data.get("subtypes", [])
    uses        = kg_data.get("uses", [])
    misconceptions = kg_data.get("misconceptions", [])

    known_topics = user_profile.get("known_topics", [])
    level        = user_profile.get("level", LEVEL_BEGINNER)

    lines = [
        f"=== Knowledge Graph Context ===",
        f"Topic: {concept.get('name', '')}",
        f"Definition: {concept.get('definition', '')}",
        f"Section in CLRS: {concept.get('section', '')}",
    ]

    if prereqs:
        lines.append("\nPrerequisites (must know before this):")
        for p in prereqs:
            known = "✓ (user knows this)" if any(
                p["name"].lower() in k.lower() or k.lower() in p["name"].lower()
                for k in known_topics
            ) else "✗ (user may NOT know this)"
            lines.append(f"  - {p['name']}: {p['definition']}  {known}")

    if unlocks:
        lines.append("\nThis topic unlocks (what comes next):")
        for u in unlocks:
            lines.append(f"  - {u['name']}: {u['definition']}")

    if subtypes:
        lines.append("\nVariants / Subtypes:")
        for st in subtypes:
            lines.append(f"  - {st['name']}: {st['definition']}")

    if uses:
        lines.append("\nThis concept uses:")
        for u in uses:
            lines.append(f"  - {u['name']}")

    if misconceptions:
        lines.append("\n⚠ Common Misconceptions to address:")
        for m in misconceptions:
            lines.append(f"  - {m['description']}")

    lines.append(f"\nUser proficiency level: {level}")
    if known_topics:
        lines.append(f"User already knows: {', '.join(known_topics)}")

    return "\n".join(lines)


# ── Ollama caller ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str, system: str = "") -> str:
    """Calls local Ollama and returns the response text."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"[ERROR] {e}"


# ── Main RAG interface ────────────────────────────────────────────────────────
def answer_with_rag(question: str, topic: str, user_profile: dict, kg: KnowledgeGraph) -> dict:
    """
    Returns a dict with:
      - without_rag : plain LLM answer
      - with_rag    : KG-grounded LLM answer
      - kg_context  : the raw context injected
    """
    # 1. Plain LLM answer (no KG)
    plain_system = (
        "You are a helpful DSA tutor. Answer the student's question clearly and concisely."
    )
    without_rag = call_ollama(question, system=plain_system)

    # 2. RAG answer (KG context injected)
    kg_data    = kg.get_concept_context(topic)
    rag_context = build_rag_context(kg_data, user_profile)

    level = user_profile.get("level", LEVEL_BEGINNER)
    level_instructions = {
        LEVEL_BEGINNER:     "Use simple language, avoid jargon, explain from scratch.",
        LEVEL_INTERMEDIATE: "Assume basic knowledge. Focus on depth and connections.",
        LEVEL_ADVANCED:     "Be concise and technical. Focus on nuances and edge cases.",
    }

    rag_system = f"""You are an adaptive DSA tutor powered by a Knowledge Graph from the CLRS textbook.

{rag_context}

Instruction: {level_instructions.get(level, '')}
- If the user is missing prerequisites, WARN them and briefly explain those first.
- Highlight common misconceptions relevant to this topic.
- Tailor your explanation to their proficiency level.
"""
    with_rag = call_ollama(question, system=rag_system)

    return {
        "question":    question,
        "topic":       topic,
        "user":        user_profile.get("name", "User"),
        "level":       level,
        "without_rag": without_rag,
        "with_rag":    with_rag,
        "kg_context":  rag_context,
    }
