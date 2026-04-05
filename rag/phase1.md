This is the "Extraction ETL" (Extract, Transform, Load) pipeline. We will use **Gemini Pro** to read your text chunk-by-chunk and convert it into structured JSON, then push that JSON into **Neo4j**.

### Prerequisites
You need to install the Neo4j driver and the Google AI SDK:

```bash
pip install neo4j google-generativeai
```

### The Python Pipeline
Create a file named `build_dsa_graph.py`.

```python
import os
import json
import google.generativeai as genai
from neo4j import GraphDatabase

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Replace with your actual keys
NEO4J_URI = "neo4j+s://8b949258.databases.neo4j.io" # Or "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "d0NbdbmArKSLJNcCF5rnBjT-cLpYb0NStSgheyZHzdU"
GOOGLE_API_KEY = "AIzaSyCaC8uBkcLkCYakBxY2BTj3XEltOEoFFhU"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. THE GRAPH BUILDER CLASS
# ==========================================
class DSAGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def extract_graph_from_text(self, text_chunk):
        """
        Uses Gemini to parse text into Nodes and Relationships.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # This prompt is the 'Intelligence' of the operation.
        # It forces Gemini to think like a Curriculum Designer.
        prompt = f"""
        You are an expert Data Structures & Algorithms Curriculum Designer.
        Analyze the following educational text and extract a Knowledge Graph.
        
        TEXT TO ANALYZE:
        "{text_chunk}"
        
        You must identify three things:
        1. CONCEPTS: Key technical terms (e.g., "Linked List", "Pointer", "Recursion").
        2. RELATIONSHIPS: 
           - "REQUIRES": If Concept A relies on understanding Concept B.
           - "SUBTYPE_OF": If Concept A is a type of Concept B.
        3. MISCONCEPTIONS: Common mistakes or confusing points mentioned in the text.

        Return ONLY a JSON object with this exact structure (no markdown formatting):
        {{
            "concepts": [
                {{"id": "unique_id_a", "name": "Concept Name", "definition": "Short definition"}}
            ],
            "relationships": [
                {{"source": "unique_id_a", "target": "unique_id_b", "type": "REQUIRES"}},
                {{"source": "unique_id_a", "target": "unique_id_b", "type": "SUBTYPE_OF"}}
            ],
            "misconceptions": [
                {{"concept_id": "unique_id_a", "description": "Description of the error"}}
            ]
        }}
        """
        
        try:
            response = model.generate_content(prompt)
            # Clean up potential markdown formatting (```json ... ```)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Error extracting with Gemini: {e}")
            return None

    def push_to_neo4j(self, graph_data):
        """
        Ingests the structured JSON into Neo4j using Idempotent MERGE queries.
        """
        if not graph_data:
            return

        with self.driver.session() as session:
            # 1. Create Concepts (Nodes)
            session.run("""
                UNWIND $concepts AS c
                MERGE (n:Concept {id: c.id})
                SET n.name = c.name, 
                    n.definition = c.definition,
                    n.category = "DSA"
            """, concepts=graph_data['concepts'])

            # 2. Create Relationships (Edges)
            # We handle REQUIRES and SUBTYPE_OF specifically
            session.run("""
                UNWIND $rels AS r
                MATCH (source:Concept {id: r.source})
                MATCH (target:Concept {id: r.target})
                CALL apoc.do.when(
                    r.type = 'REQUIRES',
                    'MERGE (source)-[:REQUIRES]->(target)',
                    'MERGE (source)-[:SUBTYPE_OF]->(target)',
                    {source:source, target:target}
                ) YIELD value
                RETURN value
            """, rels=graph_data['relationships'])
            
            # Note: If you don't have APOC installed in Neo4j, use standard cypher logic below instead:
            # (See the alternative pure Cypher implementation at the bottom of this script)

            # 3. Create Misconceptions (Nodes + Edges)
            session.run("""
                UNWIND $errors AS e
                MATCH (c:Concept {id: e.concept_id})
                MERGE (m:Misconception {description: e.description})
                MERGE (c)-[:HAS_MISCONCEPTION]->(m)
            """, errors=graph_data['misconceptions'])
            
            print(f"✅ Successfully ingested {len(graph_data['concepts'])} concepts and relationships.")

# ==========================================
# 3. EXECUTION
# ==========================================

# SAMPLE CONTENT (In reality, you would read this from your PDF/Text file)
dsa_text_chunk = """
To understand a Linked List, one must first have a solid grasp of Pointers. 
A Linked List is a linear data structure, unlike Arrays which are contiguous. 
A common mistake students make with Linked Lists is losing the reference to the head node, 
which causes the entire list to be lost in memory. 
Doubly Linked Lists are a specific type of Linked List that have pointers to both next and previous nodes.
"""

# Run the pipeline
if __name__ == "__main__":
    builder = DSAGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    print("🤖 Asking Gemini to analyze text...")
    data = builder.extract_graph_from_text(dsa_text_chunk)
    
    if data:
        print("Detailed Extraction:", json.dumps(data, indent=2))
        print("database connecting...")
        # Note: Ensure you replace the APOC query above with standard Cypher if you don't have APOC plugins
        # Here is the standard Cypher version for relationships to be safe:
        with builder.driver.session() as session:
             session.run("""
                UNWIND $rels AS r
                MATCH (source:Concept {id: r.source})
                MATCH (target:Concept {id: r.target})
                FOREACH (_ IN CASE WHEN r.type = 'REQUIRES' THEN [1] ELSE [] END |
                    MERGE (source)-[:REQUIRES]->(target)
                )
                FOREACH (_ IN CASE WHEN r.type = 'SUBTYPE_OF' THEN [1] ELSE [] END |
                    MERGE (source)-[:SUBTYPE_OF]->(target)
                )
            """, rels=data['relationships'])
             
             # Re-run concepts and misconceptions ingestion from the class method
             # (For this snippet, I am manually running the missing parts for simplicity)
             session.run("""
                UNWIND $concepts AS c
                MERGE (n:Concept {id: c.id})
                SET n.name = c.name, n.definition = c.definition
            """, concepts=data['concepts'])
             
             session.run("""
                UNWIND $errors AS e
                MATCH (c:Concept {id: e.concept_id})
                MERGE (m:Misconception {description: e.description})
                MERGE (c)-[:HAS_MISCONCEPTION]->(m)
            """, errors=data['misconceptions'])

        print("🚀 Graph Built in Neo4j!")
        
    builder.close()
```

### How to use this for the Full Book

You cannot paste a whole book into the `dsa_text_chunk` variable. You need a loop.

1.  **Split your book:** Save your book content as `dsa_content.txt`.
2.  **Chunking Logic:**
    LLMs have context limits. You should split the text by paragraphs or roughly every 2000 characters.
    ```python
    def chunk_text(text, chunk_size=2000):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # In the main block:
    with open("dsa_content.txt", "r") as f:
        full_text = f.read()
        
    chunks = chunk_text(full_text)
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}...")
        data = builder.extract_graph_from_text(chunk)
        builder.push_to_neo4j(data)
    ```

### Why this structure wins for your Report

1.  **"Misconception" Nodes:**
    Notice the code specifically extracts misconceptions. This aligns perfectly with your Socratic goal.
    *   *Report Highlight:* "Unlike standard KG construction which captures only facts, our pipeline extracts *pedagogical metadata* (common errors), allowing the Tutor to preemptively warn students."

2.  **"Requires" Relationships:**
    The prompt forces Gemini to find dependencies.
    *   *Report Highlight:* "We automated the creation of a 'Dependency Graph' (Prerequisites), allowing the system to detect when a student is attempting a topic they are not ready for."

3.  **Scalability:**
    Using Python + Gemini API means you can process an entire textbook in about 5 minutes, creating a massive graph that would take weeks to build manually.