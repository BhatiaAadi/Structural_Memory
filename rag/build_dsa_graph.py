import os
import json
import time
import re
from google import genai
from google.genai import types
from neo4j import GraphDatabase
import fitz  # PyMuPDF

# ==========================================
# 1. CONFIGURATION
# ==========================================
NEO4J_URI      = "neo4j+s://8b949258.databases.neo4j.io"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "d0NbdbmArKSLJNcCF5rnBjT-cLpYb0NStSgheyZHzdU"
GOOGLE_API_KEY = "AIzaSyCaC8uBkcLkCYakBxY2BTj3XEltOEoFFhU"

PDF_PATH        = "Cormen Introduction to Algorithms.pdf"
PROGRESS_FILE   = "progress.json"        # tracks last completed chunk_no
MAX_CHUNK_CHARS = 4000                   # max chars per semantic chunk (~1K tokens)
BATCH_SIZE      = 5                      # 5 chunks × 4000 chars = ~20K chars per API call
DELAY_SECONDS   = 1                      # 1s is safe at paid-tier limits
MAX_RETRIES     = 6                      # retries on 429 / DNS errors

# With billing active, paid-tier limits apply (no daily cap):
#   gemini-2.5-pro       : best quality KG, ~$2-3 for full CLRS book
#   gemini-2.0-flash     : good quality, ~$0.20
MODEL_NAME = "gemini-2.5-pro"

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(api_version="v1beta"),
)

# ==========================================
# 2. SEMANTIC CHUNKING FROM PDF
# ==========================================
# CLRS headings look like:
#   "1  The Role of Algorithms in Computing"
#   "1.1  Algorithms"
#   "Chapter 2  Getting Started"
HEADING_RE = re.compile(
    r"^(?:Chapter\s+)?\d+(?:\.\d+)*\s{2,}.+$",
    re.MULTILINE,
)


def extract_semantic_chunks(pdf_path: str) -> list[dict]:
    """
    Extract text from the PDF and split it into semantic chunks aligned to
    chapter/section boundaries detected by heading patterns.

    Each chunk is a dict:
        { "title": str, "text": str, "chunk_no": int }

    Sections larger than MAX_CHUNK_CHARS are further split on paragraph
    boundaries so no single API call is oversized.
    """
    print(f"📖 Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    # Collect all page text block-by-block (preserves reading order better)
    pages_text: list[str] = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        for block in blocks:
            if block["type"] != 0:      # skip image blocks
                continue
            for line in block["lines"]:
                line_text = " ".join(span["text"] for span in line["spans"]).strip()
                if line_text:
                    page_lines.append(line_text)
        pages_text.append("\n".join(page_lines))
    doc.close()

    full_text = "\n\n".join(pages_text)
    print(f"   Extracted {len(full_text):,} characters across {page_count} pages.")

    # Split on detected headings
    heading_matches = list(HEADING_RE.finditer(full_text))
    sections: list[tuple[str, str]] = []  # (title, body)

    if not heading_matches:
        print("   ⚠️  No headings detected. Falling back to full-text paragraph chunking.")
        sections = [("Full Text", full_text)]
    else:
        pre = full_text[: heading_matches[0].start()].strip()
        if pre:
            sections.append(("Front Matter", pre))
        for i, match in enumerate(heading_matches):
            title = match.group().strip()
            start = match.end()
            end   = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(full_text)
            body  = full_text[start:end].strip()
            if body:
                sections.append((title, body))

    print(f"   Found {len(sections)} semantic sections.")

    # Enforce MAX_CHUNK_CHARS by splitting large sections on paragraph boundaries
    chunks: list[dict] = []
    chunk_no = 1
    for title, body in sections:
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append({"title": title, "text": body, "chunk_no": chunk_no})
            chunk_no += 1
        else:
            paragraphs = body.split("\n\n")
            current, part = "", 1
            for para in paragraphs:
                if len(current) + len(para) > MAX_CHUNK_CHARS and current:
                    chunks.append({
                        "title":    f"{title} (part {part})",
                        "text":     current.strip(),
                        "chunk_no": chunk_no,
                    })
                    chunk_no += 1
                    part += 1
                    current = para
                else:
                    current += "\n\n" + para
            if current.strip():
                chunks.append({
                    "title":    f"{title} (part {part})",
                    "text":     current.strip(),
                    "chunk_no": chunk_no,
                })
                chunk_no += 1

    print(f"   Total chunks after size enforcement: {len(chunks)}\n")
    return chunks


# ==========================================
# 3. GRAPH BUILDER CLASS
# ==========================================
class DSAGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ------------------------------------------
    # 3a. LLM Extraction  (batch + retry + backoff)
    # ------------------------------------------
    def extract_graph_from_batch(self, batch: list[dict]) -> dict | None:
        """
        Merges multiple semantic chunks into ONE API call to maximise
        throughput within the free-tier rate limits.
        Returns a single merged graph_data dict.
        """
        # Build a combined text with section labels so context is clear
        combined_text = "\n\n---\n\n".join(
            f"[Section: {c['title']}]\n{c['text']}" for c in batch
        )

        prompt = f"""
You are an expert Data Structures & Algorithms Curriculum Designer.
Analyze the following educational text from the CLRS textbook
"Introduction to Algorithms". It contains {len(batch)} sections.

TEXT TO ANALYZE:
\"\"\"
{combined_text}
\"\"\"

Identify three things across ALL sections:
1. CONCEPTS: Key technical terms
   (e.g., "Binary Search Tree", "Amortized Analysis", "Dynamic Programming").
   Assign each a SHORT snake_case unique id (e.g., "binary_search_tree").

2. RELATIONSHIPS between concepts:
   - "REQUIRES"   : Concept A requires understanding Concept B as a prerequisite.
   - "SUBTYPE_OF" : Concept A is a specialised type / variant of Concept B.
   - "USES"       : Concept A employs Concept B as a technique or data structure.

3. MISCONCEPTIONS: Common mistakes or confusing points explicitly mentioned
   or strongly implied in the text.

Rules:
- Only include items clearly supported by the provided text.
- Every concept id used in relationships MUST also appear in the concepts list.
- Return ONLY a raw JSON object — no markdown fences, no extra text.

Required JSON structure:
{{
    "concepts": [
        {{"id": "snake_case_id", "name": "Human Readable Name", "definition": "One-sentence definition"}}
    ],
    "relationships": [
        {{"source": "id_a", "target": "id_b", "type": "REQUIRES"}},
        {{"source": "id_a", "target": "id_b", "type": "SUBTYPE_OF"}},
        {{"source": "id_a", "target": "id_b", "type": "USES"}}
    ],
    "misconceptions": [
        {{"concept_id": "snake_case_id", "description": "Description of the common error"}}
    ]
}}
"""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
                raw = response.text.strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$",        "", raw)
                return json.loads(raw.strip())

            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error: {e}")
                return None

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_dns_error  = ("nodename nor servname" in err_str
                                 or "Name or service not known" in err_str
                                 or "Errno 8" in err_str
                                 or ("DNS" in err_str and "resolution" in err_str))

                if is_rate_limit or is_dns_error:
                    retry_match = re.search(r"Please retry in ([\d.]+)s", err_str)
                    wait = min(float(retry_match.group(1)) if retry_match else (2 ** attempt) * 5, 70)
                    reason = "DNS error" if is_dns_error else "Rate limited"
                    print(f"  ⏳ {reason} (attempt {attempt}/{MAX_RETRIES}). Waiting {wait:.0f}s…")
                    time.sleep(wait)
                else:
                    print(f"  ⚠️  Gemini error: {e}")
                    return None

        print(f"  ❌ Gave up after {MAX_RETRIES} retries.")
        return None

    # ------------------------------------------
    # 3b. Neo4j Ingestion
    # ------------------------------------------
    def push_to_neo4j(self, graph_data: dict, batch_label: str) -> None:
        """
        Upserts concepts, relationships, and misconceptions into Neo4j.
        Uses idempotent MERGE — safe to re-run at any time.
        """
        if not graph_data:
            return

        concepts       = graph_data.get("concepts", [])
        relationships  = graph_data.get("relationships", [])
        misconceptions = graph_data.get("misconceptions", [])

        concept_ids = {c["id"] for c in concepts}
        valid_rels  = [
            r for r in relationships
            if r["source"] in concept_ids and r["target"] in concept_ids
        ]

        with self.driver.session() as session:
            if concepts:
                session.run(
                    """
                    UNWIND $concepts AS c
                    MERGE (n:Concept {id: c.id})
                    SET n.name       = c.name,
                        n.definition = c.definition,
                        n.category   = 'DSA',
                        n.section    = $section
                    """,
                    concepts=concepts,
                    section=batch_label,
                )

            if valid_rels:
                session.run(
                    """
                    UNWIND $rels AS r
                    MATCH (src:Concept {id: r.source})
                    MATCH (tgt:Concept {id: r.target})
                    FOREACH (_ IN CASE WHEN r.type = 'REQUIRES'   THEN [1] ELSE [] END |
                        MERGE (src)-[:REQUIRES]->(tgt)
                    )
                    FOREACH (_ IN CASE WHEN r.type = 'SUBTYPE_OF' THEN [1] ELSE [] END |
                        MERGE (src)-[:SUBTYPE_OF]->(tgt)
                    )
                    FOREACH (_ IN CASE WHEN r.type = 'USES'       THEN [1] ELSE [] END |
                        MERGE (src)-[:USES]->(tgt)
                    )
                    """,
                    rels=valid_rels,
                )

            if misconceptions:
                session.run(
                    """
                    UNWIND $errors AS e
                    MATCH (c:Concept {id: e.concept_id})
                    MERGE (m:Misconception {description: e.description})
                    MERGE (c)-[:HAS_MISCONCEPTION]->(m)
                    """,
                    errors=misconceptions,
                )

        print(
            f"  ✅ {len(concepts)} concepts | "
            f"{len(valid_rels)} relationships | "
            f"{len(misconceptions)} misconceptions"
        )

    # ------------------------------------------
    # 3c. Process one batch end-to-end
    # ------------------------------------------
    def process_batch(self, batch: list[dict]) -> bool:
        """Returns True if the batch was processed successfully."""
        start_no = batch[0]["chunk_no"]
        end_no   = batch[-1]["chunk_no"]
        titles   = " | ".join(c["title"][:30] for c in batch)
        total_chars = sum(len(c["text"]) for c in batch)
        print(f"\n� Batch chunks {start_no}–{end_no} ({total_chars:,} chars)")
        print(f"   Sections: {titles}")

        data = self.extract_graph_from_batch(batch)
        if data:
            label = batch[0]["title"]   # use first section as label
            self.push_to_neo4j(data, label)
            return True
        else:
            print(f"  ⚠️  Skipping batch {start_no}–{end_no}.")
            return False


# ==========================================
# 4. PROGRESS TRACKING
# ==========================================
def load_progress() -> int:
    """Returns the last successfully processed batch's ending chunk_no (0 if fresh start)."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f).get("last_chunk_no", 0)
    return 0

def save_progress(last_chunk_no: int) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_chunk_no": last_chunk_no}, f)


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    print(f"🤖 Model      : {MODEL_NAME}")
    print(f"📄 PDF        : {PDF_PATH}")
    print(f"📦 Batch size : {BATCH_SIZE} chunks per API call")
    print(f"⏱  Delay      : {DELAY_SECONDS}s between calls\n")

    # 5a. Semantic chunking
    chunks = extract_semantic_chunks(PDF_PATH)

    # 5b. Resume support — skip already-processed chunks
    last_done = load_progress()
    remaining = [c for c in chunks if c["chunk_no"] > last_done]

    # Drop chunks with almost no text (table-of-contents entries, page numbers, etc.)
    MIN_CHUNK_CHARS = 100
    before = len(remaining)
    remaining = [c for c in remaining if len(c["text"]) >= MIN_CHUNK_CHARS]
    skipped_empty = before - len(remaining)
    if last_done:
        print(f"▶️  Resuming from chunk {last_done + 1} "
              f"({len(chunks) - before} already done)\n")
    if skipped_empty:
        print(f"🗑  Skipped {skipped_empty} near-empty chunks (< {MIN_CHUNK_CHARS} chars)\n")

    # 5c. Build batches of BATCH_SIZE
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    total_api_calls = len(batches)
    # At 30 req/min with DELAY_SECONDS gap: estimated minutes
    est_minutes = (total_api_calls * DELAY_SECONDS) / 60
    print(f"📊 {len(remaining)} chunks → {total_api_calls} API calls "
          f"(~{est_minutes:.0f} min at {DELAY_SECONDS}s/call)\n")

    builder = DSAGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        for i, batch in enumerate(batches, start=1):
            print(f"[{i}/{total_api_calls}]", end=" ")
            success = builder.process_batch(batch)
            if success:
                save_progress(batch[-1]["chunk_no"])
            time.sleep(DELAY_SECONDS)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted. Progress saved — re-run to continue.")
    finally:
        builder.close()
        print("\n🚀 Done! Knowledge Graph is live in Neo4j.")
