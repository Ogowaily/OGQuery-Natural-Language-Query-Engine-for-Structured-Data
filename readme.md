# OGQuery — Natural Language Query Engine for Structured Data

**OGQuery** is a standalone query engine that lets you interact with structured tabular datasets using plain English. It is not a library, not a RAG pipeline, and not a document retrieval system.

It is a purpose-built engine that translates natural language directly into optimized, index-driven operations over your data — with no SQL, no document chunking, and no full-dataset scanning.

> *"Show failed missions after 2015 with cost over 5000 involving SpaceX"*
>
> OGQuery detects intent, routes each condition to the right index, intersects results, and returns a clean natural language answer — in milliseconds.

---
<img width="1536" height="1024" alt="ChatGPT Image May 8, 2026, 10_39_04 PM" src="https://github.com/user-attachments/assets/0abec7a1-39da-4729-8d65-eb53b0ae1e32" />

## What makes OGQuery different

Most "natural language + data" tools are built on RAG: they chunk documents, embed everything, retrieve passages, and feed them to an LLM. OGQuery takes a fundamentally different approach.

| | OGQuery | RAG-based tools |
|---|---|---|
| Data format | Structured tabular (CSV, etc.) | Unstructured documents |
| Embedding scope | Semantic columns only | Entire rows or documents |
| Retrieval method | Index-targeted (semantic + numeric + categorical) | Vector similarity over full corpus |
| Row embedding | Never | Always |
| Query execution | Execution plan over multiple indexes | Single vector lookup |
| Result type | Structured records + natural language answer | Retrieved passages |

**The core principle:** embed only what needs semantic understanding (specific text columns), index everything else with the right data structure, and route each query condition to the minimal required store. Raw rows are never embedded.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Embedding Strategy](#embedding-strategy)
- [Data Routing System](#data-routing-system)
- [Storage Architecture](#storage-architecture)
- [Query Processing Pipeline](#query-processing-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Example Output](#example-output)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Author](#author)

---

## How It Works

```
User asks:  "Show failed missions with cost > 5000 involving SpaceX"

  Step 1 — Parse       Detect intent, extract conditions and target columns
  Step 2 — Plan        Build execution plan: which stores, in what order
  Step 3 — Route       "failed"     → categorical index
                       "cost > 5000" → numeric index
                       "SpaceX"     → categorical index
  Step 4 — Execute     Run each targeted index lookup independently
  Step 5 — Intersect   Merge and rank matched record IDs
  Step 6 — Fetch       Reconstruct full rows from row store
  Step 7 — Answer      Generate natural language response via LLM
```

No full scans. No document retrieval. Raw rows are never embedded.

---

## Architecture

```
┌──────────────────────────────────────┐
│             User Query               │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        NLP / Intent Parser           │
│         (groq_parser.py)             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│           Query Planner              │
│      (execution strategy layer)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│           Query Engine               │
│       (orchestration layer)          │
└────────┬─────────────────┬───────────┘
         │                 │
         ▼                 ▼
┌────────────────┐  ┌──────────────────────────┐
│ Semantic Store │  │     Structured Stores     │
│  (FAISS index  │  │  numeric / categorical /  │
│  per column)   │  │  mapping / row store      │
└────────┬───────┘  └──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│          Context Builder             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Answer Generator             │
│      (Groq LLM formatting)           │
└──────────────────────────────────────┘
```

---

## Embedding Strategy

OGQuery embeds only what requires semantic understanding — specific text columns identified during schema detection. Everything else is indexed using the appropriate data structure.

| Scope | Embedded | Notes |
|---|---|---|
| Semantic text columns | Yes | e.g. `Objective`, `Key_Achievement` — column-level FAISS index |
| Numeric columns | No | Range tree / sorted index |
| Categorical columns | No | Hash map / inverted index |
| Raw rows | **Never** | Rows are stored in a row store and fetched by ID after index lookup |
| Documents / chunks | **Never** | OGQuery is not a document retrieval system |

This means ingestion is fast, indexes stay small, and retrieval is precise — without the noise of embedding irrelevant fields.

---

## Data Routing System
```
┌──────────────────────────────────────┐
│          Dataset Upload              │
│       engine.upload(csv_file)        │
│            (core.py)                 │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Ingestion Pipeline           │
│      (ingestion/pipeline.py)         │
│                                      │
│  Main orchestration layer            │
│  for dataset ingestion               │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│           Dataset Loader             │
│       (ingestion/loader.py)          │
│                                      │
│  - Read CSV                          │
│  - Create dataframe                  │
│  - Normalize columns                 │
│  - Handle null values                │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│          Schema Detector             │
│        (schema/detector.py)          │
│                                      │
│  Detect column types:                │
│  - semantic                          │
│  - numeric                           │
│  - categorical                       │
│  - identifiers                       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│          Schema Registry             │
│        (schema/registry.py)          │
│                                      │
│  Store dataset metadata              │
│  and column mappings                 │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│            Column Router             │
│       (ingestion/router.py)          │
│                                      │
│  Route each column to the            │
│  appropriate storage builder         │
└────────┬─────────────────┬───────────┘
         │                 │
         ▼                 ▼

┌────────────────┐  ┌──────────────────────────┐
│ Semantic Index │  │     Structured Indexes    │
│                │  │                           │
│ encoder.py     │  │ numeric_store.py         │
│ semantic_store │  │ mapping_store.py         │
│                │  │ row_store.py             │
│                │  │                           │
│ FAISS indexes  │  │ numeric / categorical /  │
│ per column     │  │ row storage              │
└────────┬───────┘  └──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Artifact Persistence         │
│      (ingestion/persist.py)          │
│                                      │
│ Save generated indexes               │
│ and metadata files                   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Generated Artifacts          │
│                                      │
│  registry.json                       │
│  meta.json                           │
│  *.faiss                             │
│  *.idx                               │
│  *.pkl                               │
│  rows.db                             │
└──────────────────────────────────────┘
```
Every query condition is routed to the most efficient storage layer before any data is touched.

### Storage by data type

| Data Type | Example Columns | Storage Layer | Purpose |
|---|---|---|---|
| Numeric | Cost, Duration, Dates | Range tree / sorted index | Fast range filtering and comparisons |
| Categorical | Agency, Country, Status | Hash map / inverted index | Exact match and grouping |
| Semantic text | Objective, Key_Achievement | FAISS index (per column) | Meaning-based search |
| Identifiers | Mission_ID | Primary key store | Direct row access |
| Raw rows | Full record | Row store (SQLite / file / memory) | Final reconstruction |

### Routing flow

```
Query
 → Intent Detection       (numeric / semantic / categorical / hybrid?)
 → Column Mapping         (which columns are relevant?)
 → Strategy Selection     (which storage layers to invoke?)
 → Execution              (targeted lookup — no full scans)
 → Intersection           (merge result IDs across stores)
 → Row Fetch              (reconstruct full records)
 → Answer Generation
```

### Multi-condition example

**Query:** `"Show failed missions with cost greater than 5000 involving SpaceX"`

| Condition | Routed to |
|---|---|
| `failed` | Categorical index → `Status = Failed` |
| `cost > 5000` | Numeric index → `Cost_USD_Million > 5000` |
| `SpaceX` | Categorical index → `Partner_Agencies = SpaceX` |

Results from each index are intersected by record ID, then full rows are fetched once from the row store.

---

## Storage Architecture

| Store | Purpose |
|---|---|
| Semantic store | Column-level FAISS vector index for meaning-based search |
| Numeric store | Range tree for fast numeric filtering and comparisons |
| Mapping store | Hash map / inverted index for categorical exact-match |
| Row store | Full row reconstruction after ID-based lookup |

---

## Query Processing Pipeline

### 1. Parse

Extracts from the raw query: intent type, filter conditions, target columns, sort preferences.

### 2. Plan

Builds an ordered execution plan — which stores to hit, in what order, with what parameters.

### 3. Retrieve

Executes each store lookup. No store is accessed unless the query requires it.

### 4. Assemble

Intersects result sets by record ID, scores and ranks matches, deduplicates.

### 5. Answer

Fetches full rows from the row store, passes structured context to the LLM answer generator, returns a human-readable response.

---

## Project Structure

```
ogquery/
│
├── ogquery/
│   ├── core/              # Core engine logic
│   ├── engine/
│   │   ├── query_engine   # Orchestration layer
│   │   ├── planner        # Execution strategy builder
│   │   ├── ingestion      # Data ingestion pipeline
│   │   ├── groq_parser    # NLP intent parser
│   │   └── answer_generator
│   ├── embeddings/        # Column-level vector encoding
│   ├── schema/            # Dataset registry and schema detection
│   ├── storage/
│   │   ├── semantic_store # FAISS column indexes
│   │   ├── numeric_store  # Range index
│   │   ├── mapping_store  # Categorical hash map
│   │   └── row_store      # Full row reconstruction
│   ├── utils/             # Logging and helpers
│   ├── api.py
│   ├── core.py
│   └── __init__.py
│
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Getting Started

### Requirements

- Python 3.8+
- A Groq API key

### Run the engine
#APi method test
```python
engine = OGQuery(config={
    "data_dir": "./data",
    "api_keys": {
        "groq": ""
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5
})

engine.serve(host="127.0.0.1", port=8000)
```
```bash
ogquery --data ./data --port 8000
```

### Or use it programmatically

```python
from ogquery import OGQuery

engine = OGQuery(config={
    "data_dir": "./data",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "api_keys": {
        "groq": "<YOUR_GROQ_API_KEY>"
    }
})

# Ingest a dataset
dataset_id = engine.upload("path/to/dataset.csv", name="space_missions")

# Query it
result = engine.query(dataset_id, "Show successful NASA missions after 2010")
print(result)
```

### Data ingestion pipeline

When a dataset is uploaded, OGQuery automatically:

1. Detects column types — numeric, categorical, semantic text, identifier
2. Generates vector embeddings for semantic text columns only
3. Builds numeric range indexes and categorical hash maps
4. Persists all artifacts to the configured data directory

**Artifacts produced:**
- `meta.json` — dataset metadata
- `registry.json` — schema and column type registry
- FAISS index files — one per semantic column
- Numeric index files
- Categorical mapping files

---

## API Reference

### `POST /query`

Run a natural language query against a registered dataset.

**Request:**
```json
{
  "dataset_id": "space_missions",
  "query": "failed missions after 2010 with cost over 5000"
}
```

**Response fields:**

| Field | Description |
|---|---|
| `query` | Original query string |
| `results` | Array of matched records |
| `summary.answer` | Human-readable answer |
| `summary.total_matches` | Total records matching all conditions |
| `summary.returned` | Number of records in this response |
| `execution_insight` | Filters applied, routing decisions, sort order |
| `elapsed_ms` | Total query execution time |

---

## Example Output

```json
{
  "query": "failed NASA missions",
  "results": [
    {
      "Mission_ID": "NA-00001",
      "Mission_Name": "Explorer 1",
      "Agency": "NASA",
      "Status": "Failed",
      "Launch_Date": "1976-01-12",
      "Cost_USD_Million": 7708.6,
      "_relevance_score": 0.7
    }
  ],
  "summary": {
    "answer": "Found 973 matches. Top result: NA-00001, Explorer 1, NASA.",
    "total_matches": 973,
    "returned": 3
  },
  "execution_insight": {
    "filters": [
      { "column": "Agency", "value": "NASA" },
      { "column": "Status", "value": "Failed" }
    ],
    "numeric_filters": [],
    "sort_by": null,
    "objective": "list"
  },
  "elapsed_ms": 1206.88
}
```

---

## Configuration

| Key | Type | Description |
|---|---|---|
| `data_dir` | `str` | Path to dataset storage directory |
| `api_keys.groq` | `str` | Groq API key for intent parsing and answer generation |
| `embedding_model` | `str` | HuggingFace model ID for semantic column encoding |
| `top_k` | `int` | Number of top results to return per query |

---

## Error Handling

| Scenario | Strategy |
|---|---|
| Dataset not found | Graceful fallback with descriptive error |
| Malformed query | User-friendly validation message |
| Corrupted index | Automatic rebuild |
| Embedding failure | Retry with backoff |

---

## Deployment

**Recommended stack:**

| Component | Technology |
|---|---|
| API server | FastAPI |
| Vector search | FAISS |
| Container | Docker |
| Row storage | SQLite or PostgreSQL |


## Author

**Omar Gowaily**

 
