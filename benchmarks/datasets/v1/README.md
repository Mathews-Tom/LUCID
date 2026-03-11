# Benchmark Dataset v1 Schema

Datasets are stored as JSONL files where each line is a JSON object conforming to the `SampleRecord` schema.

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `sample_id` | string | yes | Unique identifier (e.g., `smp_a1b2c3d4`) |
| `split` | string | yes | One of: `train`, `val`, `test` |
| `domain` | string | yes | Content domain (e.g., `academic`, `news`, `blog`) |
| `source_class` | string | yes | One of: `human`, `ai_raw`, `ai_edited_light`, `ai_edited_heavy` |
| `source_model` | string or null | no | Model identifier if AI-generated (e.g., `gpt-4.1`) |
| `document_format` | string | yes | One of: `latex`, `markdown`, `plaintext` |
| `text` | string | yes | Full document text |
| `metadata` | object | no | Additional key-value metadata (default: `{}`) |

## Corpus Directory Format

Alternative to JSONL, datasets can be organized as a directory tree:

```
corpus_dir/
  domain_name/
    source_class/
      document_001.txt
      document_002.txt
```

Domain and source_class are inferred from directory names. See `schema.json` for the formal JSON Schema definition.
