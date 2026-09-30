# Knowledge tools and required sources

AnesTRACE Level Three exposes four knowledge tools. Tool schemas and execution
code are included, while restricted corpora are not.

## Drug information

\`get_anesthesia_drug_information(drug_name)\` queries:

- RxNorm, for drug identity normalization;
- DailyMed, for current Structured Product Label metadata;
- openFDA drug labels, for a bounded label fallback.

These are public US government services. Network access is required on the
first request; responses are cached under \`data/tool_cache/\`. Users remain
responsible for the services' terms, attribution requirements, and determining
whether a product-specific label applies to the benchmark case.

## PubMed

\`search_pubmed(query)\` uses NCBI E-utilities ESearch followed by EFetch.
Set the following optional environment variables:

\`\`\`text
NCBI_EMAIL=researcher@example.org
NCBI_TOOL=AnesTRACE
NCBI_API_KEY=optional_ncbi_key
\`\`\`

The API key value is never included in cache identities, metadata, or run
manifests. PubMed evidence is literature, not a clinical guideline.

## Guidelines

\`search_guidelines(query)\` reads a user-supplied JSONL corpus configured as
\`tools.guideline_corpus_path\`. AnesTRACE does not redistribute guideline
documents. Obtain them from official publishers under their applicable terms,
retain the official URL/version/page provenance, and have extracted passages
reviewed before benchmark use.

Each JSONL row must include \`text\` or \`content\`. Recommended metadata:

\`\`\`json
{
  "chunk_id": "source:section:001",
  "source_id": "source",
  "source_title": "Official title",
  "section": "Recommendation heading",
  "page_start": 12,
  "page_end": 12,
  "citation": "[source | version | p.12]",
  "official_url": "https://official.example/document",
  "clinical_review_status": "reviewed",
  "text": "A complete recommendation or semantic unit."
}
\`\`\`

Use semantic recommendation boundaries rather than arbitrary fixed-length
chunks. Do not mix versions without recording version metadata.

## Miller's Anesthesia

\`search_miller_anesthesia(query)\` uses the same local JSONL/BM25 interface,
configured by \`tools.miller_corpus_path\`. Miller's Anesthesia is copyrighted
and is not included. Users must provide their own lawfully licensed copy and
derived index, and must not publish the source PDF or extracted corpus unless
their license expressly permits it.

For reproducibility, record edition, chapter, printed page, PDF page, index
builder version, and a hash of the locally licensed source. The run output
stores retrieval traces but those traces should also be reviewed before public
release because they may contain copyrighted excerpts.

## Offline behavior

Set \`tools.network_enabled=false\` to prevent new drug/PubMed requests. Cached
public API responses can still be used. Missing guideline or textbook corpora
produce an explicit \`unavailable\` tool result rather than silently falling
back to fabricated evidence.
