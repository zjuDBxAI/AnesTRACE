"""Portable knowledge backends for the public Level Three agent runner.

Public network tools use US government APIs. Guideline and textbook retrieval
operate only on user-supplied JSONL corpora; no copyrighted source text is
distributed with AnesTRACE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{1,}|[0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cache_paths(url: str, params: dict[str, Any], cache_dir: Path) -> tuple[Path, Path]:
    identity = json.dumps({"url": url, "params": params}, sort_keys=True)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.response", cache_dir / f"{key}.metadata.json"


def _cached_get(
    url: str,
    params: dict[str, Any],
    *,
    cache_dir: Path,
    offline: bool,
    timeout: float = 30.0,
    retries: int = 2,
    secret_params: dict[str, str] | None = None,
) -> bytes:
    response_path, metadata_path = _cache_paths(url, params, cache_dir)
    if response_path.exists():
        return response_path.read_bytes()
    if offline:
        raise FileNotFoundError(f"offline cache miss: {response_path.name}")
    request_params = {**params, **(secret_params or {})}
    request_url = f"{url}?{urllib.parse.urlencode(request_params, doseq=True)}"
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/json, application/xml",
            "User-Agent": "AnesTRACE-Level3/1.0 (research benchmark)",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                headers = dict(response.headers.items())
            cache_dir.mkdir(parents=True, exist_ok=True)
            response_path.write_bytes(body)
            metadata_path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "params": params,
                        "retrieved_at": _utc_now(),
                        "headers": headers,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return body
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500 and exc.code != 429:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"request failed after retries: {last_error}")


def _json_get(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return json.loads(_cached_get(*args, **kwargs))


def get_anesthesia_drug_information(
    drug_name: str,
    *,
    cache_dir: str | Path,
    offline: bool = False,
) -> dict[str, Any]:
    """Retrieve bounded identity and label metadata from public drug APIs."""

    name = str(drug_name).strip()
    if not name:
        raise ValueError("drug_name must not be empty")
    cache = Path(cache_dir)
    errors: dict[str, str] = {}
    normalized_name = name
    rxcui: str | None = None
    rxnorm: dict[str, Any] = {}
    try:
        payload = _json_get(
            RXNORM_URL,
            {"name": name, "search": 2},
            cache_dir=cache / "rxnorm",
            offline=offline,
        )
        identifiers = payload.get("idGroup", {}).get("rxnormId") or []
        rxcui = str(identifiers[0]) if identifiers else None
        rxnorm = {"best_rxcui": rxcui, "candidate_rxcuis": identifiers[:5]}
    except Exception as exc:
        errors["rxnorm"] = str(exc)

    dailymed: dict[str, Any] = {}
    try:
        payload = _json_get(
            DAILYMED_URL,
            {"drug_name": normalized_name, "name_type": "both", "pagesize": 1, "page": 1},
            cache_dir=cache / "dailymed",
            offline=offline,
        )
        rows = payload.get("data") or []
        dailymed = {
            "result_count": len(rows),
            "labels": [
                {
                    "setid": row.get("setid"),
                    "title": row.get("title"),
                    "published_date": row.get("published_date"),
                    "label_url": (
                        "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
                        + str(row.get("setid"))
                    ),
                }
                for row in rows[:1]
            ],
        }
    except Exception as exc:
        errors["dailymed"] = str(exc)

    openfda: dict[str, Any] = {}
    try:
        payload = _json_get(
            OPENFDA_URL,
            {"search": f'openfda.generic_name:"{normalized_name}"', "limit": 1},
            cache_dir=cache / "openfda",
            offline=offline,
        )
        records = payload.get("results") or []
        selected = []
        for row in records[:1]:
            selected.append(
                {
                    "generic_name": (row.get("openfda") or {}).get("generic_name", []),
                    "brand_name": (row.get("openfda") or {}).get("brand_name", []),
                    "boxed_warning": (row.get("boxed_warning") or [])[:1],
                    "contraindications": (row.get("contraindications") or [])[:1],
                    "warnings": (row.get("warnings") or [])[:1],
                    "dosage_and_administration": (
                        row.get("dosage_and_administration") or []
                    )[:1],
                }
            )
        openfda = {"result_count": len(records), "labels": selected}
    except Exception as exc:
        errors["openfda"] = str(exc)

    return {
        "query": name,
        "status": "ok" if any((rxnorm, dailymed, openfda)) else "unavailable",
        "rxnorm": rxnorm,
        "dailymed": dailymed,
        "openfda": openfda,
        "errors": errors,
        "sources": [
            "https://rxnav.nlm.nih.gov/",
            "https://dailymed.nlm.nih.gov/",
            "https://open.fda.gov/apis/drug/label/",
        ],
        "safety_notice": (
            "Drug labels are product-specific evidence, not a patient-specific "
            "prescribing recommendation."
        ),
    }


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(str(text))]


def search_local_knowledge(
    query: str,
    *,
    corpus_path: str | Path | None,
    top_k: int,
    source_kind: str,
) -> dict[str, Any]:
    """BM25 retrieval over a user-supplied provenance-preserving JSONL corpus."""

    clean_query = str(query).strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    if corpus_path is None:
        return {
            "query": clean_query,
            "source_kind": source_kind,
            "status": "unavailable",
            "results": [],
            "setup_required": (
                "Configure the corresponding corpus_path. Source documents and "
                "derived text are not distributed with this repository."
            ),
        }
    path = Path(corpus_path)
    if not path.is_file():
        return {
            "query": clean_query,
            "source_kind": source_kind,
            "status": "unavailable",
            "corpus_path": str(path),
            "results": [],
            "setup_required": "The configured JSONL corpus does not exist.",
        }
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            raise ValueError(f"{path}:{line_number}: missing text/content")
        documents.append({"row": row, "text": text, "tokens": _tokens(text)})
    query_tokens = _tokens(clean_query)
    if not documents or not query_tokens:
        return {
            "query": clean_query,
            "source_kind": source_kind,
            "status": "ok",
            "results": [],
        }
    frequencies = Counter(
        token for document in documents for token in set(document["tokens"])
    )
    average_length = (
        sum(len(document["tokens"]) for document in documents) / len(documents)
    ) or 1.0
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        counts = Counter(document["tokens"])
        length = len(document["tokens"])
        score = 0.0
        for token in set(query_tokens):
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            document_frequency = frequencies[token]
            inverse = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            score += inverse * frequency * 2.5 / (
                frequency + 1.5 * (1 - 0.75 + 0.75 * length / average_length)
            )
        if score > 0:
            scored.append((score, document))
    scored.sort(key=lambda item: -item[0])
    results = []
    allowed = (
        "id",
        "chunk_id",
        "source_id",
        "title",
        "source_title",
        "section",
        "section_path",
        "page",
        "page_start",
        "page_end",
        "citation",
        "url",
        "official_url",
        "clinical_review_status",
    )
    for score, document in scored[: max(1, min(int(top_k), 10))]:
        row = document["row"]
        results.append(
            {
                **{key: row[key] for key in allowed if key in row},
                "score": round(score, 6),
                "text": document["text"][:4000],
            }
        )
    return {
        "query": clean_query,
        "source_kind": source_kind,
        "status": "ok",
        "retrieval_method": "local_jsonl_bm25",
        "result_count": len(results),
        "results": results,
        "safety_notice": (
            "Retrieved passages require source-license compliance, provenance "
            "verification, and clinical review."
        ),
    }


def _xml_text(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def search_pubmed(
    query: str,
    *,
    top_k: int,
    cache_dir: str | Path,
    offline: bool = False,
) -> dict[str, Any]:
    """Search NCBI PubMed E-utilities and return titles and abstracts."""

    clean_query = str(query).strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    cache = Path(cache_dir)
    public_params: dict[str, Any] = {
        "db": "pubmed",
        "term": clean_query,
        "retmax": max(1, min(int(top_k), 10)),
        "retmode": "json",
        "sort": "relevance",
        "tool": os.getenv("NCBI_TOOL", "AnesTRACE"),
    }
    email = os.getenv("NCBI_EMAIL")
    if email:
        public_params["email"] = email
    secret = {"api_key": os.environ["NCBI_API_KEY"]} if os.getenv("NCBI_API_KEY") else {}
    search = _json_get(
        f"{NCBI_EUTILS}/esearch.fcgi",
        public_params,
        cache_dir=cache,
        offline=offline,
        secret_params=secret,
    )
    result = search.get("esearchresult") or {}
    identifiers = [str(item) for item in result.get("idlist") or []]
    records: list[dict[str, Any]] = []
    if identifiers:
        fetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(identifiers),
            "retmode": "xml",
            "tool": public_params["tool"],
        }
        if email:
            fetch_params["email"] = email
        body = _cached_get(
            f"{NCBI_EUTILS}/efetch.fcgi",
            fetch_params,
            cache_dir=cache,
            offline=offline,
            secret_params=secret,
        )
        root = ET.fromstring(body)
        for article in root.findall(".//PubmedArticle"):
            citation = article.find("MedlineCitation")
            body_node = article.find(".//Article")
            if citation is None or body_node is None:
                continue
            pmid = _xml_text(citation.find("PMID"))
            abstract = " ".join(
                _xml_text(node)
                for node in body_node.findall(".//Abstract/AbstractText")
                if _xml_text(node)
            )
            records.append(
                {
                    "pmid": pmid,
                    "title": _xml_text(body_node.find("ArticleTitle")),
                    "abstract": abstract,
                    "journal": _xml_text(body_node.find(".//Journal/Title")),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "citation": f"[PubMed PMID:{pmid}]",
                }
            )
    order = {identifier: index for index, identifier in enumerate(identifiers)}
    records.sort(key=lambda row: order.get(row["pmid"], len(order)))
    return {
        "query": clean_query,
        "status": "ok",
        "retrieval_method": "NCBI ESearch then EFetch",
        "total_match_count": int(result.get("count") or 0),
        "result_count": len(records),
        "results": records,
        "retrieved_at": _utc_now(),
        "source": "https://pubmed.ncbi.nlm.nih.gov/",
        "safety_notice": (
            "PubMed results are research literature, not a clinical guideline or "
            "patient-specific treatment instruction."
        ),
    }
