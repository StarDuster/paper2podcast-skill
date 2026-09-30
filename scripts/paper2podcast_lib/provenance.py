"""Content-bound review receipts (integrity metadata, not signatures)."""
import hashlib
import json


def text_hash(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def transcript_hash(entries):
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def review_receipt(entries, source, provider, model, base_url=None):
    return {'version': 1, 'status': 'reviewed', 'reviewed': True,
            'transcript_sha256': transcript_hash(entries),
            'source_sha256': text_hash(source), 'source_hash_kind': 'extracted-text-utf8',
            'provider': provider.strip().lower(), 'model': model.strip(),
            'base_url': (base_url or '').strip()}


def review_matches(receipt, entries, source, provider, model, base_url=None):
    return receipt == review_receipt(entries, source, provider, model, base_url)
