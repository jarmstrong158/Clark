"""Minimal local-only Clark inference endpoint.

Sanctioned by NOTE.md ("Cloud / API layer — scrapped; minimal local
inference endpoint sanctioned"): the speculative cloud skeleton stays
dead; this is a strictly localhost, stateless, single-consumer adapter
that exists because the local AI warehouse system (Ollama-hosted
fine-tuned LLM -> clark-mcp) is a real consumer that needs to call
Clark for inference. See dec-029.
"""
from clark.serve.app import build_app

__all__ = ["build_app"]
