import tiktoken

from src.ingestion.chunking import chunk_text
from src.ingestion.config import ChunkingConfig


def test_chunk_text_respects_token_budget_and_overlap() -> None:
    config = ChunkingConfig(chunk_size_tokens=50, chunk_overlap_tokens=10)
    paragraph = "The quarterly revenue increased due to higher electricity demand. " * 40
    chunks = chunk_text(paragraph, config)

    assert len(chunks) > 1

    encoding = tiktoken.get_encoding(config.encoding_name)
    for chunk in chunks:
        assert len(encoding.encode(chunk)) <= config.chunk_size_tokens


def test_chunk_text_drops_empty_pieces() -> None:
    config = ChunkingConfig(chunk_size_tokens=50, chunk_overlap_tokens=5)
    assert chunk_text("   \n\n  ", config) == []
