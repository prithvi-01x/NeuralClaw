"""
memory/embedder.py — Text Embedder

Wraps sentence-transformers for local embedding generation.
Runs the (synchronous) model in a thread pool executor so it
doesn't block the async event loop.

Model: BAAI/bge-small-en-v1.5 (default) — fast, accurate, ~130MB
Fallback: all-MiniLM-L6-v2 — smaller, slightly less accurate

Usage:
    embedder = Embedder()
    await embedder.load()
    vector = await embedder.embed("Python async programming")
    vectors = await embedder.embed_batch(["text 1", "text 2"])
"""

from __future__ import annotations

import atexit
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Async wrapper around a local sentence-transformers model.

    The first call to embed() will load the model (lazy init).
    Subsequent calls reuse the loaded model.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")
        self._lock = asyncio.Lock()
        # Register clean shutdown on any process exit (SIGTERM, Ctrl+C, normal exit).
        # wait=False so the GC doesn't block indefinitely on in-flight embeds.
        atexit.register(self._executor.shutdown, wait=False)

    async def load(self) -> None:
        """
        Explicitly load the embedding model.
        Called at startup so the first embed() call isn't slow.
        """
        async with self._lock:
            if self._model is not None:
                return
            await self._load_model()

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a float vector.

        Args:
            text: Text to embed (will be truncated by model if too long).

        Returns:
            List of floats representing the embedding vector.
        """
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(
            self._executor,
            self._encode_single,
            text,
        )
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single model call (more efficient).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            self._executor,
            self._encode_batch,
            texts,
        )
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        """Return the embedding dimension for the loaded model."""
        if self._model is None:
            # Return known dimension for common models
            if "bge-small" in self.model_name:
                return 384
            if "MiniLM-L6" in self.model_name:
                return 384
            return 384  # safe default
        return self._model.get_sentence_embedding_dimension()

    # ── Private ───────────────────────────────────────────────────────────────

    async def _ensure_loaded(self) -> None:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    await self._load_model()

    async def _load_model(self) -> None:
        """Load the model in a thread (import + download is blocking)."""
        loop = asyncio.get_running_loop()
        try:
            log.info("embedder.loading", model=self.model_name)
            self._model = await loop.run_in_executor(
                self._executor,
                self._import_and_load,
                self.model_name,
            )
            log.info(
                "embedder.loaded",
                model=self.model_name,
                dimension=self.dimension(),
            )
        except (OSError, RuntimeError, ImportError, ValueError, AttributeError) as e:
            if self.model_name != _FALLBACK_MODEL:
                log.warning(
                    "embedder.load_failed_trying_fallback",
                    model=self.model_name,
                    fallback=_FALLBACK_MODEL,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self.model_name = _FALLBACK_MODEL
                self._model = await loop.run_in_executor(
                    self._executor,
                    self._import_and_load,
                    _FALLBACK_MODEL,
                )
            else:
                raise EmbedderError(f"Failed to load embedding model: {e}") from e

    @staticmethod
    def _import_and_load(model_name: str):
        """Synchronous model load — runs in thread pool."""
        import sys, io, logging, warnings, os
        from sentence_transformers import SentenceTransformer
        
        # Suppress HF Hub unauthenticated warning and tqdm
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TQDM_DISABLE", "1")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._headers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

        class _SuppressStdout:
            def __enter__(self):
                self._orig = sys.stdout
                sys.stdout = io.StringIO()
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("mlx").setLevel(logging.CRITICAL)
                return self
            def __exit__(self, *_):
                sys.stdout = self._orig
                logging.getLogger("transformers").setLevel(logging.WARNING)
                
        with _SuppressStdout():
            return SentenceTransformer(model_name)

    def _encode_single(self, text: str):
        """Synchronous single encode — runs in thread pool."""
        return self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)

    def _encode_batch(self, texts: list[str]):
        """Synchronous batch encode — runs in thread pool."""
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def close(self) -> None:
        """Shut down the thread-pool executor cleanly to avoid interpreter-shutdown warnings."""
        self._executor.shutdown(wait=True)
        log.debug("embedder.executor_shutdown")

    def __repr__(self) -> str:
        loaded = "loaded" if self._model else "not loaded"
        return f"<Embedder model={self.model_name} status={loaded}>"


class EmbedderError(Exception):
    """Raised when the embedding model fails to load or encode."""