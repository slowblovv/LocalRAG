from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_root: str = "./knowledge"
    index_root: str = "./indexes"
    metadata_db_url: str = "sqlite:///./localrag.db"
    job_db_path: str = "./localrag_jobs.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_type: str = "numpy"
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    llm_provider: str = "fake"
    llm_model: str = "fake-v1"
    max_context_tokens: int = 6000
    top_k_lexical: int = 20
    top_k_vector: int = 20
    top_k_rerank: int = 10
    cache_ttl: int = 300
    max_query_length: int = 4096
    max_top_k: int = 50
    max_context_chunks: int = 20
    max_document_size: int = 25 * 1024 * 1024
    refuse_top_score: float = 0.65
    prompt_version: str = "v2-grounded"
    worker_lease_seconds: int = 300
    worker_poll_seconds: float = 1.0
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
