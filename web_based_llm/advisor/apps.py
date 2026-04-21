from django.apps import AppConfig


class AdvisorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'advisor'

    def ready(self):
        """Pre-warm embedding model on app startup to avoid timeout on first ingestion."""
        from .rag_ingest import _warmup_embedding_model
        _warmup_embedding_model()
