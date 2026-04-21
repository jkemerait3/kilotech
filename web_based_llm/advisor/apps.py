from django.apps import AppConfig
import os
import sys


class AdvisorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'advisor'

    def ready(self):
        """Pre-warm embedding model on app startup to avoid timeout on first ingestion."""
        management_commands_to_skip = {
            'migrate',
            'makemigrations',
            'collectstatic',
            'shell',
            'createsuperuser',
            'test',
            'check',
        }
        if any(cmd in sys.argv for cmd in management_commands_to_skip):
            return

        if os.getenv('SKIP_EMBEDDING_WARMUP', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            return

        from .rag_ingest import _warmup_embedding_model
        _warmup_embedding_model()
