from django.contrib import admin
from django.contrib import messages
from django.template.defaultfilters import filesizeformat

from .models import ConversationHistory, RAGChunk, RAGSource
from .rag_ingest import schedule_ingest


@admin.register(RAGSource)
class RAGSourceAdmin(admin.ModelAdmin):
    fields = (
        "title",
        "first_author",
        "has_multiple_authors",
        "uploaded_file",
        "source_size_display",
        "source_type",
        "is_active",
        "ingestion_status",
        "chunk_count",
        "error_message",
        "uploaded_at",
        "processed_at",
    )
    list_display = (
        "title",
        "first_author",
        "has_multiple_authors",
        "source_size_display",
        "source_type",
        "ingestion_status",
        "is_active",
        "chunk_count",
        "uploaded_at",
        "processed_at",
    )
    list_filter = ("source_type", "ingestion_status", "is_active", "uploaded_at")
    search_fields = ("title", "first_author")
    readonly_fields = (
        "source_size_display",
        "ingestion_status",
        "chunk_count",
        "error_message",
        "uploaded_at",
        "processed_at",
    )
    actions = ["reprocess_sources"]

    def save_model(self, request, obj, form, change):
        if obj.uploaded_file:
            obj.source_size_bytes = obj.uploaded_file.size or 0
        super().save_model(request, obj, form, change)
        try:
            schedule_ingest(obj.id)
            try:
                obj.refresh_from_db()
            except Exception:
                self.message_user(
                    request,
                    "Ingestion ran, but status refresh failed. Reload this page to see latest status.",
                    level=messages.WARNING,
                )
                return
            status = obj.ingestion_status
            if status == RAGSource.IngestionStatus.COMPLETED:
                msg = f"✓ Ingestion completed: {obj.chunk_count} chunks created."
                self.message_user(request, msg, level=messages.SUCCESS)
            elif status == RAGSource.IngestionStatus.FAILED:
                msg = f"✗ Ingestion failed: {obj.error_message}"
                self.message_user(request, msg, level=messages.ERROR)
            else:
                self.message_user(request, f"Ingestion status: {status}", level=messages.INFO)
        except Exception as exc:
            msg = f"Ingestion error: {obj.error_message or str(exc)}"
            self.message_user(request, msg, level=messages.ERROR)

    @admin.action(description="Reprocess selected RAG sources")
    def reprocess_sources(self, request, queryset):
        success_count = 0
        error_count = 0
        for source in queryset:
            try:
                schedule_ingest(source.id)
                source.refresh_from_db()
                if source.ingestion_status == RAGSource.IngestionStatus.COMPLETED:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        msg_parts = []
        if success_count:
            msg_parts.append(f"{success_count} source(s) completed")
        if error_count:
            msg_parts.append(f"{error_count} source(s) failed")
        
        msg = "; ".join(msg_parts) if msg_parts else "No sources processed."
        level = messages.SUCCESS if error_count == 0 else messages.WARNING
        self.message_user(request, msg, level=level)

    @admin.display(description="Source Size")
    def source_size_display(self, obj):
        return filesizeformat(obj.source_size_bytes or 0)


@admin.register(RAGChunk)
class RAGChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "chunk_index", "chunk_size_display", "embedding_dim", "created_at")
    list_filter = ("source",)
    search_fields = ("content",)
    readonly_fields = (
        "source",
        "chunk_index",
        "chunk_size_display",
        "content",
        "embedding",
        "embedding_dim",
        "metadata",
        "created_at",
    )

    @admin.display(description="Chunk Size")
    def chunk_size_display(self, obj):
        return filesizeformat(obj.chunk_size_bytes or 0)


@admin.register(ConversationHistory)
class ConversationHistoryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "session_key", "created_at")
    list_filter = ("user", "created_at")
    search_fields = ("user__username", "session_key", "user_query", "assistant_response")
    readonly_fields = ("user", "session_key", "user_query", "assistant_response", "retrieved_context", "created_at")
