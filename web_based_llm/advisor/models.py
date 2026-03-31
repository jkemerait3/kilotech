from django.db import models
from django.contrib.auth.models import User


class RAGSource(models.Model):
    class IngestionStatus(models.TextChoices):
        QUEUED = "queued", "Queued"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    AUTHOR_COUNT_CHOICES = [
        (False, "No"),
        (True, "Yes"),
    ]

    title = models.CharField(max_length=255)
    first_author = models.CharField(max_length=255)
    has_multiple_authors = models.BooleanField(default=False, choices=AUTHOR_COUNT_CHOICES)
    source_size_bytes = models.PositiveBigIntegerField(default=0)
    uploaded_file = models.FileField(upload_to="rag_sources/%Y/%m/%d/")
    source_type = models.CharField(max_length=20, default="unknown")
    ingestion_status = models.CharField(
        max_length=12,
        choices=IngestionStatus.choices,
        default=IngestionStatus.COMPLETED,
    )
    is_active = models.BooleanField(default=True)
    chunk_count = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    @property
    def citation_author(self):
        author = (self.first_author or "").strip()
        if self.has_multiple_authors:
            return f"{author} et. al"
        return author

    def __str__(self):
        return f"{self.title} ({self.chunk_count} chunks)"


class RAGChunk(models.Model):
    source = models.ForeignKey(RAGSource, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    content = models.TextField()
    embedding = models.BinaryField()
    embedding_dim = models.PositiveIntegerField(default=384)
    metadata = models.JSONField(default=dict, blank=True)
    chunk_size_bytes = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["source", "chunk_index"]),
        ]
        unique_together = ("source", "chunk_index")

    def __str__(self):
        return f"{self.source.title} #{self.chunk_index}"


class ConversationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="conversation_history", null=True, blank=True)
    session_key = models.CharField(max_length=100, db_index=True)
    user_query = models.TextField()
    assistant_response = models.TextField()
    retrieved_context = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        owner = self.user.username if self.user else self.session_key
        return f"{owner} @ {self.created_at:%Y-%m-%d %H:%M:%S}"
