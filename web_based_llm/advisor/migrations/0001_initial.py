from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ConversationHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("session_key", models.CharField(db_index=True, max_length=100)),
                ("user_query", models.TextField()),
                ("assistant_response", models.TextField()),
                ("retrieved_context", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="RAGSource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255)),
                ("uploaded_file", models.FileField(upload_to="rag_sources/%Y/%m/%d/")),
                ("source_type", models.CharField(default="unknown", max_length=20)),
                ("is_active", models.BooleanField(default=True)),
                ("chunk_count", models.PositiveIntegerField(default=0)),
                ("error_message", models.TextField(blank=True)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("processed_at", models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="RAGChunk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chunk_index", models.PositiveIntegerField()),
                ("content", models.TextField()),
                ("embedding", models.BinaryField()),
                ("embedding_dim", models.PositiveIntegerField(default=384)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "source",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="chunks", to="advisor.ragsource"),
                ),
            ],
            options={
                "indexes": [models.Index(fields=["source", "chunk_index"], name="advisor_ragc_source__4f4ecc_idx")],
                "unique_together": {("source", "chunk_index")},
            },
        ),
    ]
