from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("advisor", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="ragsource",
            name="ingestion_status",
            field=models.CharField(
                choices=[
                    ("queued", "Queued"),
                    ("processing", "Processing"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                ],
                default="completed",
                max_length=12,
            ),
        ),
    ]
