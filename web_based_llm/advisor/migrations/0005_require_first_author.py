from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("advisor", "0004_rename_name_ragsource_title_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="ragsource",
            name="first_author",
            field=models.CharField(max_length=255),
        ),
    ]
