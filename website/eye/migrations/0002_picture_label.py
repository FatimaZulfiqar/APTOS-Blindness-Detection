# Generated by Django 3.1.2 on 2021-08-13 22:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('eye', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='picture',
            name='label',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]