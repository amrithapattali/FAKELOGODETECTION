# Generated by Django 5.0.1 on 2024-04-08 07:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0007_rename_ram_product_size_remove_product_rom_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='product',
            name='battery',
        ),
        migrations.RemoveField(
            model_name='product',
            name='warranty',
        ),
    ]