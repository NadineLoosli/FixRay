import os
from PIL import Image


def save_image(image: Image.Image, path: str) -> None:
    """Speichert ein PIL-Image unter dem angegebenen Pfad."""
    image.save(path)


def load_image(path: str) -> Image.Image:
    """Lädt ein Bild vom angegebenen Pfad als PIL-Image."""
    return Image.open(path)


def delete_file(path: str) -> None:
    """Löscht eine Datei, falls sie existiert."""
    if os.path.exists(path):
        os.remove(path)


def get_file_extension(filename: str) -> str:
    """Gibt die Dateiendung (inkl. Punkt) des Dateinamens zurück."""
    return os.path.splitext(filename)[1]
