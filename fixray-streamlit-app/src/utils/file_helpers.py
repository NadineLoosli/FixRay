def save_image(image, path):
    """Saves an image to the specified path."""
    image.save(path)

def load_image(path):
    """Loads an image from the specified path."""
    from PIL import Image
    return Image.open(path)

def delete_file(path):
    """Deletes a file at the specified path."""
    import os
    if os.path.exists(path):
        os.remove(path)

def get_file_extension(filename):
    """Returns the file extension of the given filename."""
    return os.path.splitext(filename)[1]