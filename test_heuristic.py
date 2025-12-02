import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.inference.predict_single_image import analyze_image


def test_analyze_image_on_uniform_image(tmp_path):
    img = Image.new("RGB", (512, 512), color=(200, 200, 200))
    img_path = tmp_path / "uniform.png"
    img.save(img_path)

    res = analyze_image(
        image_path=str(img_path),
        model=None,
        output_dir=None,
        confidence_threshold=0.15,
    )

    assert res["status"] == "SUCCESS"
    assert res["score"] < 0.05
