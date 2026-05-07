import cv2
from image_checks import analyze_quality
import gradio as gr


def main():
    image_path = "C:\\CRKN Projects\\paddle_pipeline\\download.jpg"
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    result = analyze_quality(img)

    print("===== QUALITY CHECK =====")

    print(f"Skew: {'YES' if result.has_skew else 'NO'}")
    print(f"Warping: {'YES' if result.has_warping else 'NO'}")
    print(f"Illumination Problem: {'YES' if result.has_illumination_problem else 'NO'}")


if __name__ == "__main__":
    main()