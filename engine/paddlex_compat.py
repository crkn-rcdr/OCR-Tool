from functools import lru_cache
import importlib
import sys


def apply_headless_opencv_patch():
    """
    Let PaddleX accept the headless OpenCV wheel in WSL/Linux environments.

    PaddleX checks for the distribution name `opencv-contrib-python`, but in WSL
    we prefer `opencv-contrib-python-headless` to avoid a hard dependency on
    `libGL.so.1`. Both provide the same `cv2` module for this app's use case.
    """
    try:
        import paddlex.utils.deps as deps
    except Exception:
        return

    if getattr(deps, "_ocr_tool_headless_patch", False):
        return

    original_is_dep_available = deps.is_dep_available

    if original_is_dep_available("opencv-contrib-python"):
        deps._ocr_tool_headless_patch = True
        return

    if not original_is_dep_available("opencv-contrib-python-headless"):
        return

    @lru_cache()
    def patched_is_dep_available(dep, /, check_version=False):
        if dep == "opencv-contrib-python":
            return original_is_dep_available("opencv-contrib-python-headless")
        return original_is_dep_available(dep, check_version=check_version)

    deps.is_dep_available = patched_is_dep_available
    deps.is_extra_available.cache_clear()

    try:
        cv2 = importlib.import_module("cv2")
    except Exception:
        cv2 = None

    if cv2 is not None:
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith("paddlex."):
                continue

            if not hasattr(module, "cv2"):
                module.cv2 = cv2

            if getattr(module, "is_dep_available", None) is original_is_dep_available:
                module.is_dep_available = patched_is_dep_available

    deps._ocr_tool_headless_patch = True
