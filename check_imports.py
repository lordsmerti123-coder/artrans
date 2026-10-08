import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def check_imports():
    print(f"--- Checking imports in {Path(__file__).name} ---")
    modules_to_test = [
        "gui.main_window",
        "src.model_manager",
        "src.audio_pipeline",
        "src.nmt",
        "src.tts",
        "src.language_detector",
        "src.audio_devices",
        "src.segmentation",
        "src.language_gate",
        "src.noise",
        "src.audio_router",
        "src.session_log",
    ]
    
    failed = []
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError as e:
            print(f"[FAILED] {module} - ImportError: {e}")
            failed.append(module)
        except Exception as e:
            print(f"[ERROR] {module} - {type(e).__name__}: {e}")
            failed.append(module)
            
    if failed:
        print("\nSummary: Some modules failed to import.")
        sys.exit(1)
    else:
        print("\nSummary: All core modules imported successfully.")
        sys.exit(0)

if __name__ == "__main__":
    check_imports()
