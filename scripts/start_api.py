"""Simple script to start the FastAPI server"""
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Smart Grid IDS API Server...")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("  - Health Check: http://127.0.0.1:8000/health")
    print("  - Interactive Docs: http://127.0.0.1:8000/docs")
    print("  - Predict: http://127.0.0.1:8000/predict")
    print("  - Prevent: http://127.0.0.1:8000/prevent")
    print("\n" + "=" * 60)
    print("Server is starting... Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    try:
        uvicorn.run(
            "src.api:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

