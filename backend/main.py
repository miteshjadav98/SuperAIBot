"""Entry point for the FastAPI gateway (port 8000).

The app now lives in ``api/app.py``. This module re-exports it so existing
commands (`python main.py`, `uvicorn main:app`) and start.ps1 keep working.
"""

from api.app import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
