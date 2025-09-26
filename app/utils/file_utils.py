# app/utils/file_utils.py
import os
import tempfile
import requests
from fastapi import HTTPException

def download_pdf_from_s3(url: str, timeout: int = 30) -> str:
    """
    안전하게 PDF를 다운로드하여 임시 파일 경로 반환.
    Caller is responsible for deleting the file when done.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            # S3 may serve application/octet-stream — allow common PDF types but do a content check later
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tf.write(chunk)
                tmp_path = tf.name

        # Quick sanity check (file size)
        if os.path.getsize(tmp_path) < 200:  # too small to be a valid PDF
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Downloaded file seems too small to be a valid PDF")

        return tmp_path
    except requests.HTTPError as e:
        raise HTTPException(status_code=404, detail=f"Failed to download file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {e}")
