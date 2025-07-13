import httpx
import logging
import os
import tempfile

from fastapi import HTTPException
from app.core.constant import Constants

logger = logging.getLogger("app")

def file_ext(url: str):
    return url.split("?")[0].split(".")[-1].lower()

def looks_like_text(content: bytes) -> bool:
    try:
        content.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

async def validate_url(client: httpx.AsyncClient, url: str, expected_type: Constants.EXPECTED_TYPE) -> bool:
    try:
        head = await client.head(url, timeout=10.0)
        ctype = head.headers.get("Content-Type", "")

        if head.status_code == 200:
            if expected_type == "text" and any(v in ctype for v in Constants.SUPPORTED_TEXT_MIME_TYPES):
                return True

        get = await client.get(url, headers={"Range": "bytes=0-1023"}, timeout=10.0)
        if get.status_code in (200, 206):
            content = get.content
            if expected_type == "text" and looks_like_text(content):
                return True

        ext = file_ext(url)
        if expected_type == "text" and ext in Constants.VALIDATE_TEXT_EXT:
            return True
        
    except httpx.RequestError:
        pass
    return False

async def download_file(client: httpx.AsyncClient, url: str, save_path: str):
    try:
        async with client.stream("GET", url, timeout=60.0) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Failed to download {url}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download {url}: {e}")
    
async def parse_file_to_context(file_url):
    async with httpx.AsyncClient() as client:
        if not await validate_url(client, file_url, "text"):
            raise ValueError(f"Invalid or unsupported URL: {file_url}")

        with tempfile.TemporaryDirectory() as temp_dir:
            _, file_ext = os.path.splitext(file_url)
            text_ext = file_ext if file_ext.lower() in Constants.VALIDATE_TEXT_EXT else ".txt"
            file_path = os.path.join(temp_dir, f"downloaded_file{text_ext}")

            await download_file(client, file_url, file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                context = f.read()
            
            if not context:
                logger.warning(f"No context found in file from {file_url}")
                return []

            return context
        
def split_into_chunks(text: str):
        lines = text.splitlines()
        return ["\n".join(lines[i:i + Constants.CHUNK_SIZE]) for i in range(0, len(lines), Constants.CHUNK_SIZE)]