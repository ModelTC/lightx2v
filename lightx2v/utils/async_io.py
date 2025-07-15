import aiofiles
import asyncio
from PIL import Image
import io
from typing import Union
from pathlib import Path
from loguru import logger


async def load_image_async(path: Union[str, Path]) -> Image.Image:
    """异步加载图片"""
    try:
        async with aiofiles.open(path, 'rb') as f:
            data = await f.read()
        
        # 在线程池中处理图片解码
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as e:
        logger.error(f"Failed to load image from {path}: {e}")
        raise


async def save_video_async(video_path: Union[str, Path], video_data: bytes):
    """异步保存视频"""
    try:
        # 确保目录存在
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(video_path, 'wb') as f:
            await f.write(video_data)
        
        logger.info(f"Video saved to {video_path}")
    except Exception as e:
        logger.error(f"Failed to save video to {video_path}: {e}")
        raise


async def read_text_async(path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """异步读取文本文件"""
    try:
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Failed to read text from {path}: {e}")
        raise


async def write_text_async(path: Union[str, Path], content: str, encoding: str = 'utf-8'):
    """异步写入文本文件"""
    try:
        # 确保目录存在
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, 'w', encoding=encoding) as f:
            await f.write(content)
        
        logger.info(f"Text written to {path}")
    except Exception as e:
        logger.error(f"Failed to write text to {path}: {e}")
        raise


async def exists_async(path: Union[str, Path]) -> bool:
    """异步检查文件是否存在"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Path(path).exists())


async def read_bytes_async(path: Union[str, Path]) -> bytes:
    """异步读取二进制文件"""
    try:
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Failed to read bytes from {path}: {e}")
        raise


async def write_bytes_async(path: Union[str, Path], data: bytes):
    """异步写入二进制文件"""
    try:
        # 确保目录存在
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)
        
        logger.debug(f"Bytes written to {path}")
    except Exception as e:
        logger.error(f"Failed to write bytes to {path}: {e}")
        raise