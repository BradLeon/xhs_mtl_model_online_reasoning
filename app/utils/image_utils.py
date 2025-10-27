import base64
import io
import requests
from PIL import Image
from typing import Union, Optional
from loguru import logger


def decode_image(image_data: str) -> Optional[Image.Image]:
    """
    解码图像数据（支持base64和URL）
    
    Args:
        image_data: base64编码的图像或图像URL
        
    Returns:
        PIL Image对象，失败返回None
    """
    try:
        # 检查是否为URL
        if image_data.startswith(('http://', 'https://')):
            response = requests.get(image_data, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        
        # 处理base64编码
        else:
            # 移除可能的data URI前缀
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # 解码base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
            
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将PIL Image转换为base64编码
    
    Args:
        image: PIL Image对象
        format: 图像格式
        
    Returns:
        base64编码的字符串
    """
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        return ""


def resize_image(image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """
    调整图像大小，保持纵横比
    
    Args:
        image: PIL Image对象
        max_size: 最大尺寸(width, height)
        
    Returns:
        调整后的图像
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        return image


def validate_image(image_data: str) -> bool:
    """
    验证图像数据是否有效
    
    Args:
        image_data: base64编码的图像或图像URL
        
    Returns:
        是否有效
    """
    image = decode_image(image_data)
    return image is not None