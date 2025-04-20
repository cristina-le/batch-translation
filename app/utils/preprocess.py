from typing import List
import os

def reader(file_path: str, size: int = 100) -> List[str]:
    """
    Read text from file and split into chunks of lines.

    Args:
        file_path (str): Path to the text file
        size (int): Number of lines per chunk (default 100)

    Returns:
        list: List of text segments, each containing up to 'size' lines
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.readlines()

            segments = [
                "".join(text[i:i+size]) 
                for i in range(0, len(text), size)
            ]

        print(f"Split text into {len(segments)} line-based chunks")
        return segments

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def writer(file_path: str, data: List[str]) -> bool:
    """
    Write segments to a file, each chunk followed by a newline.

    Args:
        file_path (str): Path to write the data to
        data (list): List of text segments

    Returns:
        bool: Success status
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("\n".join(data))
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False
