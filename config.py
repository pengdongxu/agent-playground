import os
from pathlib import Path


class Config:
    """
    配置类
    """
    # 项目根目录
    ROOT_DIR = Path(__file__).parent

    # 基础存储目录(从环境变量中读取, 默认为data)
    DATA_DIR = ROOT_DIR / os.getenv("DATA_BASE_DIR", "data")

    # 子目录
    HISTORY_DIR = DATA_DIR / os.getenv("HISTORY_SUBDIR", "history")
    PROFILE_DIR = DATA_DIR / os.getenv("PROFILE_SUBDIR", "profiles")

    @classmethod
    def init_dirs(cls):
        """初始化所有必要的文件夹"""
        cls.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: 存储目录已初始化: {cls.DATA_DIR}")

# 在程序启动时自动执行初始化
Config.init_dirs()