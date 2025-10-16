# boot.py
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent.parent / 'configs' / 'default.env'
load_dotenv(dotenv_path=str(env_path))

