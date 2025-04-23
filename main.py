from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
