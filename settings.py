import os
import yaml
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()


class Settings:
    def __init__(self, config_path="config.yaml"):
        # Load constants from YAML
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # SECRETS
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

        # MODELS
        self.FAST_MODEL = self.config['models']['fast_model']
        self.QUALITY_MODEL = self.config['models']['quality_model']

        # PATHS
        self.DATA_PATH = self.config['retrieval']['data_path']
        self.VECTOR_DB_PATH = self.config['retrieval']['vector_db_path']

        # LIMITS
        self.BATCH_SIZE = self.config['processing']['batch_size']
        self.SLEEP_TIME = self.config['processing']['sleep_time']
        self.TOP_K_MASTER = self.config['retrieval']['top_k_master']

        # TELEMETRY
        self.LANGSMITH_PROJECT = self.config['telemetry']['project_name']
        self.LANGSMITH_ENDPOINT = self.config['telemetry']['endpoint']

        # EVALUATION PATHS
        self.GOLDEN_DATASET = self.config['evaluation']['golden_dataset_path']
        self.QUALITY_REPORT = self.config['evaluation']['quality_report_path']

# Global settings instance
settings = Settings()