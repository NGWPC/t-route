
import os

from dotenv import load_dotenv
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STAGES : set = {
        "action",
        "minor",
        "major",
        "moderate",
    }

    BASE_URL : str = "https://api.water.noaa.gov/nwps/v1"

    S3_DOMAIN_URL: str = "s3://fim-services-data/replace-and-route/v0.2.0/domain_gpkgs"

    rate_limit: int = 8

    rabbitmq_default_username: str = "guest"
    rabbitmq_default_password: str = "guest"
    rabbitmq_default_host: str = "localhost"
    rabbitmq_default_port: int = 5672

    pika_url: str = "amqp://{}:{}@{}:{}/"
    redis_url: str = "localhost"
    redis_port: int = 6379

    flooded_data_queue: str = "flooded_data_queue"
    error_queue: str = "error_queue"

    log_path: str = "/app/data/logs"

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        super(Settings, self).__init__(**data)

        load_dotenv()

        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')

        if os.getenv("RABBITMQ_HOST") is not None:
            self.rabbitmq_default_host = os.getenv("RABBITMQ_HOST")  

        self.pika_url = self.pika_url.format(
            self.rabbitmq_default_username,
            self.rabbitmq_default_password,
            self.rabbitmq_default_host,
            self.rabbitmq_default_port,
        )

        if os.getenv("PIKA_URL") is not None:
            self.pika_url = os.getenv("PIKA_URL")
        if os.getenv("REDIS_URL") is not None:
            self.redis_url = os.getenv("REDIS_URL")
        if os.getenv("SUBSET_URL") is not None:
            self.base_subset_url = os.getenv("SUBSET_URL")
        if os.getenv("TROUTE_URL") is not None:
            self.base_troute_url = os.getenv("TROUTE_URL")
