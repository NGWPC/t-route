import configparser
import os
from pathlib import Path

import boto3
from icefabric_manage import build
from pyiceberg.catalog import load_catalog


class Settings:
    """The global settings object for Troute-rnr.

    Notes
    -----
    This will assume you are running the code in the troute-rnr/ dir
    """

    def __init__(self):
        module_dir = Path.cwd()
        project_root = module_dir.parents[1]

        config_file = module_dir / "settings.ini"
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.headers = {
            "Accept": self.config["HEADERS"]["accept"],
            "User-Agent": self.config["HEADERS"]["user_agent"],
        }

        self.STAGES = set(self.config["STAGES"]["stages"].split(","))

        self.BASE_URL = self.config["DEFAULT"]["BASE_URL"]
        self.S3_DOMAIN_URL = self.config["DEFAULT"]["S3_DOMAIN_URL"]
        self.rate_limit = self.config.getint("DEFAULT", "rate_limit")
        self.reach_limit = self.config.getint("DEFAULT", "reach_limit")

        self.rabbitmq_username = self.config["RABBITMQ"]["username"]
        self.rabbitmq_password = self.config["RABBITMQ"]["password"]
        self.rabbitmq_host = self.config["RABBITMQ"]["host"]
        self.rabbitmq_port = self.config.getint("RABBITMQ", "port")

        self.redis_url = self.config["REDIS"]["url"]
        self.redis_port = self.config.getint("REDIS", "port")

        self.flooded_data_queue = self.config["QUEUES"]["flooded_data"]
        self.error_queue = self.config["QUEUES"]["error"]

        self.log_path = self.config["PATHS"]["log_path"]
        self.user = self.config["DEFAULT"]["user"]

        self.data_dir = (project_root / "data").resolve()
        self.catalog_settings = {
            "type": "sql",
            "uri": f"sqlite:///{str(self.data_dir / 'warehouse/pyiceberg_catalog.db')}",  # Note the three slashes for absolute path
            "warehouse": f"file://{str(self.data_dir.resolve())}/warehouse",  # Use resolved absolute path
        }

        self.catalog = load_catalog("hydrofabric", **self.catalog_settings)
        build(self.catalog, Path(f"{self.data_dir.resolve()}/parquet"))

        self.base_config_path = module_dir / "base_files/base_config.yaml"
        self.tmp_config = module_dir / "base_files/tmp_config.yaml"
        self.tmp_geopackage = module_dir / "base_files/tmp_domain.gpkg"

        if os.getenv("RABBITMQ_HOST"):
            self.rabbitmq_host = os.getenv("RABBITMQ_HOST")
        if os.getenv("RABBITMQ_USERNAME"):
            self.rabbitmq_username = os.getenv("RABBITMQ_HOST")
        if os.getenv("RABBITMQ_PASSWORD"):
            self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD")

        self.pika_url = f"amqp://{self.rabbitmq_username}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"

        if os.getenv("PIKA_URL"):
            self.pika_url = os.getenv("PIKA_URL")

        if os.getenv("REDIS_URL"):
            self.redis_url = os.getenv("REDIS_URL")

        session = boto3.Session()
        credentials = session.get_credentials()

        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        if credentials.token:  # If you're using temporary credentials
            os.environ["AWS_SESSION_TOKEN"] = credentials.token
