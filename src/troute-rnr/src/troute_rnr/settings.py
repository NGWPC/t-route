import os
import configparser
from pathlib import Path

class Settings:
    def __init__(self):
        config_file = Path.cwd() / "src/troute-rnr/settings.ini"
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.STAGES = set(self.config['STAGES']['stages'].split(','))
        
        self.BASE_URL = self.config['DEFAULT']['BASE_URL']
        self.S3_DOMAIN_URL = self.config['DEFAULT']['S3_DOMAIN_URL']
        self.rate_limit = self.config.getint('DEFAULT', 'rate_limit')
        
        self.rabbitmq_username = self.config['RABBITMQ']['username']
        self.rabbitmq_password = self.config['RABBITMQ']['password']
        self.rabbitmq_host = self.config['RABBITMQ']['host']
        self.rabbitmq_port = self.config.getint('RABBITMQ', 'port')
        
        self.redis_url = self.config['REDIS']['url']
        self.redis_port = self.config.getint('REDIS', 'port')
        
        self.flooded_data_queue = self.config['QUEUES']['flooded_data']
        self.error_queue = self.config['QUEUES']['error']
        
        self.log_path = self.config['PATHS']['log_path']
        
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
