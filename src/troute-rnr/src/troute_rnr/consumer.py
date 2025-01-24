import pika

from troute_rnr.settings import Settings
from troute_rnr.replace_and_route import run

def consume(settings: Settings) -> None:
    connection = pika.BlockingConnection(pika.URLParameters(settings.pika_url))
    channel = connection.channel()

    channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.flooded_data_queue, on_message_callback=run)

    channel.start_consuming()


if __name__ == "__main__":
    consume(Settings())
