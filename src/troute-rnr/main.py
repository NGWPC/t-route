import json
import socket

import httpx
import pika
from pydantic.error_wrappers import ValidationError
from troute_rnr.format import format_xml, get_site_data, pull_nwm_inputs
from troute_rnr.settings import Settings
from troute_rnr.utils import get

settings = Settings()


def run(
    ch: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
) -> None:
    """The main function for the T-Route replace and route module

    Parameters
    ----------
    ch: pika.channel.Channel
        The Queue we're reading from
    method: pika.spec.Basic.Deliver,
        The delivery object for message acknowledgement
    properties: pika.spec.BasicProperties,
        Pika properties
    body: bytes
        The message content body

    """
    hml = json.loads(body.decode())
    print(f"Reading forecast for {hml['rdf']}, issued at {hml['issuance_time']}")
    site_response = get(hml["rdf"], headers=settings.headers).json()
    sites = format_xml(site_response["productText"])
    for site in sites:
        try:
            gauge_data = get_site_data(site, settings)
            if gauge_data is None:
                continue
        except ValidationError:
            #  ValidationError: Pydantic validation error for the ingested forecast
            continue
        except httpx.HTTPStatusError:
            #  HTTPStatusError: There was no forecast/record within NWPS for the site given
            continue
        inputs = pull_nwm_inputs(gauge_data, settings)
        if inputs is not None:
            print("Forecast successfully read")

    # Acknowledging message since all HML files are read
    ch.basic_ack(delivery_tag=method.delivery_tag)


def consume(settings: Settings) -> None:
    """
    The message consumer interfacing with RabbitMQ

    Parameters
    ----------
    settings : Settings
        Configuration object containing RabbitMQ connection parameters and flooding information

    Raises
    ------
    pika.exceptions.AMQPConnectionError
        If connection to RabbitMQ server fails
    socket.timeout
        If connection attempt times out
    """
    try:
        # Set connection parameters with timeout
        connection_params = pika.URLParameters(settings.pika_url)
        connection_params.socket_timeout = 10  # 10-second timeout for connection

        # Attempt connection with timeout
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
        print(f" [*] Waiting for messages from Queue on URL: {settings.pika_url}")
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=settings.flooded_data_queue, on_message_callback=run)
        channel.start_consuming()
    except (pika.exceptions.AMQPConnectionError, socket.timeout) as e:
        print(f" [!] Failed to connect to RabbitMQ: {e}")
        print(" [!] Service is not running - check RabbitMQ connection")


if __name__ == "__main__":
    consume(Settings())
