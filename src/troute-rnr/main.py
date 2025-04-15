import json

import pika
from troute_rnr.format import format_xml, pull_nwm_inputs
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
    site_data = get(hml["rdf"], headers=settings.headers).json()
    forecasts = format_xml(site_data["productText"], settings)
    if len(forecasts) == 0:
        # There is no forecast present in this message. End the process
        ch.basic_ack(delivery_tag=method.delivery_tag)
    else:
        for forecast in forecasts:
            inputs = pull_nwm_inputs(forecast, settings)
            if inputs is None:
                # The streamflow forecast is -999
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                print("Forecast successfully read")

        ch.basic_ack(delivery_tag=method.delivery_tag)


def consume(settings: Settings) -> None:
    """The message consumer function to handle Rabbit MQ interactions

    Parameters
    ----------
    settings: Settings
    """
    connection = pika.BlockingConnection(pika.URLParameters(settings.pika_url))
    channel = connection.channel()

    channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
    print(f" [*] Waiting for messages from Queue on URL: {settings.pika_url}")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.flooded_data_queue, on_message_callback=run)

    channel.start_consuming()


if __name__ == "__main__":
    consume(Settings())
