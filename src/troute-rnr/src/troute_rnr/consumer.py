import json

import pika

from nwm_routing.__main__ import main_v04 as t_route
from troute_rnr.format import format_xml, pull_nwm_inputs, write_forcast_csvs, write_config
from troute_rnr.replace_and_route import read_remote_gpkg
from troute_rnr.settings import Settings
from troute_rnr.utils import get

settings = Settings()

def run(
   ch: pika.channel.Channel,
   method: pika.spec.Basic.Deliver, 
   properties: pika.spec.BasicProperties,
   body: bytes
):
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
                gdf = read_remote_gpkg(forecast.lid)
                write_forcast_csvs(gdf, inputs)
                restart_file = create_initial_start_file(params, settings)
                yaml_file_path = write_config(base_config, params, restart_file)
                t_route(["-f", yaml_file_path.__str__()])
                yaml_file_path.unlink()
            
        ch.basic_ack(delivery_tag=method.delivery_tag)

def consume(settings: Settings) -> None:
    connection = pika.BlockingConnection(pika.URLParameters(settings.pika_url))
    channel = connection.channel()

    channel.queue_declare(queue=settings.flooded_data_queue, durable=True)
    print(f' [*] Waiting for messages from Queue on URL: {settings.pika_url}')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.flooded_data_queue, on_message_callback=run)

    channel.start_consuming()


if __name__ == "__main__":
    consume(Settings())
