def run(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")
    print(" [x] Done")
    ch.basic_ack(delivery_tag=method.delivery_tag)
