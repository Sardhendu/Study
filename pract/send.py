import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello World!')
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Baba ji ki Booti!')
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Jama Gaga Bala Gasta')
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Japru mucha kala pucha')
print(" [x] Sent 'Hello World!'")
connection.close()