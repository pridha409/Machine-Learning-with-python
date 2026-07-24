import logging
from datetime import datetime

logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def log_prediction(address, probability, prediction, latency_ms):
    logging.info(f"{address} | {probability} | {prediction} | {latency_ms:.2f}ms")