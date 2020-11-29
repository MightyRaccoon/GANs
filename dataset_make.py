import os
import logging

import click
from selenium import webdriver

from NetUtils.utils import persist_image, fetch_image_urls

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level="INFO"
)


@click.command()
@click.option("--data-root", default="//home/mightyraccoon/GitHub/GANs/data_generated/parsed/")
@click.option("--dataset-size", default=200)
def main(data_root, dataset_size):

    queries_list = [
        "Panda",
        "Панды",
        "Панды фото",
        "Панды в дикой природе",
        "Панды в заоопарке",
        "Giant panda",
        "Большая панда"
    ]

    if not os.path.exists(data_root):
        os.makedirs(data_root)
        logger.info("Directory is created")

    with webdriver.Chrome(executable_path="/home/mightyraccoon/Downloads/chromedriver") as wd:
        counter = 0
        for query in queries_list:
            logger.info("Parse images for query: %s" % query)
            res = fetch_image_urls(query, dataset_size, wd=wd, sleep_between_interactions=0.5)
            logger.info("Images links processing")
            for elem in res:
                persist_image(data_root, elem, counter)
                counter += 1
            logger.info("Images for query %s are parsed" % query)


if __name__ == "__main__":
    main()
