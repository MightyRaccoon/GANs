import os
import logging
import requests
import time
from PIL import Image
import io

import torch
import torch.nn as nn
from selenium import webdriver


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level="INFO"
)


def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: float = 1.0):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)



    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        logger.info("Found: %i search results. Extracting links from %i:%i" % (number_results, results_start, number_results))

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector("img.n3VNCb")
            for actual_image in actual_images:
                if actual_image.get_attribute("src") and "http" in actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                logger.info("Found: %i image links, done!" % len(image_urls))
                break
        else:
            logger.info("Found: %i image links, looking for more ..." % len(image_urls))
            time.sleep(30)
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path: str, url: str, counter: int):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        logger.error("ERROR - Could not download %s - %s" % (url, e))

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(folder_path, str(counter) + ".jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
        logger.info("SUCCESS - saved %s - as %s" % (url, file_path))
    except Exception as e:
        print("ERROR - Could not save %s - %s" % (url, e))


