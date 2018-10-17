""" Logging initialization for use in notebooks (%run -m cytokit.nb.logging) """
import logging
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console)
