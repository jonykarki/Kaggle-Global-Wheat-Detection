import logging
import os, sys

def init_logger(name, out_dir, process_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ADD FOR PROCESS_RANK
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if out_dir:
        fh = logging.FileHandler(os.path.join(out_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger