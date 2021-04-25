import logging
import os


def set_logger(save_path, print_on_screen):
    '''
    Write logs to console and log file

    Parameters
    ----------
    save_path : str, path to save the log
    print_on_screen : bool, print log info on the console or not
    '''
    log_file = os.path.join(save_path, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)