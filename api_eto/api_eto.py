import argparse

from etoLib.log_logger import log_make_logger

from etoLib.s3_func import s3_hello

from etoLib.util_func import unique
from etoLib.util_func import grepfxn

def get_parser():
    parser = argparse.ArgumentParser(description='Run the eto code')
    parser.add_argument('tile', metavar='TILE', type=str, nargs='*',
            help='the tile to process - example: 40N-80E')
    parser.add_argument('-c', '--configdir', help='specify and alternate config_dict dir example: -c sample_config ', default='./sample_config', type=str)
    parser.add_argument('-o', '--optimize', help='optimize caching on ', default='yes', type=str)
    return parser


def command_line_runner():
    parser = get_parser()
    args = vars(parser.parse_args())

    if args['configdir']:
        print("configdir", args['configdir'])

    optimize = False
    opt = args['optimize']

    config_directory = args['configdir']

    log.info('USing configdir {}'.format(config_directory))

    log.info('this is just a starter kit for our cmdline api for eto - Help Greg!')
    log.info('or logging agents and logging backends ... docker deployments')

    # RUN the class Veget
    #myveg = VegET(config_directory, tile, shp, optimize)
    #myveg.run_veg_et()

    log.info('this is how you call one of your functions')

    s3_hello('Greg')


if __name__ == '__main__':
    log = log_make_logger('THE_ETO_CREATOR')
    command_line_runner()
