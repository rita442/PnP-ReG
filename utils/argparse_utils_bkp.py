import argparse
import os

########################################       Parameters Check Functions       ########################################

#strictly positive numeric value
def check_pos_numeric(value):
    try:
        ivalue = float(value)
        if ivalue <= 0: raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a strictly positive numeric value.')
    return ivalue

#non-negative numeric value
def check_nneg_numeric(value):
    try:
        fvalue = float(value)
        if fvalue < 0: raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) numeric value.')
    return fvalue

#strictly integer
def check_pos_int(value):
    try:
        ivalue = int(value)
        if ivalue <= 0: raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a strictly positive int value.')
    return ivalue

#non-negative integer
def check_nneg_int(value):
    try:
        ivalue = int(value)
        if ivalue < 0: raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) int value.')
    return ivalue

#non-negative integer, and zero is replaced by infinity
def check_nneg_int_zinf(value):
    try:
        ivalue = int(value)
        if ivalue < 0: raise Exception
    except:
        raise argparse.ArgumentTypeError(f'{value} is not a positive (or null) int value.')
    if ivalue == 0:
        return float('inf')
    else:
        return ivalue

#name of an existing file
def check_isfile(value):
    realpath = os.path.realpath(value)
    if not os.path.isfile(realpath):
        raise argparse.ArgumentTypeError(f'Can\'t find the file: \'{realpath}\'')
    return realpath



