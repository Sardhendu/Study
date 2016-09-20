# coding: utf-8

try :
    import ConfigParser as configparser
except ImportError:
    import configparser
import os
import json




# Make your config-changes here for running an application solely, which is basically for debugging
def get_config_settings():                  
    dir_name = os.path.dirname(os.path.abspath(__file__))
    conf_name = 'config.conf'
    dir_name = os.path.join(dir_name, conf_name)
    conf_dict = parse_config(dir_name)
    return conf_dict


def xget(func, section, option, default=None):
    try:
        return func(section, option)
    except:
        return default


def parse_config(filename=None):
    cfg = configparser.RawConfigParser()
    with open(filename) as fp:
        cfg.readfp(fp)

    config_settings = {}
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))    
    getpath = lambda k, v: os.path.join(root_dir, xget(cfg.get, k, v))


    # Talking Data User's Demographics
    config_settings["iris_data"] = getpath("DataMining", "iris_data")
    config_settings["titanic_data"] = getpath("DataMining", "titanic_data")
    config_settings["voting_data"] = getpath("DataMining", "voting_data")
    config_settings["auto_mpg_data"] = getpath("DataMining", "auto_mpg_data")

    return config_settings


# print get_config_settings()