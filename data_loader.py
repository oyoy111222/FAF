import os
import pandas as pd
import glob


def load_electricity_data(dataset_config):
    """
    加载电力数据集
    """
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]

    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        df = pd.read_excel(file, sheet_name=None) 
        for sheet_name, sheet_data in df.items():
            filename = os.path.basename(file) 
            year = filename.split('_')[0]  # '2018'
            sheet_name_with_year = f"{year}_{sheet_name}"  
            sheet_data = sheet_data[[time_column, target_column]]
            data.append((sheet_name_with_year, sheet_data))
    return data


def load_temperature_data(dataset_config):
    """
    加载温度数据集
    """
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]

    folders = glob.glob(os.path.join(data_directory, '*'))
    for folder in folders:
        if os.path.isdir(folder):
            files = glob.glob(os.path.join(folder, '*' + file_pattern))  
            for file in files:
                df = pd.read_csv(file)
                df = df[[time_column, target_column]]  
                city_name = os.path.basename(file).split('.')[0]  
                data.append((city_name, df)) 
    return data


def load_walmart_data(dataset_config):
    """
    加载Walmart销售数据集
    """
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]
    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        #df = pd.read_csv(file)
        df = pd.read_excel(file)
        df = df[[time_column, target_column]]  
        store_name = os.path.basename(file).split('.')[0]
        data.append((store_name, df)) 
    return data

def load_co2emission_data(dataset_config):
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]
    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        df = pd.read_excel(file)
        df = df[[time_column, target_column]]  
        store_name = os.path.basename(file).split('.')[0]  
        data.append((store_name, df)) 
    return data

def load_gdp_data(dataset_config):
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]
    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        df = pd.read_excel(file)
        df = df[[time_column, target_column]]  
        store_name = os.path.basename(file).split('.')[0]  
        data.append((store_name, df))  
    return data

def load_weather_data(dataset_config):
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]
    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        df = pd.read_excel(file)
        df = df[[time_column, target_column]]  
        store_name = os.path.basename(file).split('.')[0]  
        data.append((store_name, df))  
    return data

def load_indian_data(dataset_config):
    data = []
    data_directory = dataset_config["data_directory"]
    file_pattern = dataset_config["file_pattern"]
    time_column = dataset_config["time_column"]
    target_column = dataset_config["target_column"]
    files = glob.glob(os.path.join(data_directory, '*' + file_pattern))
    for file in files:
        df = pd.read_excel(file)
        df = df[[time_column, target_column]]  
        store_name = os.path.basename(file).split('.')[0]  
        data.append((store_name, df))  
    return data
