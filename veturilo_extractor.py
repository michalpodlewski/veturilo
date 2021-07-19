"""
The data are downloaded as zip files for each day of the data collection.
In each zip file we have normally 24*6=144 individual snapshots of veturilo website. 
Aim of the script is to :
-extract a json object from each snapshot, 
-convert it into dataframe,
-concatenate dataframes together into monthly batches
-save the monthly dataframes into separate .csv.gz files 
-and log possible errors in the process

Resulting dataframes have following columns:
uid - station identifier
lat - station geographic latitude (collected for each row)
lng - station geographic longitude (collected for each row)
name - station name
number - station number
bikes - number of available bikes in given time
bike_racks - number of all racks at given station
free_racks - number of free rachs at given station in given time
place_type - type of station 
bike_numbers - array of numbers of bikes available in given station and given time
"""

import json
import os
import re
from zipfile import ZipFile

import pandas as pd
from icecream import ic
from joblib import Parallel, delayed

DATA_DIR = "/data/veturilo/"
N_CORES = 6

fname = os.path.join(DATA_DIR, "20190822.zip")


def extract_timestamp(inner_fname):
    """
    Helper: Extract proper timestamp from filename
    """
    try:
        dt = (
            inner_fname[:4]
            + "-"
            + inner_fname[4:6]
            + "-"
            + inner_fname[6:8]
            + " "
            + inner_fname[9:11]
            + ":"
            + inner_fname[11:13]
            + ":"
            + inner_fname[13:15]
        )
    except:
        dt = ""
    return dt


def list_months(data_dir=DATA_DIR):
    """
    List all distinct months in the source directory

    :param data_dir: Source directory where zip files are stored
    """
    all_files = [f[:6] for f in os.listdir(data_dir) if f.endswith(".zip")]
    all_files = list(set(all_files))
    all_files.sort()
    return all_files


def process_month(mth, output_dir, data_dir):
    """
    Extract data from all zips representing a month passed as first argument 
    and store resulting dataframe in csv.gz in output_dir

    :param mth: Month to be processed in format YYYYMM
    :param output_dir: Where to store resulting DataFrame
    :param data_dir: Where to look for zips

    :return : no meaningful value is returned, data and processing log is written to output dir
    """
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".zip") and f.startswith(mth)
    ]
    dfs = Parallel(N_CORES)(delayed(process_zip)(f) for f in files)
    # dfs = [process_zip(f) for f in files]
    data_df = pd.concat([df[0] for df in dfs])
    processing_log = pd.concat([df[1] for df in dfs if df[1] is not None])
    os.makedirs(output_dir, exist_ok=True)
    data_df.to_csv(
        os.path.join(output_dir, f"{mth}.csv.gz"), compression="gzip", index=False
    )
    processing_log.to_csv(os.path.join(output_dir, f"{mth}.log"), index=False)
    return True


def extract_json(html_string):
    """
    Extract json from html content. The method is super simplistic using simple regex.
    might need rethinking in the future.

    :param html_string: raw HTML as it was saved using wget

    :return : JSON string extracted from HTML
    """
    pattern = re.compile(r".*var NEXTBIKE_PLACES_DB = '(.*)';", re.M)
    return pattern.findall(html_string)[0].replace("Gaulle\\'a", "Gaullea")


def inner_file_wrapper(zip_object, inner_fname):
    """
    Wrapper around extracting the json and creating a dataframe along with collecting 
    possible errors. Handy when used together with list complrehension.

    :param zip_object: Object as returned by ZipFile
    :param inner_fname: Name of file inside the archive

    :return: a tuple of the form: 
    (Boolean indicator of success, 
    name of inner file being processed, 
    stage when failure happened (if processing was not successful, None otherwise), 
    error message (if there was error, None otherwise), 
    resulting DataFrame (if processing was successfull, None otherwise)
    """

    dt = extract_timestamp(inner_fname)
    html_string = zip_object.read(inner_fname).decode("utf-8")
    try:
        json_string = extract_json(html_string)
    except Exception as err:
        return (False, inner_fname, "extract_json", err)

    try:
        df = process_json(json_string)
    except Exception as err:
        return (False, inner_fname, "process_json", err, None)
    df = normalize_column_list(df)
    df["dt"] = dt
    return (True, inner_fname, None, None, df)


def process_zip(fname):
    """
    Processing single zip file (containing up to 144 individual downloaded site snapshots)
    
    :param fname: Path to zip file containing downloaded content

    :return : A tuple of (resulting dataframe for given date, processing log)
    Processing log has 3 columns: Name of input file (one single snapshot), stage when failure 
    happened and error message
    Resulting dataframe has relevant data about situation at all accessible veturilo station 
    with columns:
    ["uid","lat","lng","name","number","bikes","bike_racks","free_racks","place_type",
    "bike_numbers"]
    """
    zip_object = ZipFile(fname)
    dfs_list = [inner_file_wrapper(zip_object, f) for f in zip_object.namelist()]
    processing_log = pd.DataFrame(
        [tpl[1:4] for tpl in dfs_list if not tpl[0]],
        columns=["fname", "stage", "error"],
    )
    output_data = pd.concat([tpl[4] for tpl in dfs_list if tpl[0]])
    return (output_data, processing_log)


def normalize_column_list(
    df,
    expected_cols=[
        "uid",
        "lat",
        "lng",
        "name",
        "number",
        "bikes",
        "bike_racks",
        "free_racks",
        "place_type",
        "bike_numbers",
    ],
):
    """
    Helper function to select from the data frame only the columns that were available throughout whole time of data collection.
    It's convenient because otherwise pd.concat would create some mostly empty columns

    :param df: Dataframe as returned in second element of the process_zip()
    :param expected_cols: Columns expected to be present throughout whole data gathering process

    :return : Column-wise subset of input df
    """
    if len(expected_cols) == 0:
        return df
    existing_cols = set(df.columns.tolist()).intersection(set(expected_cols))
    df = df[existing_cols]
    return df


def process_json(json_string, selected_region_name="VETURILO Poland"):
    """
    Converting extracted JSON into dataframe with information about veturilo system 
    (json object contains also other Polish Nextbike affiliates)

    :param json_string: JSON extracted from downloaded html, as returned by extract_json()
    :param selected_region_name: self-explanatory

    :return : Contents of json for given region name convertend into pd.DataFrame
    """
    data = json.loads(json_string)

    data = [i for i in data if i["region_info"]["name"] == selected_region_name][0]
    extracted_df = pd.DataFrame.from_dict(data["places"])
    return extracted_df


if __name__ == "__main__":
    for m in list_months():
        ic(m)
        process_month(m, "/data/veturilo/csv2", DATA_DIR)
