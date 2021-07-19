from zipfile import ZipFile
import re 
import json
from joblib import delayed, Parallel
import os
import pandas as pd
from icecream import ic


DATA_DIR = "/data/veturilo/"
N_CORES = 6

fname=os.path.join(DATA_DIR,"20190822.zip")


def extract_timestamp(inner_fname):
    try:
        dt = inner_fname[:4] + "-" + inner_fname[4:6] + "-" + inner_fname[6:8] + " " + inner_fname[9:11] + ":" + inner_fname[11:13]+ ":" + inner_fname[13:15]
    except:
        dt =""
    return dt

def list_months(data_dir=DATA_DIR):
    all_files = [f[:6] for f in os.listdir(data_dir) if f.endswith(".zip")]
    all_files = list(set(all_files))
    all_files.sort()
    return all_files


def process_month(mth,output_dir,data_dir):
    files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(".zip") and f.startswith(mth)]
    dfs = Parallel(N_CORES)(delayed(process_zip)(f) for f in files)
    # dfs = [process_zip(f) for f in files]
    data_df = pd.concat([df[0] for df in dfs])
    processing_log = pd.concat([df[1] for df in dfs if df[1] is not None])
    os.makedirs(output_dir,exist_ok=True)
    data_df.to_csv(os.path.join(output_dir,f"{mth}.csv.gz"),compression="gzip",index=False)
    processing_log.to_csv(os.path.join(output_dir,f"{mth}.log"),index=False)
    return True

def extract_json(html_string):
    pattern = re.compile(r".*var NEXTBIKE_PLACES_DB = '(.*)';",re.M)
    return pattern.findall(html_string)[0].replace("Gaulle\\\'a","Gaullea")

def inner_file_wrapper(zip_object,inner_fname):
    dt = extract_timestamp(inner_fname) 
    html_string = zip_object.read(inner_fname).decode('utf-8')
    try:
        json_string = extract_json(html_string)
    except Exception as err:
        return (False,inner_fname,"extract_json",err)
    
    try:
        df = process_json(json_string)
    except Exception as err:
        return (False,inner_fname,"process_json",err,None)
    df = normalize_column_list(df)
    df["dt"] = dt
    return (True,inner_fname,None,None,df)


def process_zip(fname):
    zip_object=ZipFile(fname)
    dfs_list = [inner_file_wrapper(zip_object,f) for f in zip_object.namelist()]
    processing_log = pd.DataFrame([tpl[1:4] for tpl in dfs_list if not tpl[0]],columns=['fname','stage','error'])
    output_data = pd.concat([tpl[4] for tpl in dfs_list if tpl[0]])
    return (output_data,processing_log)

def normalize_column_list(df,expected_cols=["uid","lat","lng","name","number","bikes","bike_racks","free_racks","place_type","bike_numbers"]):
    if(len(expected_cols) == 0):
        return df
    existing_cols = set(df.columns.tolist()).intersection(set(expected_cols))
    df = df[existing_cols]
    return df 


def process_json(json_string,selected_region_name = 'VETURILO Poland'):
    data = json.loads(json_string)

    data = [i for i in data if i["region_info"]["name"] == selected_region_name][0]
    extracted_df = pd.DataFrame.from_dict(data["places"])
    return extracted_df


for m in list_months():
    ic(m)
    process_month(m,"/data/veturilo/csv2",DATA_DIR)

