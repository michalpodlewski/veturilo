"""
Helper functions for analysing the data of Veturilo bike sharing system.
Note: Functions are heavily context specific

"""
import itertools
import os
import re

import pandas as pd
from icecream import ic

DATA_DIR = "/data/veturilo/processed_csv/"


def generate_fname(year, month, data_dir=DATA_DIR):
    return f"{data_dir}/{year}{str(month).zfill(2)}.csv.gz"


def list_files():
    return os.listdir(DATA_DIR)


def read_data(fname=None, year=None, month=None):
    """
    Function to read monthly batch of data

    :param fname: Name of csv.gz file to read data from
    :param year: if fname is not given it will be 
    constructed based on year and month
    :param month: if fname is not given it will be 
    constructed based on year and month

    :returns : pd.DataFrame with all available data for given month
    Collected columns are:
    """
    if fname is None and year is not None and month is not None:
        fname = generate_fname(year, month)
    elif type(fname) is int and year < 13:
        fname = generate_fname(fname, year)

    try:
        df = pd.read_csv(fname, low_memory=False)
    except FileNotFoundError:
        fname = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fname, low_memory=False)

    df["bike_numbers"] = df["bike_numbers"].str.split(",")

    df.loc[df["bike_numbers"].isnull(), "bike_numbers"] = pd.Series(
        [[]] * df["bike_numbers"].isnull().sum()
    ).values

    df["dt"] = pd.to_datetime(df["dt"])

    df["bikes"] = pd.to_numeric(df["bikes"], errors="coerce")
    df["free_racks"] = pd.to_numeric(df["free_racks"], errors="coerce")
    df["bike_racks"] = pd.to_numeric(df["bike_racks"], errors="coerce")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def bike_station_pairs(df):
    """
    Function to transform original dataframe into triplets:
    station uid | bike number | timestamp

    :param df: Dataframe as produced by data transformer.
    Mandatory columns: 
    dt - time
    uid - station identifier
    bike_numbers - array of bike identifiers, will be converted to numbers
    
    :return : pd.DataFrame of form uid | bike | dt
    uid == -1 in returned dataframe denotes situation, that given bike 
    is not recorded at any existing station 
    """
    pad_df = df[["uid", "dt"]].drop_duplicates()
    pairs_df = (
        df[["uid", "dt", "bike_numbers"]]
        .explode("bike_numbers")
        .rename(columns={"bike_numbers": "bike"})
    )
    pad_df = pd.DataFrame(
        itertools.product(pairs_df["dt"].unique(), pairs_df["bike"].unique()),
        columns=["dt", "bike"],
    )
    pairs_df = pad_df.merge(pairs_df, on=["dt", "bike"], how="left")
    pairs_df["bike"] = pd.to_numeric(pairs_df["bike"], errors="coerce")
    pairs_df["uid"] = pairs_df["uid"].fillna(-1)
    pairs_df = pairs_df.sort_values(["bike", "dt"])
    return pairs_df


def _bike_station_pairs_virtual(df):
    """
    Special variation of bike_station_pairs creating "virtual" bikes (unique 
    identifiers created for situation, where there is no bike on the station).
    It is useful to identify rentals (advantage over vanilla bike_station_pairs
    is that it automatically handles 0 rentals when there were no real bike 
    in the station)
    
    TODO: merge it into one function with bike_station_pairs()
    """

    df["len_bikes_array"] = df["bike_numbers"].apply(lambda x: len(x))
    df.loc[df["len_bikes_array"] == 0, "bike_numbers"] = df.loc[
        df["len_bikes_array"] == 0
    ].apply(lambda x: -1 * x[df["len_bikes_array"] == 0].index)

    pairs_df = (
        df[["uid", "dt", "bike_numbers"]]
        .explode("bike_numbers")
        .rename(columns={"bike_numbers": "bike"})
    )
    pairs_df["bike"] = pd.to_numeric(
        pairs_df["bike"], errors="coerce"
    )  # there are some erroneus values in source data
    pairs_df = pairs_df[~pairs_df["bike"].isnull()]

    pad_df = pd.DataFrame(
        itertools.product(
            pairs_df["dt"].unique(), pairs_df.loc[pairs_df["bike"] > 0, "bike"].unique()
        ),
        columns=["dt", "bike"],
    )
    pairs_df = pad_df.merge(pairs_df, on=["dt", "bike"], how="outer")
    pairs_df["uid"] = pairs_df["uid"].fillna(-1)
    pairs_df = pairs_df.sort_values(["bike", "dt"])
    return pairs_df


def prepare_hourly_rentals(df):
    """
    Apply condition of bike-station pairs dataframe to identify situation 
    when a bike was rented.
    
    :param df: Source dataframe (as created by crawler)
    
    :return : pd.DataFrame with columns 
    uid - uid of a station
    dt - observation time
    rental_count - number of rentals recorded for this station and time (being 
    number of bikes rented between given dt and next observation, usually 10 minutes)
    """
    pairs_df = _bike_station_pairs_virtual(df)
    pairs_df = pairs_df[(pairs_df["bike"] != "?") & (~pairs_df["bike"].isnull())]
    pairs_df["bike"] = pairs_df["bike"].astype(float)

    pairs_df = pairs_df.sort_values(["bike", "dt"])
    pairs_df["next_uid"] = pairs_df.groupby("bike")["uid"].shift(-1)
    pairs_df["rental"] = (
        (pairs_df["uid"] != pairs_df["next_uid"])
        & (~pairs_df["next_uid"].isnull())
        & (pairs_df["uid"] != -1)
    ).astype(int)
    pairs_df["dt"] = pairs_df["dt"].dt.floor("H")
    return (
        pairs_df.groupby(["dt", "uid"])["rental"]
        .sum()
        .reset_index()
        .rename(columns={"rental": "rent_count"})
    )


def get_hourly_rentals_df(recompute=False, filename="hourly_rentals.pkl"):
    """
    Wrapper applying prepare_hourly_rentals() to all collected data 
    in monthly batches.
    If the data has already been computed it is read from pickle unless 
    explicitly forced otherwise

    :param recompute: Should the data be computed
    """
    if recompute or not os.path.exists(filename):
        hourly_rentals = [
            prepare_hourly_rentals(read_data(fname=f)) for f in list_files()
        ]
        hourly_rentals = pd.concat(hourly_rentals)
        hourly_rentals.to_pickle(filename)
        return hourly_rentals
    else:
        return pd.read_pickle(filename)


def get_hourly_available_bikes(recompute=False, filename="hourly_available_bikes.pkl"):
    """
    Wrapper applying aggregation of number of available bikes and racks to all collected data 
    in monthly batches.
    If the data has already been computed it is read from pickle unless 
    explicitly forced otherwise
    """
    if recompute or not os.path.exists(filename):
        dfs = []
        for f in list_files():
            ic(f)
            df = read_data(f)
            df["dt"] = df["dt"].dt.floor("H")
            df = (
                df.groupby(["uid", "dt"])
                .agg({"bike_racks": "first", "bikes": lambda x: round(np.mean(x), 0)})
                .reset_index()
            )
            dfs.append(df)
        dfs = pd.cponcat(dfs)
        dfs.to_pickle(filename)
        return dfs
    else:
        return pd.read_pickle(filename)
