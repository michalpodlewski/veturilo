"""
Set of functions used for time series predicion, logistics around per-station model fitting etc.
The functions are used in the Veturilo_additional_supply_recommender.ipynb notebook
"""
import numpy as np
import pandas as pd
import veturilo_helper as vh
from icecream import ic


def add_rolling_sum_feature(df, max_h, min_h, rolsum_column="rent_count"):
    """
    Compute sum of selected variable in given rolling window that is offset from 'dt'.
    The period of time is defined by offsets from current time

    :param df: Source (and target) dataframe, columns used: dt (datetime), uid 
    and the column supplied as rolsum_column
    :param max_h: Offset of the begining of the rolling window
    :param min_h: Offset of the end of the rolling window
    :param rolsum_column: Column for which rolling sum will be computed

    :return : Input df with added rolling sum column, 
    naming convention for the resulting column: [rolsum_column]_[max_h]_[min_h]
    """
    max_df = (
        df.groupby("uid")
        .rolling(window=f"{max_h}H", on="dt")[rolsum_column]
        .sum()
        .reset_index()
        .rename(columns={rolsum_column: "rol_max"})
    )
    min_df = (
        df.groupby("uid")
        .rolling(window=f"{min_h}H", on="dt")[rolsum_column]
        .sum()
        .reset_index()
        .rename(columns={rolsum_column: "rol_min"})
    )
    tmp_df = max_df.merge(min_df, on=["uid", "dt"])
    tmp_df[f"{rolsum_column}_{max_h}_{min_h}"] = tmp_df["rol_max"] - tmp_df["rol_min"]
    df = df.merge(
        tmp_df[["uid", "dt", f"{rolsum_column}_{max_h}_{min_h}"]],
        how="left",
        on=["uid", "dt"],
    )

    return df


def extract_features(df, rolsum_column="rent_count"):
    """
    Wrapper function for creating features used in models

    :param df: Source (and target) dataframe, columns used: dt (datetime), uid 
    and the column supplied as rolsum_column
    :param rolsum_column: Column for which rolling sums will be computed.

    :return : dataframe df with computed features
    """

    df.loc[:, "month"] = df.loc[:, "dt"].dt.month
    df.loc[:, "dayofweek"] = df.loc[:, "dt"].dt.dayofweek
    df.loc[:, "hour"] = df.loc[:, "dt"].dt.hour
    df.loc[:, "weeknum"] = df.loc[:, "dt"].dt.week
    df = add_rolling_sum_feature(df, 48, 24, rolsum_column)
    df = add_rolling_sum_feature(df, 25, 24, rolsum_column)
    df = add_rolling_sum_feature(df, 7 * 24, 6 * 24, rolsum_column)
    return df


def create_model(df, features, mdl_fun, tgt="rent_count", **kwargs):
    """
    Function to create and fit a model. 
    Intention is to use this function in list comprehension for all stations
    
    :param df: dataframe to fit the model to. 
    Recommended use is to supply a slice of source dataframe representing one uid
    :param features: List of features to be used in given model
    :param mdl_fun: Model object to be fitted
    :param tgt: Name of target column

    Fitted model object
    """
    mdl = mdl_fun(**kwargs)
    mdl.fit(df[features], df[tgt])
    return mdl


def predict_from_modeldirectory(
    df, model_directory, features_list, destination_column="pred"
):
    """
    Compute predictions using dict of models per each uid and return full dataset with predictions

    Args:
    :param df: DataFrame for which prediction is to be computed. Must contain columns 
    listed in features_list argument
    :param model_directory: Dict with uid as keys and fitted models as values
    :param features_list: List of features used by given model
    :param destination_column: Column name to store the prediction

    :return : Input dataframe with prediction
    NOTE: As it is only a PoC prediction are returned only for uid present in training data
    For production a fallback model must be created and used for missing uids
    """
    output_dfs = []
    for u in df["uid"].unique():
        try:
            tmp_df = df[df["uid"] == u].reset_index()
            tmp_df.loc[:, destination_column] = model_directory[u].predict(
                tmp_df[features_list]
            )
            output_dfs.append(tmp_df)
        except KeyError:
            pass
    return pd.concat(output_dfs)


def add_predictions(df, params_dict):
    """
    Wrapper around predict_from_modeldirectory()

    :param df: DataFrame for which prediction is to be computed.
    :param params_dict : Configuration dict.
    Keys are names of columns to hold the predictions
    values are dict of the form: 
    {
        'model_directory' : Dict with uid as keys and fitted models as values 
        'features_list' : List of features used by models in model_directory dict
    }

    :return: Input dataframe with additional columns holding predictions of models according to params_dict
    """

    for destination_variable, params in params_dict.items():
        df = predict_from_modeldirectory(
            df,
            params["model_directory"],
            params["features_list"],
            destination_column=destination_variable,
        )

    return df


def aggregate_daily_predictions(
    df, columns=["rent_count", "global_prediction", "local_prediction"]
):
    """[summary]
    Aggregation hourly predictions to daily values and computing the difference between global and local prediction
    NOTE: columns "global_prediction" and "local_prediction" are hardcoded but wrapped in try/catch
    Not elegant but does the job
    
    """
    agg_config = {col: "sum" for col in columns}
    daily_counts = df.groupby(["uid", "D"]).agg(agg_config).reset_index()
    try:
        daily_counts["unmet_demand"] = (
            daily_counts["global_prediction"] - daily_counts["local_prediction"]
        )
    except KeyError:
        pass
    return daily_counts
