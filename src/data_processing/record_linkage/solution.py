from functools import reduce

import numpy as np


def preprocessing(df_youtube, df_spotify):
    for df in [df_youtube, df_spotify]:
        df["owner"] = (df["first_name"] + " " + df["last_name"]).str.lower()

        df["first_three_octets_ip"] = df["non_mfa_ip_addresses"].map(
            lambda ip_list: [".".join(ip.split(".")[:-1]) for ip in ip_list]
        )
    return df_youtube, df_spotify


def link_records(df_youtube, df_spotify):
    df_youtube, df_spotify = preprocessing(df_youtube, df_spotify)

    matched_on_contact = match_on_preferred_contact(df_youtube, df_spotify)
    matched_name_zip_and_digits, matched_name_zip_and_ip = (
        match_on_name_zip_and_digits_ip(df_youtube, df_spotify)
    )

    all_matches = reduce(
        np.union1d,
        [matched_on_contact, matched_name_zip_and_digits, matched_name_zip_and_ip],
    )
    return df_youtube.iloc[all_matches]


def match_on_preferred_contact(df_youtube, df_spotify):
    return np.argwhere(
        np.isin(df_spotify["preferred_contact"], df_youtube["preferred_contact"])
    ).ravel()


def match_on_name_zip_and_digits_ip(df_youtube, df_spotify):
    """Match on name, zip and last four digits AND on name, zip and three octets"""
    matched_name = np.argwhere(
        np.isin(df_spotify["owner"], df_youtube["owner"])
    ).ravel()
    matched_zip = np.argwhere(
        np.isin(df_spotify["billing_zip_code"], df_youtube["billing_zip_code"])
    ).ravel()
    matched_last_four = np.argwhere(
        np.isin(df_spotify["last_four_digits"], df_youtube["last_four_digits"])
    ).ravel()
    matched_three_octets = np.argwhere(
        np.any(
            np.isin(
                [ip for ip in df_spotify["first_three_octets_ip"]],
                [ip for ip in df_youtube["first_three_octets_ip"]],
            ),
            axis=1,
        )
    ).ravel()
    matched_name_zip_and_last_four = reduce(
        np.union1d, [matched_name, matched_zip, matched_last_four]
    )
    matched_name_zip_and_ip = reduce(
        np.union1d, [matched_name, matched_zip, matched_three_octets]
    )

    return matched_name_zip_and_last_four, matched_name_zip_and_ip
