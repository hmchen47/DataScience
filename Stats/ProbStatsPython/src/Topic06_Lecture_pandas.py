#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv_file(fname):
    """loading given file name and return a dataframe

    Arguments:
        fname {str} -- path/file name to open
    """

    return pd.read_csv(fname)


def main():

    # load files
    df_heroes_info = load_csv_file("./data/heroes_information.csv")
    df_heroes_powers = load_csv_file("./data/super_hero_powers.csv")

    print("\ndf_heroes_info: \n{}".format(df_heroes_info.head()))
    print("\ndf_heroes_powers: \n{}".format(df_heroes_powers.head()))

    input("\nPress Enter to continue ...")

    # querying the dataset
    print("\ndf_heros_info['name']: \n{}".format(df_heroes_info['name']))
    print("\ntype(df_heroes_info['name']: {}".format(type(df_heroes_info['name'])))

    input("\nPress Enter to continue ...")


    # columns() to access all the available possible powers
    list_powers = [power for power in df_heroes_powers.columns][1:]
    print("\nList of powers:")
    for powers in list_powers:
        print("  ", powers)

    input("\nPress Enter to continue ...")


    # counting the number of records in a column
    print("\ndf_heroes_info['Alignment'].value_counts(): \n{}".format(df_heroes_info['Alignment'].value_counts()))

    input("\nPress Enter to continue ...")

    # find the 5 most common powers and all unique powers in dataset
    num_powers = df_heroes_powers.sum(axis=0)[1:] #+ np.zeros(len(list_powers)-1)

    common_pow = num_powers.sort_values(ascending=False)[0:5].index
    print("\n5 most common powers:")
    for power in common_pow:
        print("  ", power)

    input("\nPress Enter to continue ...")

    # all the unique power
    unique_pow = num_powers.sort_values(ascending=True)[num_powers.sort_values(ascending=True)==1].index
    print("\nList of unique power:")
    for power in unique_pow:
        print("  ", power)

    input("\nPress Enter to continue ...")

    # more complex query
    pub_power_data = pd.merge(df_heroes_info[['name','Publisher']],df_heroes_powers,left_on='name',\
        right_on='hero_names',left_index=True)
    grouped = pub_power_data.groupby("Publisher")
    print("\ngrouped.sum().sum(axis=1): \n\n{}".format(grouped.sum().sum(axis=1)))


    return None


if __name__ == "__main__":

    print("\nEntering Topic 6 Pandas python code ...")

    main()

    print("\nEnd Topic 6 Pandas python code ...\n")


