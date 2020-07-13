# Max Van Gelder
# 7/9/20

# Tools for interacting with database created by ct-data-puller.py
# See EXPLANATORY_FILE for information on the database

from ct_data_puller import SongChunk
import sqlite3
import sys


def regenerate_from_archipelagos():
    """
    Regenerates the archipelago rendering from a chunk using ONLY the information in the database
    :return:
    """


if __name__ == "__main__":
    # connect to database
    # Get handle for working with database
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    # perform kmeans on chunks
    s = SongChunk.from_database_id(2, c)

    print("Success?")

