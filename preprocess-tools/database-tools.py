# Max Van Gelder
# 7/9/20

# Tools for interacting with database created by ct-data-puller.py
# See EXPLANATORY_FILE for information on the database

from ct_data_puller import SongChunk, cfg_default
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
import subprocess
import sqlite3
import time

RAVEN_PRO_PATH = "C:\\Users\\ACS-2\\Raven Pro 1.4\\Raven.exe"


def invoke_raven_pro(audio_file):
    """
    Invokes raven pro on the given audio_file. RavenPro CLI does not work as specified in its documentation
    unfortunately because the argument parser fails on everything.
    :param audio_file: The full path to the audio file to be opened in RavenPro
    :return: None
    """
    subprocess.Popen('"{}" "{}"'.format(RAVEN_PRO_PATH, audio_file), stderr=subprocess.DEVNULL, shell=True)


def get_valid_input(valid_input_dict, prompt):
    """
    Prompts user for a valid input until the user gives one
    :param valid_input_dict: A dictionary with valid inputs as keys, and descriptions of those inputs as values. EX:
    {'1': "Select Chicken", '2': "Select Pig"}
    :param prompt: The prompt which the user will be provided to ask for their input
    :return: The valid user input once provided
    """
    user_input = input(prompt)
    while user_input not in valid_input_dict.keys():
        print("Valid inputs are: ")
        for k in valid_input_dict.keys():
            print("\t '{}': {}".format(k, valid_input_dict[k]))
        user_input = input(prompt)
    return user_input


def manual_database_classify(db_connection, record_id):
    """
    Displays the record ID of a spectrogram for the viewer to classify
    :param db_connection: The connection to the SQLite3 Database containing the information
    about the portion of whale song which will be manually classified
    :param record_id: The RecordID field of the portion of the whale song which will be manually classified
    in the database
    :return: True if a new manual classification for record_id was added to the
    database and there was no preexisting classification. False if there was a preexisting classification for record_id
    in the database already.
    """
    # Check if the entry with RecordID == record_id already has a classification
    db_connection.execute("SELECT Label FROM chunks WHERE RecordID={}".format(record_id))
    preexisting_classification = db_connection.fetchone()[0]
    if preexisting_classification is not None:
        return False

    # The chunk hasn't yet been classified in the database. Load object and display archipelagos
    chunk = SongChunk.from_database_id(record_id, db_connection)
    chunk.display_archipelagos()

    # Let user know where the clip starts in the full spectrogram
    db_connection.execute("SELECT SpecPath from chunks where RecordID={}".format(record_id))
    pos_in_record = int(db_connection.fetchone()[0].split('.')[0].split('-')[-1])
    print("Clip starts at {}".format(time.strftime('%H:%M:%S', time.gmtime(cfg_default.chunk_len * pos_in_record))))

    # Get user input as to what the classification should be
    # 0 = Definitely no whale song in the chunk
    # 1 =  Definitely whale song in the chunk
    # 2 = Unsure, experienced opinion needed for checking
    accepted_inputs = {'0': "No whale song", '1': "Whale song", '2': "Uncertain"}
    manual_classification = get_valid_input(accepted_inputs, "RID={} What do you classify this chunk?: ".format(record_id))
    # Show full spectrogram if unsure
    if manual_classification == '2':
        print("Fetching RavenPro spectrogram")

        db_connection.execute("SELECT ParentRecording from chunks where RecordID={}".format(record_id))
        parent_record = int(db_connection.fetchone()[0])
        db_connection.execute("SELECT Path from recordings where FileID={}".format(parent_record))
        invoke_raven_pro(db_connection.fetchone()[0])

        manual_classification = get_valid_input(accepted_inputs, "What is your opinion after looking at the RavenPro: ")

    # Put the user's classification into the database
    db_connection.execute("update chunks set Label={} where RecordID={}".format(manual_classification,
                                                                                         record_id))
    return True


if __name__ == "__main__":
    # connect to database
    # Get handle for working with database
    conn = sqlite3.connect('new_data.db')
    c = conn.cursor()

    # Give images for the user to manually classify
    for i in range(310, 500):
        manual_database_classify(c, i + 1)
        conn.commit()

    # Random forest classifier
    # Get all of the data which has been classified by hand and load it into a numpy array
    c.execute("SELECT NumACP, AvgACPSize FROM chunks WHERE Label is not null")
    data = c.fetchall()
    X = np.ndarray([len(data), len(data[0])])
    for datum_idx, datum in enumerate(data):
        for sub_idx in range(len(datum)):
            X[datum_idx][sub_idx] = datum[sub_idx]

    # Get all of the hand classifications
    c.execute("SELECT Label FROM chunks WHERE Label is not null")
    labels = c.fetchall()
    y = np.ndarray([len(labels)])
    for c_idx, classification in enumerate(labels):
        y[c_idx] = classification[0]

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)


    # Analyize performance
    ps = clf.predict_proba(X)
    cutoffs = [.05 * i for i in range(21)]

    res = []
    total_positives = []

    for cutoff in cutoffs:
        tp = 0
        fp = 0
        tot_pos = 0
        for p, l in zip(ps, labels):
            if p[1] >= cutoff:
                tot_pos += 1
                if l[0] == 1:
                    tp += 1
                else:
                    fp += 1
        if tp + fp != 0:
            print("With cutoff {} we achieve a TP/FP of {}".format(cutoff, float(tp) / (tp + fp)))
            res.append((cutoff, float(tp) / (tp + fp)))
            total_positives.append(tot_pos)
        else:
            print("With cutoff {} nothing was classified as positive".format(cutoff))



    print([x[0] for x in res])
    print([x[1] for x in res])
    print(total_positives)


    # # See how what it fits to some random data
    # for i in range(15):
    #     random_selection = random.randint(500, 750)
    #     manual_database_classify(c, random_selection)
    #     c.execute("SELECT NumACP, AvgACPSize FROM chunks WHERE RecordID={}".format(random_selection))
    #     d = c.fetchone()
    #     print("The random forest classifier chose {}".format(clf.predict([[d[0], d[1]]])))
    #     conn.commit()


    # Commit and close DB connection
    conn.commit()
    conn.close()

