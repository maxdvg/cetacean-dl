# Max Van Gelder
# 7/9/20

# Tools for interacting with database created by ct-data-puller.py
# See EXPLANATORY_FILE for information on the database

import argparse
from ct_data_puller import SongChunk, cfg_default
import numpy as np
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
    :raises Exception: If record_id is negative raises exception, because the unique ID in a sqlite database is
    non-negative
    :return: True if a new manual classification for record_id was added to the
    database and there was no preexisting classification. False if there was a preexisting classification for record_id
    in the database already.
    """
    if record_id <= 0:
        raise Exception("record_id must be positive")

    # Check if the entry with RecordID == record_id already has a classification
    db_connection.execute("SELECT Label FROM chunks WHERE RecordID={}".format(record_id))
    preexisting_classification = db_connection.fetchone()[0]
    if preexisting_classification is not None:
        return False

    # # The chunk hasn't yet been classified in the database. Load object and display archipelagos
    # chunk = SongChunk.from_database_id(record_id, db_connection)
    # chunk.reconstruct_archipelagos_image()

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


def feature_list_to_string(feat_list):
    """
    Turns a list of strings into a comma seperated string. Intended to make working with feature column names easier
    by providing a conversion between Python lists and SQLite command format
    :param feat_list: A list of strings. E.G. ["a", "b", "x"]
    :raises Exception: If feat_list is empty
    :return: A comma seperated string of all the values in 'feat_list'; E.G. "a, b, x" for the example feat_list
    """
    if not feat_list:
        raise Exception("Your feature list cannot be empty")

    ret_str = feat_list[0]
    for col_name in feat_list[1:]:
        ret_str += ", {}".format(col_name)
    return ret_str


def get_random_forest(table: str, db_cursor, desired_tp_rate: float, col_names, verbose=False):
    """
    Create a random forest classifier on the database using all of the data which has been hand labeled
    :param db_cursor: Connection to the database which has the
    :param table: The table from which to read in the data to train the random forest classifier
    :param desired_tp_rate: The desired rate of ((True Positives) / (True Positives + False Positives)_
    :param col_names: A list of names of the columns which we want to use to train the random forest classifier on.
     E.G., if we want to use AvgACPSize and NumACP only to train the classifier, then we pass ["AvgACPSize", "NumACP"]
    :raises Exception: If there is no cutoff such that the random forest achieves the desired true positive rate, then
    raises an Exception
    :return: A tuple containing:
        1) A SKLearn random forest classifier that has learned on the parameters specified in 'col_names' from the
    table 'table'
        2) The lowest 'cutoff' for probabilities in the random forest which gives the desired_tp_rate
    """
    # Get all of the data which has been classified by hand from the database
    search_cmd = "SELECT {} FROM {} WHERE Label is not null".format(feature_list_to_string(col_names), table)
    db_cursor.execute(search_cmd)
    labeled_data = db_cursor.fetchall()

    # Load all of that data into a numpy array for compatibility with SKLearn random forest classifier
    X = np.ndarray([len(labeled_data), len(labeled_data[0])])
    for datum_idx, datum in enumerate(labeled_data):
        for sub_idx in range(len(datum)):
            X[datum_idx][sub_idx] = datum[sub_idx]

    # Get all labels and load them into a seperate numpy array for the SKLearn random forest classifier
    db_cursor.execute("SELECT Label FROM chunks WHERE Label is not null")
    labels = db_cursor.fetchall()
    y = np.ndarray([len(labels)])
    for c_idx, classification in enumerate(labels):
        y[c_idx] = classification[0]

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    # Analyize performance and find cutoff
    ps = clf.predict_proba(X)
    optimal_cutoff = None
    candidate_cutoffs = [.05 * i for i in range(21)]

    # TODO: Use numpy's powerful features!
    for cand_cutoff in candidate_cutoffs:
        true_positives = 0
        false_positives = 0
        for probability, label in zip(ps, labels):
            if probability[1] >= cand_cutoff:
                if label[0] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
        if true_positives + false_positives != 0:
            tp_rate = float(true_positives) / (true_positives + false_positives)
            if verbose:
                print("Cutoff {} yields true positive rate of {}".format(cand_cutoff, tp_rate))
            if tp_rate > desired_tp_rate and optimal_cutoff is None:
                optimal_cutoff = cand_cutoff

    # Fail if the desired true positive rate cannot be reached
    if optimal_cutoff is None:
        raise Exception("Random forest cannot yield a true positive rate as high as {}".format(desired_tp_rate))

    return clf, optimal_cutoff


def predict_for_entire_database(table, db_cursor, db_connection, rf_classifier, prediction_col, col_names):
    """
    Use the random forest classifier 'rf_classifier' to make label predictions for everything in the database pointed
    to by 'db_cursor'. Stores predictions in the 'prediction_col' of the database
    :param table: The name of the table in the database which will be classified
    :param db_cursor: Cursor to the database
    :param db_connection: Connection to the database
    :param rf_classifier: The random forest classifier which can make probabilistic predictions about the data in
    the database
    :param prediction_col: The name of the column which the prediction should be stored in in 'table'
    :param col_names: A list containing the names of the features which are used by 'rf_classifier' to make a
    classification
    :return: None
    """
    # WARNING: THIS COULD BREAK FOR VERY LARGE TABLES IF MEMORY RUNS OUT
    # Extract all of the useful features from the desired database table
    db_cursor.execute("SELECT {} FROM chunks".format(feature_list_to_string(col_names)))
    data = db_cursor.fetchall()

    # Load the features into a numpy array that the random forest accepts
    features = np.empty([len(data), len(data[0])])
    for datum_idx, datum in enumerate(data):
        for feat_idx, feature in enumerate(datum):
            features[datum_idx][feat_idx] = feature

    # Get the prediction probabilities for each of the rows in the table and store it in the prediction column
    rf_pred_probs = rf_classifier.predict_proba(features)
    for rf_idx, rf_pred in enumerate(rf_pred_probs):
        db_cursor.execute("UPDATE {} SET {}={} WHERE rowid={}".format(table, prediction_col, rf_pred[1], rf_idx))
        db_connection.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", help="The .db for the song database")
    sys_args = parser.parse_args()

    # connect to database
    # Get handle for working with database
    conn = sqlite3.connect(sys_args.db)
    c = conn.cursor()

    # Get some manual classifications before generating a predictor
    for i in range(1, 300):
        manual_database_classify(c, i)
        conn.commit()

    # Generate predictions
    feature_columns = ["NumACP", "AvgACPSize", "AvgACPDensity"]

    random_forest, cutoff = get_random_forest("chunks", c, .95, feature_columns, verbose=True)
    predict_for_entire_database("chunks", c, conn, random_forest, "RFPredForOne", feature_columns)

    # Close DB connection
    conn.commit()
    conn.close()
