# Max Van Gelder
# 7/9/20

# Tools for interacting with database created by ct-data-puller.py
# See EXPLANATORY_FILE for information on the database

from ct_data_puller import SongChunk, cfg_default
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
    subprocess.call('"{}" "{}"'.format(RAVEN_PRO_PATH, audio_file))


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
    db_connection.execute("SELECT Classification FROM chunks WHERE RecordID={}".format(record_id))
    preexisting_classification = db_connection.fetchone()[0]
    if preexisting_classification is not None:
        return False

    # The chunk hasn't yet been classified in the database. Load object and display archipelagos
    chunk = SongChunk.from_database_id(record_id, db_connection)
    chunk.display_archipelagos()

    # Get user input as to what the classification should be
    # 0 = Definitely no whale song in the chunk
    # 1 =  Definitely whale song in the chunk
    # 2 = Unsure, experienced opinion needed for checking
    accepted_inputs = {'0': "No whale song", '1': "Whale song", '2': "Uncertain"}
    manual_classification = get_valid_input(accepted_inputs, "What do you classify this chunk?: ")
    # Show full spectrogram if unsure
    if manual_classification == '2':
        print("Fetching RavenPro spectrogram")

        # Let user know where the clip starts in the full spectrogram
        db_connection.execute("SELECT SpecPath from chunks where RecordID={}".format(record_id))
        pos_in_record = int(db_connection.fetchone()[0].split('.')[0][-1])
        print("Clip starts at {}".format(time.strftime('%H:%M:%S', time.gmtime(cfg_default.chunk_len * pos_in_record))))
        time.sleep(1)

        db_connection.execute("SELECT ParentRecording from chunks where RecordID={}".format(record_id))
        parent_record = int(db_connection.fetchone()[0])
        db_connection.execute("SELECT Path from recordings where FileID={}".format(parent_record))
        invoke_raven_pro(db_connection.fetchone()[0])

        manual_classification = get_valid_input(accepted_inputs, "What is your opinion after looking at the RavenPro: ")

    # Put the user's classification into the database
    db_connection.execute("update chunks set Classification={} where RecordID={}".format(manual_classification,
                                                                                         record_id))
    return True


if __name__ == "__main__":
    # connect to database
    # Get handle for working with database
    conn = sqlite3.connect('new_data.db')
    c = conn.cursor()
    for i in range(5):
        manual_database_classify(c, i + 1)
        conn.commit()

    print("Success?")

    # Commit and close DB connection
    conn.commit()
    conn.close()

