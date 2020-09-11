# Max Van Gelder
# 7/28/20
# Dataset for working with whale databases and torch
# WARNING: NOT SAFE FOR CONCURRENT MODIFICATION. ENSURE THAT DATABASES ARE NOT BEING MODIFIED WHEN RUNNING THIS PROGRAM

import sqlite3
import torch
from torch.utils.data import Dataset
from ct_data_puller import SongChunk


class WhaleSongDataset(Dataset):
    """Whale Songs Dataset"""

    def __init__(self, bh_db_cursor: sqlite3.Cursor, hb_db_cursor: sqlite3.Cursor,
                 bh_cutoff: float, hb_cutoff: float, transform=None):
        """
        :param bh_db_cursor: Cursor to sqlite3 database containing bowhead song data (see EXPLANATORY_FILE for
         information on database structure)
        :param hb_db_cursor: Cursor to sqlite3 database containing humback song data
        :param bh_cutoff: Bowhead cutoff
        :param hb_cutoff: Humpback cutoff
        :param transform: Optional transform to be applied on a sample.
        """
        self.hb_cutoff = hb_cutoff
        self.bh_cutoff = bh_cutoff
        self.hb_db_cursor = hb_db_cursor
        self.bh_db_cursor = bh_db_cursor
        self.transform = transform

        # Calculate how many chunks there are which meet the random forest prediction probability cutoff for
        # humpback and bowhead respectively
        pull_num_acceptable_chunks = "select count(*) from chunks where RFPredForOne>"
        self.hb_db_cursor.execute(pull_num_acceptable_chunks + str(self.hb_cutoff))
        self.num_hb = self.hb_db_cursor.fetchone()[0]
        self.bh_db_cursor.execute(pull_num_acceptable_chunks + str(self.bh_cutoff))
        self.num_bh = self.bh_db_cursor.fetchone()[0]

        self.record_ids = []
        pull_record_ids = "select RecordID from chunks where RFPredForOne>"
        self.hb_db_cursor.execute(pull_record_ids + str(self.hb_cutoff))
        hb_record_ids = self.hb_db_cursor.fetchall()
        self.record_ids.append(hb_record_ids)
        self.bh_db_cursor.execute(pull_record_ids + str(self.bh_cutoff))
        bh_record_ids = self.bh_db_cursor.fetchall()
        self.record_ids.append(bh_record_ids)

    def __len__(self):
        """
        :return: the number of chunks in the databases which have a random forest prediction probability of greater
        than the cutoffs for belonging to bowhead and humpback
        """
        return self.num_bh + self.num_hb

    def __getitem__(self, idx):
        # 0 = humpback, 1 = bowhead
        if idx < len(self.record_ids[0]):
            chunk = SongChunk.from_database_id(self.record_ids[0][idx][0], self.hb_db_cursor)
            classification = 0
        else:
            chunk = SongChunk.from_database_id(self.record_ids[1][idx - len(self.record_ids[0])][0], self.bh_db_cursor)
            classification = 1

        # Generate the image of the archipelago and convert to a torch tensor
        image = torch.from_numpy(chunk.reconstruct_archipelagos_image(show=False).flatten())
        sample = {'image': image, 'classification': classification}

        if self.transform:
            sample = self.transform(sample)

        return sample
