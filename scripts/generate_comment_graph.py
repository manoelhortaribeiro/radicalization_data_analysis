from pandas.errors import EmptyDataError
from sqlitedict import SqliteDict
import pandas as pd
import datetime
import swifter


def get_author_ids(comments_raw):
    author_ids = []
    timestamps = []

    if type(comments_raw) != str:
        return author_ids, timestamps
    else:
        comments_video = eval(comments_raw.replace("\0", ""))

    for comment in comments_video:
        try:
            author_ids.append(comment["authorLink"])
            timestamps.append(comment["timestamp"])
        except KeyError:
            pass

        if comment["hasReplies"] is False:
            continue

        for reply in comment["replies"]:
            try:
                author_ids.append(reply["authorLink"])
                timestamps.append(reply["timestamp"])
            except KeyError:
                pass

    return author_ids, timestamps


src_csv = "/home/manoelribeiro/PycharmProjects/radicalization_data_collection/data/youtube/sources.csv"
src = "./data/cm/"
dst = "./data/authors_dict_2.sqlite"

df_src = pd.read_csv(src_csv)

count = 0

author_dict = SqliteDict(dst)
for channel_id, category in zip(df_src["Id"], df_src["Category"]):

    now = datetime.datetime.now()
    print(channel_id)
    try:
        for chunk in pd.read_csv(src + channel_id + ".csv.gz", chunksize=1000, compression='gzip'):
            count += 1
            for author_list, timestamp_list in chunk['comments'].swifter.apply(get_author_ids).values:

                for author, timestamp in zip(author_list, timestamp_list):
                    dict_val = {"timestamp": timestamp, "channel_id": channel_id}
                    val = author_dict.get(author, [])
                    val.append(dict_val)
                    author_dict[author] = val
        author_dict.commit()

    except EmptyDataError:
        print("EmptyDataError:", channel_id)
        pass

    except FileNotFoundError:
        print("FileNotFoundError:", channel_id)
        pass

    print(datetime.datetime.now() - now)
author_dict.commit()
author_dict.close()
