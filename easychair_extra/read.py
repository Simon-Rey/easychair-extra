import csv

import pandas as pd


def authors_as_list(authors):
    and_split = authors.split(" and ")
    if len(and_split) == 2:
        last_author = and_split[1]
        authors = and_split[0]
    elif len(and_split) == 1:
        last_author = None
        authors = and_split[0]
    else:
        raise ValueError("The and split returned more than 2 values")
    res = [a.strip() for a in authors.split(",")]
    if last_author:
        res.append(last_author)
    return res


def read_topic(topic_file_path):
    area_topics = {}
    topics_area = {}
    with open(topic_file_path) as f:
        reader = csv.DictReader(f)
        current_area = ""
        current_topics = []
        for row in reader:
            topic = row["topic"]
            if row["header?"] == "yes":
                if current_topics:
                    area_topics[current_area] = current_topics
                current_area = topic
                current_topics = []
            else:
                current_topics.append(topic)
                topics_area[topic] = current_area
        area_topics[current_area] = current_topics
    return area_topics, topics_area


def read_committee(
    committee_file_path,
    *,
    committee_topic_file_path=None,
    topics_to_areas=None,
    bids_file_path=None,
):
    df = pd.read_csv(committee_file_path, delimiter=",", encoding="utf-8")
    df["person #"] = pd.to_numeric(df["person #"], downcast="integer")
    df["full name"] = df["first name"] + " " + df["last name"]

    if committee_topic_file_path:
        if not topics_to_areas:
            raise ValueError(
                "If you set the committee_topic_file_path, then you also need to "
                "provide the topics_to_areas mapping."
            )
        pc_to_topics = {}
        pc_to_areas = {}
        with open(committee_topic_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic = row["topic"]  # The topic
                member_id = int(row["member #"].strip())  # The id of the PC member
                if member_id in pc_to_topics:
                    pc_to_topics[member_id].append(topic)
                else:
                    pc_to_topics[member_id] = [topic]
                area = topics_to_areas[topic]
                if member_id in pc_to_areas:
                    pc_to_areas[member_id].append(area)
                else:
                    pc_to_areas[member_id] = [area]
        df["topics"] = df.apply(
            lambda df_row: pc_to_topics.get(df_row["#"], []), axis=1
        )
        df["areas"] = df.apply(lambda df_row: pc_to_areas.get(df_row["#"], []), axis=1)

    if bids_file_path:
        pc_to_bids = {}
        bid_levels = set()
        with open(bids_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                member_id = int(row["member #"])
                submission = int(row["submission #"])
                bid = row["bid"]
                bid_levels.add(bid.strip().replace(" ", "_"))
                if member_id in pc_to_bids:
                    if bid in pc_to_bids[member_id]:
                        pc_to_bids[member_id][bid].append(submission)
                    else:
                        pc_to_bids[member_id][bid] = [submission]
                else:
                    pc_to_bids[member_id] = {bid: [submission]}

        def find_bids(df_row, bid_level):
            bids_dict = pc_to_bids.get(df_row["#"], None)
            if bids_dict is not None:
                return bids_dict.get(bid_level, [])
            return []

        for bid in bid_levels:
            df["bids_" + bid] = df.apply(lambda df_row: find_bids(df_row, bid), axis=1)

    return df


def read_submission(
    submission_file_path,
    *,
    submission_topic_file_path=None,
    author_file_path=None,
    review_file_path=None,
    submission_field_value_path=None,
    topics_to_areas=None,
):
    df = pd.read_csv(submission_file_path, delimiter=",", encoding="utf-8")
    df.drop(df[df["deleted?"] == "yes"].index, inplace=True)
    df.drop(df[df["decision"] == "desk reject"].index, inplace=True)

    if submission_topic_file_path:
        sub_to_topics = {}
        sub_to_areas = {}
        with open(submission_topic_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic = row["topic"]  # The topic
                sub_id = int(row["submission #"].strip())  # The id of the submission
                if sub_id in sub_to_topics:
                    sub_to_topics[sub_id].append(topic)
                else:
                    sub_to_topics[sub_id] = [topic]
                if topics_to_areas:
                    area = topics_to_areas[topic]
                    if sub_id in sub_to_areas:
                        sub_to_areas[sub_id].append(area)
                    else:
                        sub_to_areas[sub_id] = [area]
        df["topics"] = df.apply(
            lambda df_row: sub_to_topics.get(df_row["#"], []), axis=1
        )
        if topics_to_areas:
            df["areas"] = df.apply(
                lambda df_row: sub_to_areas.get(df_row["#"], []), axis=1
            )

    if author_file_path:
        sub_to_authors = {}
        with open(author_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = int(row["submission #"].strip())  # The id of the submission
                person_id = int(row["person #"].strip())  # The id of the person in EC
                if sub_id in sub_to_authors:
                    sub_to_authors[sub_id].append(person_id)
                else:
                    sub_to_authors[sub_id] = [person_id]
        df["authors_id"] = df.apply(
            lambda df_row: sub_to_authors.get(df_row["#"], []), axis=1
        )

    if submission_field_value_path:
        sub_to_is_students = {}
        with open(submission_field_value_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = int(row["submission #"].strip())  # The id of the submission
                field_name = row["field name"]
                if field_name == "student-paper":
                    sub_to_is_students[sub_id] = row["value"] == "allstudent"
        df["all_authors_students"] = df.apply(
            lambda df_row: sub_to_is_students.get(df_row["#"], False), axis=1
        )

    if review_file_path:
        sub_to_total_scores = {}
        with open(review_file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = int(row["submission #"].strip())
                total_score = int(row["total score"].strip())
                if sub_id in sub_to_total_scores:
                    sub_to_total_scores[sub_id].append(total_score)
                else:
                    sub_to_total_scores[sub_id] = [total_score]
        df["total_scores"] = df.apply(
            lambda df_row: sub_to_total_scores.get(df_row["#"], []), axis=1
        )
    return df
