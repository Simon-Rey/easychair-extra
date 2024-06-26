import csv
import os

import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from mip import Model, BINARY, xsum, maximize
from sklearn.cluster import KMeans, DBSCAN

from easychair_extra.read import read_committee, read_submission
from easychair_extra.submission import bid_similarity, topic_similarity


class TimeSlot:

    def __init__(self, name, num_rooms, min_num_papers, max_num_papers):
        self.name = name
        self.num_rooms = num_rooms
        self.min_num_papers = min_num_papers
        self.max_num_papers = max_num_papers

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other


def avg(iterable):
    d = 0
    res = 0
    for x in iterable:
        res += x
        d += 1
    if d > 0:
        return res / d
    return 0


def cluster_submissions(submission_df, submission_similarities, n_clusters=10, algorithm="kmeans"):
    similarity_matrix = np.array(
        [[submission_similarities[item1].get(item2, 0) for item2 in submission_similarities] for item1 in submission_similarities])
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    if algorithm == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(distance_matrix)
        labels = kmeans.labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric='precomputed', eps=0.9, min_samples=2)
        dbscan.fit(distance_matrix)
        labels = dbscan.labels_
        n_clusters = max(labels) + 1
    else:
        raise ValueError("The algorithm is unknown")

    clusters = [[] for _ in range(n_clusters)]
    for item, label in zip(submission_similarities, labels):
        clusters[label].append(item)
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {len(c)} {c}")
        # for p in c:
        #     print(f"\t{p}: {submission_df.loc[submission_df['#'] == p]['title'].iloc[0]}")
    return clusters


def schedule_flow(submission_df, time_slots, submission_similarities):

    all_sessions = [s.name + "_" + str(i) for s in time_slots for i in range(s.num_rooms)]

    graph = nx.DiGraph()

    graph.add_node("source", demand=submission_df["#"].count())

    for s in all_sessions:
        graph.add_edge(s, "target", capacity=1)

    for sub_id in submission_df["#"]:
        graph.add_edge("source", f"{sub_id}_in", capacity=1)
        graph.add_edge(f"{sub_id}_in", f"{sub_id}_out", capacity=1)
        for s in all_sessions:
            graph.add_edge(f"{sub_id}_out", s, capacity=1)
        for sub_id2, sim in submission_similarities[sub_id].items():
            if sim > 0:
                graph.add_edge(f"{sub_id}_out", f"{sub_id2}_in", capacity=1, weight=-sim)

    flow = nx.max_flow_min_cost(graph, "source", "target")
    mincost = nx.cost_of_flow(graph, flow)
    print(mincost)

    # edge_colors = []
    # for u, v in graph.edges():
    #     if flow[u][v] > 0:
    #         edge_colors.append('black')
    #     else:
    #         edge_colors.append('lightgrey')
    #
    # pos = nx.shell_layout(graph)
    # nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=700)
    # nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=2)
    # nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black')
    # plt.show()

    clusters = []
    for sub_id in submission_df["#"]:
        if flow["source"][f"{sub_id}_in"] > 0.9:
            current_cluster = []
            current_node = f"{sub_id}_in"
            while True:
                next_node = ""
                for node, value in flow[current_node[:-2] + "out"]:
                    if value > 0.9:
                        next_node = node
                if next_node in all_sessions:
                    break
                current_cluster.append(next_node)
                current_node = next_node
            clusters.append(current_cluster)
    return clusters


def schedule_ilp(submission_df, time_slots, submission_similarities):
    start_time = time.time()
    m = Model()

    subs_with_sim_zero = []
    for sub_id, similarities in submission_similarities.items():
        if sum(similarities.values()) == 0:
            subs_with_sim_zero.append(sub_id)
    if subs_with_sim_zero:
        print(f"The following papers are not compatible with any others: {subs_with_sim_zero}")
    else:
        print("No paper is completely incompatible.")
    submission_df = submission_df[~submission_df["#"].isin(subs_with_sim_zero)]

    slot_to_session = {}
    all_sessions = []
    for time_slot in time_slots:
        slot_to_session[time_slot] = []
        for i in range(time_slot.num_rooms):
            session_name = time_slot.name + "_" + str(i)
            all_sessions.append(session_name)
            slot_to_session[time_slot].append(session_name)

    # Variable: Paper is matched to session
    sub_to_session_vars = {}
    for sub_id in submission_df["#"]:
        sub_to_session_vars[sub_id] = {}
        for session in all_sessions:
            sub_to_session_vars[sub_id][session] = m.add_var(name=f"x_{sub_id}_{session}", var_type=BINARY)
    current_time = time.time()
    print(f"Added paper vars {current_time - start_time:.5f}")

    # Variable: Papers are together in a session
    session_together_subs_vars = {}
    for sub_id1 in submission_df["#"]:
        session_together_subs_vars[sub_id1] = {}
        for sub_id2 in submission_df["#"]:
            session_together_subs_vars[sub_id1][sub_id2] = {}
            if sub_id1 != sub_id2:
                if submission_similarities[sub_id1][sub_id2] > 0:
                    for session in all_sessions:
                        session_together_subs_vars[sub_id1][sub_id2][session] = m.add_var(name=f"y_{sub_id1}_{sub_id2}_{session}")
    print(f"Added together vars {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Variable: Session is not empty
    sessions_populated_vars = {}
    for session in all_sessions:
        sessions_populated_vars[session] = m.add_var(f"z_{session}", var_type=BINARY)
    print(f"Added session populated vars {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Variable: Score of a session
    sessions_score_vars = {}
    for session in all_sessions:
        sessions_score_vars[session] = m.add_var(f"a_{session}")
    print(f"Added session score vars {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Variable: Score of a paper
    submissions_score_vars = {}
    for sub_id in submission_df["#"]:
        submissions_score_vars[sub_id] = m.add_var(f"b_{sub_id}")
    print(f"Added paper score vars {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: Together variables cannot be 1 if not both are 1
    for sub_id1, other_vars in session_together_subs_vars.items():
        for sub_id2, session_vars in other_vars.items():
            for session, v in session_vars.items():
                m += v <= sub_to_session_vars[sub_id1][session]
                m += v <= sub_to_session_vars[sub_id2][session]
    # We do not add the complement formula as the direction of optimisation renders it useless
    print(f"Added together constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: All papers are assigned exactly once
    for submission_vars in sub_to_session_vars.values():
        m += xsum(submission_vars.values()) <= 1
    print(f"Added unicity constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: Session is not empty
    for submission_vars in sub_to_session_vars.values():
        for session, v in submission_vars.items():
            m += v <= sessions_populated_vars[session]
    print(f"Added session populated constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: Number of papers per session
    for time_slot, sessions in slot_to_session.items():
        for session in sessions:
            session_vars = [sub_vars[session] for sub_vars in sub_to_session_vars.values()]
            m += xsum(session_vars) <= time_slot.max_num_papers
            m += xsum(session_vars) >= time_slot.min_num_papers * sessions_populated_vars[session]
    print(f"Added number of papers per session constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: Score of a paper
    for sub_id, v in submissions_score_vars.items():
        m += v == xsum(session_together_subs_vars[sub_id][sub_id2].get(session, 0) * submission_similarities[sub_id][sub_id2] for sub_id2 in session_together_subs_vars[sub_id] for session in all_sessions)
        # m += v >= min(v for s in submission_similarities.values() for v in s.values() if v > 0)
    print(f"Added paper score constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Constraint: Score of a session
    for session, v in sessions_score_vars.items():
        m += v == xsum(session_together_subs_vars[sub_id1][sub_id2].get(session, 0) * submission_similarities[sub_id1][sub_id2] for sub_id1 in session_together_subs_vars for sub_id2 in session_together_subs_vars[sub_id1])
        # m += v >= min(min(s.values()) for s in submission_similarities.values())
    print(f"Added session score constraints {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    # Objective: Max score
    m.objective = maximize(xsum(sessions_score_vars.values()))
    print(f"Added objectives {time.time() - start_time:.5f} (+{time.time() - current_time:.5f})")
    current_time = time.time()

    m.optimize(max_seconds=60 * 10)

    solution = {session: [] for session in all_sessions}
    for paper_id, session_vars in sub_to_session_vars.items():
        for session, v in session_vars.items():
            if v.x >= 0.9:
                solution[session].append(paper_id)

    return solution


def schedule_to_csv(schedule, submission_df, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timeslot", "session", "submission #", "title"])

        for session, papers in schedule.items():
            for paper in papers:
                writer.writerow([session.split("_")[0], session.split("_")[1], paper, submission_df.loc[submission_df['#'] == paper]['title'].iloc[0]])


def compute_paper_score(paper, papers, submission_similarities):
    res = 0
    for p in papers:
        if p != paper:
            res += submission_similarities[paper][p]
    return res


def compute_session_total_score(papers, submission_similarities):
    res = 0
    for i in range(len(papers) - 1):
        for j in range(i + 1, len(papers)):
            res += submission_similarities[papers[i]][papers[j]]
    return res


def compute_session_min_score(papers, submission_similarities):
    return min(compute_paper_score(p, papers, submission_similarities) for p in papers)


def schedule_to_html(schedule, submission_df, file_path, submission_similarities):
    with open(file_path, "w", encoding="utf-8") as f:
        schedule_per_slot = {}
        for session, papers in schedule.items():
            if len(papers) > 0:
                time_slot, session_name = session.split("_")
                if time_slot in schedule_per_slot:
                    schedule_per_slot[time_slot][session_name] = papers
                else:
                    schedule_per_slot[time_slot] = {session_name: papers}
        f.write("""<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://ecai2024.simonrey.fr/static/css/style.css">
    <title>ECAI 2024 Schedule</title>
</head>
<body>
<header><h1>ECAI-2024 — Schedule</h1></header>
<main>""")
        for time_slot, schedule in schedule_per_slot.items():
            f.write(f"<section><div class='section-content'><h2>{time_slot}</h2>")
            for session, papers in schedule.items():
                f.write(f"<h3>Session {session} (s = {compute_session_total_score(papers, submission_similarities):.3f}, min = {compute_session_min_score(papers, submission_similarities):.3f})</h3>")
                f.write("""<table class="lined-table"><tr><th>Paper</th><th>Score</th></tr>""")
                for paper in papers:
                    f.write(f"<tr><td>#{paper}: {submission_df.loc[submission_df['#'] == paper]['title'].iloc[0]}</td><td>{compute_paper_score(paper, papers, submission_similarities)}</td></tr>")
                f.write("</table>")
            f.write("</div></section>")
        f.write("</main></body></html>")


if __name__ == "__main__":
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")

    committee = read_committee(
        os.path.join(csv_dir, "committee.csv"),
        bids_file_path=os.path.join(csv_dir, "bidding.csv")
    )

    submissions = read_submission(
        os.path.join(csv_dir, "submission.csv"),
        review_file_path=os.path.join(csv_dir, "review.csv"),
        submission_topic_file_path=os.path.join(csv_dir, "submission_topic.csv")
    )
    submissions["avg_total_scores"] = submissions.apply(lambda df_row: avg(df_row.get("total_scores", [])),
                                      axis=1)
    top_submissions = submissions.sort_values("avg_total_scores", ascending=False).head(500)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', 6, 'max_colwidth', 100):
    #     print(top_submissions.head(50))

    bid_level_weights = {'yes': 1, 'maybe': 0.5}

    bid_sim = bid_similarity(top_submissions, committee, bid_level_weights)
    topic_sim = topic_similarity(top_submissions)
    aggregated_similarity = {}
    for sub_id1, similarities in bid_sim.items():
        aggregated_similarity[sub_id1] = {}
        for sub_id2, s in similarities.items():
            aggregated_similarity[sub_id1][sub_id2] = s * topic_sim[sub_id1][sub_id2]
    # aggregated_similarity = bid_sim

    all_time_slots = [
        TimeSlot("Mon-am", 10, 5, 10),
        TimeSlot("Mon-pm", 10, 5, 10),
        TimeSlot("Tue-am", 10, 5, 10),
        TimeSlot("Tue-pm", 10, 5, 10),
        TimeSlot("Wed-am", 10, 5, 10),
        TimeSlot("Wed-pm", 10, 5, 10),
        TimeSlot("Thu-am", 10, 5, 10)
    ]

    # s = schedule_ilp(top_submissions, all_time_slots, aggregated_similarity)
    # schedule_to_csv(s, submissions, "final_schedule.csv")
    # schedule_to_html(s, submissions, "final_schedule.html", aggregated_similarity)

    # schedule_flow(top_submissions, all_time_slots, aggregated_similarity)
    #
    clusters = cluster_submissions(top_submissions, aggregated_similarity, n_clusters=20, algorithm="kmeans")

    # first_cluster_df = top_submissions[top_submissions["#"].isin(clusters[0])]
    # s = schedule_ilp(first_cluster_df, all_time_slots, aggregated_similarity)
    # schedule_to_csv(s, first_cluster_df, "final_schedule.csv")
    # schedule_to_html(s, first_cluster_df, "final_schedule.html", aggregated_similarity)

    # other_clusters = [p for c in clusters[1:] for p in c]
    # other_cluster_df = top_submissions[top_submissions["#"].isin(other_clusters)]
    # s = schedule_ilp(other_cluster_df, all_time_slots, aggregated_similarity)
    # schedule_to_csv(s, other_cluster_df, "final_schedule.csv")
    # schedule_to_html(s, other_cluster_df, "final_schedule.html", aggregated_similarity)

