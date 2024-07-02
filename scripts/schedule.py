import csv
import enum
import os

import time
from collections.abc import Collection

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from mip import Model, BINARY, xsum, maximize, OptimizationStatus
from sklearn.cluster import KMeans, DBSCAN

from easychair_extra.read import read_committee, read_submission, authors_as_list
from easychair_extra.submission import bid_similarity, topic_similarity


class Presentation:
    def __init__(
        self,
        name,
        pairwise_scores=None,
        title=None,
        authors=None,
        duration=None,
        group_of=None,
    ):
        self.name = str(name)
        self.title = title
        self.authors = authors
        self.duration = duration
        self.group_of = group_of
        self.pairwise_scores = pairwise_scores

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Presentation):
            return self.name == other.name
        return self.name.__eq__(other)

    def __le__(self, other):
        if isinstance(other, Presentation):
            return self.name <= other.name
        return self.name.__le__(other)

    def __lt__(self, other):
        if isinstance(other, Presentation):
            return self.name < other.name
        return self.name.__lt__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


class Session:
    def __init__(self, name, max_duration=None, min_duration=None, presentations=None):
        self.name = name
        self.min_duration = min_duration
        self.max_duration = max_duration
        if presentations is None:
            presentations = []
        self.presentations = presentations

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Session):
            return self.name == other.name
        return self.name.__eq__(other)

    def __le__(self, other):
        if isinstance(other, Session):
            return self.name <= other.name
        return self.name.__le__(other)

    def __lt__(self, other):
        if isinstance(other, Session):
            return self.name < other.name
        return self.name.__lt__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


class TimeSlot:
    def __init__(self, name, sessions=None):
        self.name = name
        if sessions is None:
            sessions = []
        self.sessions = sessions

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name.__eq__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


def avg(iterable):
    d = 0
    res = 0
    for x in iterable:
        res += x
        d += 1
    if d > 0:
        return res / d
    return 0


def cluster_presentations(
    presentations: Collection[Presentation], n_clusters: int, algorithm: str = "kmeans"
) -> list[list[Presentation]]:
    score_matrix = np.array(
        [
            [p1.pairwise_scores.get(p2, 0) for p1 in presentations]
            for p2 in presentations
        ]
    )
    distance_matrix = 1 - score_matrix
    np.fill_diagonal(distance_matrix, 0)

    if algorithm == "kmeans":
        os.environ['OMP_NUM_THREADS'] = '16'
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(distance_matrix)
        labels = kmeans.labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric="precomputed", eps=0.9, min_samples=2)
        dbscan.fit(distance_matrix)
        labels = dbscan.labels_
        n_clusters = max(labels)
    else:
        raise ValueError("The algorithm is unknown")

    clusters = [[] for _ in range(n_clusters)]
    for item, label in zip(presentations, labels):
        if label >= 0:
            clusters[label].append(item)
    return clusters


def iterative_merge_clustering(
    presentations: Collection[Presentation],
    merging_bound: int,
    n_cluster_ratio_min: float = 0,
    n_cluster_ratio_max: float = 1,
) -> set[Presentation]:
    current_pres = set(presentations)
    while True:
        global_merged_happened = False
        min_n_clusters = max(2, int(n_cluster_ratio_min * len(current_pres)))
        max_n_clusters = min(
            len(current_pres), int(n_cluster_ratio_max * len(current_pres))
        )
        for n_clusters in range(min_n_clusters, max_n_clusters):
            clusters = cluster_presentations(current_pres, n_clusters)
            local_merged_happened = False
            for c in clusters:
                if len(c) > 1 and sum(p.duration for p in c) <= merging_bound:
                    print(f"Merging {c}")
                    current_pres.difference_update(c)
                    new_group = []
                    for p in c:
                        if p.group_of:
                            new_group.extend(p.group_of)
                        else:
                            new_group.append(p)
                    merged_presentation = Presentation(
                            name=" - ".join(str(p.name) for p in new_group),
                            duration=sum(p.duration for p in new_group),
                            pairwise_scores={
                                p2: sum(p1.pairwise_scores[p2] for p1 in c)
                                for p2 in current_pres
                            },
                            group_of=new_group,
                        )
                    merged_presentation.pairwise_scores[merged_presentation] = 0
                    current_pres.add(merged_presentation)
                    for p in current_pres:
                        p.pairwise_scores[merged_presentation] = merged_presentation.pairwise_scores[p]
                    print(f"\t...done, new len = {len(current_pres)}")
                    local_merged_happened = True
                    global_merged_happened = True
            if local_merged_happened:
                break
        if not global_merged_happened:
            break
    return current_pres


def schedule_flow(presentations: Collection[Presentation], sessions: Collection[Session]) -> dict[Session, list[Presentation]]:
    graph = nx.DiGraph()

    graph.add_node("source", demand=sum(p.duration for p in presentations))

    print("Building network flow")
    for session in sessions:
        graph.add_edge(session, "target", capacity=1)

    for presentation in presentations:
        graph.add_edge("source", f"{presentation}_in", capacity=1)
        graph.add_edge(f"{presentation}_in", f"{presentation}_out", capacity=1)
        for session in sessions:
            graph.add_edge(f"{presentation}_out", session, capacity=1)

        for presentation2 in presentations:
            score = presentation.pairwise_scores.get(presentation2, 0)
            if score > 0:
                graph.add_edge(f"{presentation}_out", f"{presentation2}_in", capacity=1, weight=-score)
    print(f"Done, {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    flow = nx.max_flow_min_cost(graph, "source", "target")
    for source in flow:
        print(source)
        for target, value in flow[source].items():
            if value > 0:
                print(f"\t{target} ({value})")
    mincost = nx.cost_of_flow(graph, flow)
    print(f"Minimum cost of value {mincost} found")

    schedule = {session: [] for session in sessions}
    session_names = {session.name for session in sessions}
    for presentation in presentations:
        if flow["source"][f"{presentation}_in"] > 0.9:
            print(f"Flow for {presentation}")
            connected_presentations = [presentation]
            current_node = f"{presentation}_in"
            while True:
                next_node = ""
                for node, value in flow[current_node[:-2] + "out"].items():
                    if value > 0.9:
                        print(f"\tfound flow to {node}")
                        next_node = node
                print(f"\tnext node is {next_node}")
                if next_node in session_names:
                    break
                for p in presentations:
                    if p.name == next_node[:-3]:
                        connected_presentations.append(p)
                current_node = next_node
            print(f"For {next_node} with {connected_presentations}")
            schedule[next_node] = connected_presentations
    print(schedule)

    edge_colors = []
    for u, v in graph.edges():
        if flow[u][v] > 0:
            edge_colors.append('black')
        else:
            edge_colors.append('lightgrey')

    pos = nx.shell_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=700)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black')
    plt.show()
    return schedule


class Objectives(enum.Enum):
    MIN_SCORE = 1
    TOTAL_SC0RE = 2
    MIN_THEN_TOTAL_SCORE = 3


def egalitarian_schedule_ilp(
    presentations: Collection[Presentation], sessions: Collection[Session], max_run_time: int = 300,
        objective: Objectives = Objectives.MIN_SCORE, initial_solution=False,
):
    start_time = time.time()
    m = Model()

    # Variable: Presentation is in session
    pres_to_session_vars = {}
    for presentation in presentations:
        vars_dict = {}
        for session in sessions:
            vars_dict[session] = m.add_var(
                f"x_{presentation}_{session}", var_type=BINARY
            )
        pres_to_session_vars[presentation] = vars_dict
    current_time = time.time()
    print(f"Added presentations vars in {current_time - start_time:.3f}s")

    # Constraint: All presentations are assigned to exactly one session
    for vars_dict in pres_to_session_vars.values():
        m += xsum(vars_dict.values()) == 1
    print(
        f"Added presentation to unique session constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Variable: Session is not empty
    sessions_non_empty_vars = {
        s: m.add_var(f"y_{s}", var_type=BINARY) for s in sessions
    }
    print(
        f"Added session non empty vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: Session not empty
    for vars_dict in pres_to_session_vars.values():
        for session, v in vars_dict.items():
            m += v <= sessions_non_empty_vars[session]
    print(
        f"Added session non empty constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: Duration of the session
    for session in sessions:
        m += (
            xsum(pres_to_session_vars[p][session] * p.duration for p in presentations)
            <= session.max_duration
        )
        m += (
            xsum(pres_to_session_vars[p][session] * p.duration for p in presentations)
            >= session.min_duration * sessions_non_empty_vars[session]
        )
    print(
        f"Added session duration constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Variable: Papers are together in a session
    pres_together_vars = {}
    for pres1 in presentations:
        pres_together_vars[pres1] = {}
        for pres2 in presentations:
            if pres1 != pres2:
                vars_dict = {}
                for session in sessions:
                    vars_dict[session] = m.add_var(
                        f"z_{pres1}_{pres2}_{session}", var_type=BINARY
                    )
                pres_together_vars[pres1][pres2] = vars_dict
    print(
        f"Added together vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: If together variables are 1 then papers are in session, and reciprocally
    for pres1, other_vars_dict in pres_together_vars.items():
        for pres2, session_vars in other_vars_dict.items():
            for session, v in session_vars.items():
                m += v <= pres_to_session_vars[pres1][session]
                m += v <= pres_to_session_vars[pres2][session]
                m += (
                    pres_to_session_vars[pres1][session]
                    + pres_to_session_vars[pres2][session]
                    <= 1 + v
                )
    print(
        f"Added together constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    if objective in [Objectives.MIN_SCORE, Objectives.MIN_THEN_TOTAL_SCORE]:
        # Variable: Min score of a session
        sessions_min_score_vars = {
            session: m.add_var(f"a_{session}") for session in sessions
        }
        print(
            f"Added session min score vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

        # Constraint: Min score of a session is min match
        for pres1, other_vars_dict in pres_together_vars.items():
            for pres2, session_vars_dict in other_vars_dict.items():
                for session, v in session_vars_dict.items():
                    m += (
                        sessions_min_score_vars[session]
                        <= pres_together_vars[pres1][pres2][session]
                        * pres1.pairwise_scores[pres2]
                        + (1 - pres_together_vars[pres1][pres2][session]) * 100
                    )
        for session, v in sessions_min_score_vars.items():
            m += v <= sessions_non_empty_vars[session] * 100
        print(
            f"Added session min score constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

    if objective in [Objectives.TOTAL_SC0RE, Objectives.MIN_THEN_TOTAL_SCORE]:
        # Variable: Total score of a session
        sessions_total_score_vars = {
            session: m.add_var(f"b_{session}") for session in sessions
        }
        print(
            f"Added session total score vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

        # Constraint: Total score of a session is total match
        for session, v in sessions_total_score_vars.items():
            m += v == xsum(pres_together_vars[pres1][pres2][session] * pres1.pairwise_scores[pres2] for pres1, other_vars_dict in pres_together_vars.items() for pres2 in other_vars_dict)
        print(
            f"Added session total score constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

    if objective == Objectives.MIN_SCORE:
        m.objective = maximize(xsum(sessions_min_score_vars.values()))
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    elif objective == Objectives.TOTAL_SC0RE:
        m.objective = maximize(xsum(sessions_total_score_vars.values()))
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    elif objective == Objectives.MIN_THEN_TOTAL_SCORE:
        m.objective = maximize(100 * len(sessions) * xsum(sessions_min_score_vars.values()) + xsum(sessions_total_score_vars.values()))
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    else:
        raise ValueError("Incorrect value for the objective.")

    if initial_solution:
        start = [
            (pres_to_session_vars[p][s], int(p in s.presentations)) for s in sessions for p in presentations
        ]
        start += [
            (pres_together_vars[p1][p2][s], int(p1 in s.presentations and p2 in s.presentations)) for s in sessions for p1, vars_dict in pres_together_vars.items() for p2 in vars_dict
        ]
        m.start = start

    opt_status = m.optimize(max_seconds=max_run_time)

    if opt_status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        solution = {session: [] for session in sessions}
        for presentation, vars_dict in pres_to_session_vars.items():
            for session, v in vars_dict.items():
                if v.x >= 0.9:
                    solution[session].append(presentation)
        return solution


def presentation_score_all(presentation: Presentation, presentations: Collection[Presentation]):
    return tuple(presentation.pairwise_scores[p] for p in presentations if p != presentation)


def presentation_score_sum(presentation: Presentation, presentations: Collection[Presentation]):
    return sum(presentation_score_all(presentation, presentations))


def presentation_score_avg(presentation: Presentation, presentations: Collection[Presentation]):
    return sum(presentation_score_all(presentation, presentations)) / len(presentations)


def presentation_score_min(presentation: Presentation, presentations: Collection[Presentation]):
    all_scores = presentation_score_all(presentation, presentations)
    if all_scores:
        return min(all_scores)
    return 0


def session_score_min(presentations: Collection[Presentation], presentation_score_func):
    all_scores = [presentation_score_func(p, presentations) for p in presentations]
    if all_scores:
        return min(all_scores)
    return 0


def session_score_sum(presentations: Collection[Presentation], presentation_score_func):
    return sum(presentation_score_func(p, presentations) for p in presentations)


def session_score_avg(presentations: Collection[Presentation], presentation_score_func):
    return sum(presentation_score_func(p, presentations) for p in presentations) / len(presentations)


def session_score_leximin(presentations: Collection[Presentation], presentation_score_func):
    return tuple(sorted(presentation_score_func(p, presentations) for p in presentations))


def schedule_score_all(sessions: Collection[Session], session_score_func, presentation_score_func):
    return tuple(session_score_func(s.presentations, presentation_score_func) for s in sessions if len(s.presentations) > 0)


def schedule_score_min(sessions: Collection[Session], session_score_func, presentation_score_func):
    all_scores = schedule_score_all(sessions, session_score_func, presentation_score_func)
    if all_scores:
        return min(all_scores)
    return 0


def schedule_score_sum(sessions: Collection[Session], session_score_func, presentation_score_func):
    return sum(schedule_score_all(sessions, session_score_func, presentation_score_func))


def schedule_score_avg(sessions: Collection[Session], session_score_func, presentation_score_func):
    return sum(schedule_score_all(sessions, session_score_func, presentation_score_func)) / len(sessions)


def schedule_score_leximin(sessions: Collection[Session], session_score_func, presentation_score_func):
    res = []
    for session_score in schedule_score_all(sessions, session_score_func, presentation_score_func):
        if isinstance(session_score, tuple):
            for i, s in enumerate(session_score):
                if len(res) <= i:
                    res.append(s)
                else:
                    res[i] += s
        else:
            res.append(session_score)
    return tuple(sorted(res))


def schedule_to_str(sessions: Collection[Session]):
    res = ""
    for s in sorted(sessions):
        for p in sorted(s.presentations):
            res += str(p)
    return res


def schedule_local_search(
        sessions: Collection[Session],
        schedule_score_func=schedule_score_leximin,
        session_score_func=session_score_leximin,
        presentation_score_func=presentation_score_avg
):
    print("Starting local search")
    visited_schedules = {schedule_to_str(sessions)}
    while True:
        current_score = schedule_score_func(sessions, session_score_func, presentation_score_func)
        best_new_score = None
        arg_best_new_score = None
        for session in sessions:
            for presentation in session.presentations:
                for session2 in sessions:
                    if session2 != session:
                        if sum(p.duration for p in session2.presentations) + presentation.duration <= session2.max_duration:
                            new_sessions = [s for s in sessions if s != session and s != session2]
                            new_sessions.append(Session(
                                    name="tmp_session1",
                                    presentations=list(session2.presentations) + [presentation]
                            ))
                            new_sessions.append(Session(
                                    name="tmp_session2",
                                    presentations=[p for p in session.presentations if p != presentation]
                            ))
                            new_sessions_str = schedule_to_str(new_sessions)
                            if new_sessions_str not in visited_schedules:
                                new_score = schedule_score_func(new_sessions, session_score_func, presentation_score_func)
                                if new_score > current_score:
                                    if best_new_score is None or new_score > best_new_score:
                                        best_new_score = new_score
                                        arg_best_new_score = (session, session2, presentation)
        if arg_best_new_score is not None:
            session, session2, presentation = arg_best_new_score
            session.presentations.remove(presentation)
            session2.presentations.append(presentation)
            print(f"I'm moving {presentation} from {session} to {session2}")
        else:
            print("No improvement, I'm stopping.")
            break


def schedule_to_csv(schedule, submission_df, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timeslot", "session", "submission #", "title"])

        for session, papers in schedule.items():
            for paper in papers:
                writer.writerow(
                    [
                        session.split("_")[0],
                        session.split("_")[1],
                        paper,
                        submission_df.loc[submission_df["#"] == paper]["title"].iloc[0],
                    ]
                )


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
        f.write(
            """<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://ecai2024.simonrey.fr/static/css/style.css">
    <title>ECAI 2024 Schedule</title>
</head>
<body>
<header><h1>ECAI-2024 — Schedule</h1></header>
<main>"""
        )
        for time_slot, schedule in schedule_per_slot.items():
            f.write(f"<section><div class='section-content'><h2>{time_slot}</h2>")
            for session, papers in schedule.items():
                f.write(
                    f"<h3>Session {session} (s = {compute_session_total_score(papers, submission_similarities):.3f}, min = {compute_session_min_score(papers, submission_similarities):.3f})</h3>"
                )
                f.write(
                    """<table class="lined-table"><tr><th>Paper</th><th>Score</th></tr>"""
                )
                for paper in papers:
                    f.write(
                        f"<tr><td>#{paper}: {submission_df.loc[submission_df['#'] == paper]['title'].iloc[0]}</td><td>{compute_paper_score(paper, papers, submission_similarities)}</td></tr>"
                    )
                f.write("</table>")
            f.write("</div></section>")
        f.write("</main></body></html>")


def draw_similarities(similarities, num_bins):
    all_s = [s for sim_dict in similarities.values() for s in sim_dict.values()]
    bin_step = max(all_s)/num_bins
    bins_cutoff = [bin_step * k for k in range(num_bins + 1)]
    data = [0 for _ in bins_cutoff]
    for s in all_s:
        cutoff = 0
        while cutoff < num_bins and s >= bins_cutoff[cutoff]:
            cutoff += 1
        if cutoff > 0:
            data[cutoff - 1] += 1

    labels = [f"{round(bins_cutoff[i], 2)} - {round(bins_cutoff[i + 1], 2)}" for i in
              range(num_bins)]
    labels.append(f"{round(bins_cutoff[-1], 2)} - {max(all_s)}")
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Similarity Bins')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarities')
    plt.tight_layout()
    plt.show()


def main():
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")

    committee = read_committee(
        os.path.join(csv_dir, "committee.csv"),
        bids_file_path=os.path.join(csv_dir, "bidding.csv"),
    )

    submissions = read_submission(
        os.path.join(csv_dir, "submission.csv"),
        review_file_path=os.path.join(csv_dir, "review.csv"),
        submission_topic_file_path=os.path.join(csv_dir, "submission_topic.csv"),
    )
    submissions["avg_total_scores"] = submissions.apply(
        lambda df_row: avg(df_row.get("total_scores", [])), axis=1
    )
    accepted_submissions = submissions[submissions["decision"] == "accept"]
    print("=" * 50)
    print(f"Data loaded: {len(accepted_submissions.index)} submissions")

    bid_level_weights = {"yes": 1, "maybe": 0.5}

    bid_sim = bid_similarity(accepted_submissions, committee, bid_level_weights)
    # draw_similarities(bid_sim, 10)
    topic_sim = topic_similarity(accepted_submissions)
    # draw_similarities(topic_sim, 10)
    aggregated_similarity = {}
    for sub_id1, similarities in bid_sim.items():
        aggregated_similarity[sub_id1] = {}
        for sub_id2, s in similarities.items():
            if sub_id1 == sub_id2:
               aggregated_similarity[sub_id1][sub_id2] = 0
            else:
                s2 = topic_sim[sub_id1][sub_id2]
                if s * s2 == 0 and s + s2 > 0:
                    aggregated_similarity[sub_id1][sub_id2] = max(1, int(max(s, s2) * 10))
                else:
                    aggregated_similarity[sub_id1][sub_id2] = max(1, int(s * s2 * 1000))

    print("=" * 50)
    print("SIMILARITY SCORES")
    print(f"Number of 0 similarity: {len([s for d in aggregated_similarity.values() for s in d.values() if s == 0])}\n")
    # draw_similarities(aggregated_similarity, 40)

    all_presentations = []
    accepted_submissions.apply(
        lambda df_row: all_presentations.append(
            Presentation(
                df_row["#"],
                title=df_row["title"],
                authors=authors_as_list(df_row["authors"]),
                pairwise_scores={str(k): v for k, v in aggregated_similarity[df_row["#"]].items()},
                duration=1,
            )
        ),
        axis=1,
    )

    all_time_slots = [
        TimeSlot("Mon-am"),
        TimeSlot("Mon-pm"),
        TimeSlot("Tue-am"),
        TimeSlot("Tue-pm"),
        TimeSlot("Wed-am"),
        TimeSlot("Wed-pm"),
        TimeSlot("Thu-am"),
    ]

    for t in all_time_slots:
        for k in range(1, 11):
            t.sessions.append(Session(f"{t.name}_{k}", min_duration=5, max_duration=10))

    schedule = schedule_flow(all_presentations[:8], [s for t in all_time_slots for s in t.sessions][:2])
    for session, presentations in schedule.items():
        session.presentations = presentations
    print(f"TOTAL: min={schedule_score_min(schedule.keys(), session_score_min, presentation_score_avg)}, total={schedule_score_sum(schedule.keys(), session_score_min, presentation_score_avg)}")
    print(f"leximin={schedule_score_leximin(schedule.keys(), session_score_min, presentation_score_avg)}")

    # print("\n" + "=" * 50)
    # print("Merge clustering")
    # merged_presentations = iterative_merge_clustering(
    #     all_presentations, 6, n_cluster_ratio_min=0, n_cluster_ratio_max=0.1
    # )
    # all_presentations = merged_presentations
    #
    # print("\n" + "=" * 50)
    # print("Cluster in 10")
    # clusters = cluster_presentations(all_presentations, 10, algorithm="kmeans")
    # for i, c in enumerate(clusters):
    #     print(f"Cluster {i}: {sum(p.duration for p in c)} {c}")
    #
    # print("\n" + "=" * 50)
    # print("MIN_SCORE ILP")
    # schedule = egalitarian_schedule_ilp(all_presentations, [s for t in all_time_slots for s in t.sessions], max_run_time=120, objective=Objectives.MIN_SCORE)
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    # print("\n" + "=" * 50)
    # print("MIN_THEN_TOTAL_SCORE ILP")
    # schedule = egalitarian_schedule_ilp(all_presentations, [s for t in all_time_slots for s in t.sessions], max_run_time=120, objective=Objectives.MIN_THEN_TOTAL_SCORE, initial_solution=True)
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    #
    # all_sessions = []
    # for session in schedule.items():
    #     new_presentations = []
    #     presentations_to_remove = []
    #     for presentation in session.presentations:
    #         if presentation.group_of:
    #             new_presentations.extend(presentation.group_of)
    #             presentations_to_remove.append(presentation)
    #     session.presentations.extend(new_presentations)
    #     for p in presentations_to_remove:
    #         session.presentations.remove(p)
    #
    # print("\n" + "=" * 50)
    # print("Local search")
    # schedule_local_search(schedule.keys())
    # for s, pres in schedule.items():
    #     if len(pres) > 0:
    #         print(f"{s}: {len(pres)} {pres} min={session_score_min(pres, presentation_score_avg)}, avg={session_score_avg(pres, presentation_score_avg)},  total={session_score_sum(pres, presentation_score_avg)}")
    #
    # print(f"TOTAL: min={schedule_score_min(schedule.keys(), session_score_min, presentation_score_avg)}, total={schedule_score_sum(schedule.keys(), session_score_min, presentation_score_avg)}")
    # print(f"leximin={schedule_score_leximin(schedule.keys(), session_score_min, presentation_score_avg)}")


if __name__ == "__main__":
    main()
