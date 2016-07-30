import csv
import numpy as np
import networkx as nx
import random


def read_responses(csv_file):
    reader = csv.reader(open(csv_file))
    hdr = reader.next()
    rows = [_ for _ in reader]
    return hdr, rows


def parse_interests(value):
    return [_.strip() for _ in value.split(",")]


def get_topics(data):
    return filter(None, enumerate_interests(data).keys())


def get_education(data):
    levels = set()
    for row in data:
        levels.add(row[4])
    levels = list(levels)
    levels.sort()
    return levels[:-1]


def get_countries(data):
    countries = set()
    for row in data:
        countries.add(row[5].split("/")[-1].strip())
    return list(countries)


def get_affilations(data):
    affilations = set()
    for row in data:
        for aff in row[3].split("/"):
            affilations.add(aff.strip())
    return list(affilations)


def enumerate_interests(data, col_idx=7):
    interests = set()
    for row in data:
        for i in parse_interests(row[col_idx]):
            interests.add(i)
    return dict([(k, n) for n, k in enumerate(interests)])


def interest_bitmap(data, enum_map, col_idx=7):
    N = len(data)
    K = len(enum_map)

    bitmap = np.zeros([N, K], dtype=bool)
    for idx, row in enumerate(data):
        for i in parse_interests(row[col_idx]):
            bitmap[idx, enum_map[i]] = True
    return bitmap


def index_seniority(data, col_idx=6):
    tiers = list(set([_[col_idx] for _ in data]))
    tiers.sort()
    enum_map = dict([(k, n) for n, k in enumerate(tiers)])
    return [enum_map[row[col_idx]] for row in data]


def index_group(data, col_idx=8):
    return np.array([int(row[col_idx]) for row in data])


def categorical_sample(pdf):
    """Randomly select a categorical index of a given PDF."""
    pdf = pdf / pdf.sum()
    return int(np.random.multinomial(1, pdf).nonzero()[0])


def build_graph(data, forced_edges=None, null_edges=None, interest_func='l0',
                seniority_func='l0', combination_func=np.sum):
    """
    Parameters
    ----------
    data: list
        Rows from the CSV file, minus the header.
    forced_edges: np.ndarray, or None
        One-hot assignment matrix; no row or column can sum to more than one.
    null_edges: np.ndarray, or None
        Matches to set to zero.
    interest_func: str
        'l1', 'l0'
    seniority_func: str
        'l1', 'l0'
    combination_func: function
        Numpy functions, e.g. prod, sum, max.
    """
    N = len(data)
    seniority = index_seniority(data, 6)
    interests_to_idx = enumerate_interests(data, 7)
    interests = interest_bitmap(data, interests_to_idx, 7)

    if null_edges is None:
        null_edges = np.zeros([N]*2, dtype=bool)

    null_edges[np.eye(N, dtype=bool)] = True

    if forced_edges is None:
        forced_edges = np.zeros([N]*2, dtype=bool)

    graph = nx.Graph()
    for i, row_i in enumerate(data):
        for j, row_j in enumerate(data):
            # Skip self, shared affiliations, or same grouping
            if row_i[3] == row_j[3] or null_edges[i, j]:
                continue

            weights = []
            # Interest weighting
            weights += [(interests[i] * interests[j])]
            if interest_func == 'l0':
                weights[-1] = float(weights[-1].sum() > 0)
            elif interest_func == 'l1':
                weights[-1] = float(weights[-1].sum())
            elif interest_func == 'mean':
                weights[-1] = float(weights[-1].sum()) / interests.mean()
            else:
                weights[-1] = 0.0

            # Seniority weighting
            weights += [(seniority[i] - seniority[j]) ** 2.0]
            if seniority_func == 'l0':
                weights[-1] = float(weights[-1] > 0)
            elif seniority_func == 'euclidean':
                weights[-1] = np.sqrt(weights[-1])
            elif seniority_func == 'norm_euclidean':
                weights[-1] = np.sqrt(weights[-1]) / 3.0
            elif seniority_func == 'quadratic':
                weights[-1] = weights[-1]
            elif seniority_func == 'norm_quadratic':
                weights[-1] = weights[-1] / 9.0
            else:
                weights[-1] = 0.0

            if forced_edges[i, j]:
                weights = [2.0**32]
                # print row_i[1], row_j[1]
            graph.add_weighted_edges_from([(i, j, combination_func(weights))])

    return graph


def harmonic_mean(values):
    return np.power(np.prod(values), 1./len(values))


def select_matches(data, k_matches):
    matches = dict()
    null_edges = np.eye(len(data), dtype=bool)
    forced_edges = np.zeros_like(null_edges, dtype=bool)
    forced_edges[13, 60] = True
    forced_edges[60, 13] = True
    forced_edges[35, 65] = True
    forced_edges[65, 35] = True

    for k in range(k_matches):
        graph = build_graph(
            data, null_edges=null_edges, forced_edges=forced_edges,
            seniority_func='quadratic', interest_func='mean',
            combination_func=np.mean)
        forced_edges = None
        links = nx.max_weight_matching(graph)
        for row, col in links.items():
            null_edges[row, col] = True
            if not row in matches:
                matches[row] = list()
            matches[row].append(col)

    catch_count = 0
    for row in matches:
        while len(matches[row]) != k_matches:
            col = categorical_sample(1.0 - null_edges[row])
            matches[row].append(col)
            null_edges[row, col] = True
            catch_count += 1

    print "backfilled %d" % catch_count
    return matches


def select_topic(row_a, row_b):
    topics_a = parse_interests(row_a[7])
    topics_b = parse_interests(row_b[7])

    topics = list(set(topics_a).intersection(set(topics_b)))
    if topics:
        return topics[categorical_sample(np.ones(len(topics)))]

text_1fmts = [
    ("Find someone from %s.", 'affiliation'),
    ("Find someone currently located in %s.", 'country'),
    ("Find someone who is an expert on %s", 'topics'),
    ("Find someone in academia at the %s level", 'education')]


text_fmts = [
    "Find someone who works in industry",
    "Introduce someone to someone else",
    "Help someone solve a square",
    "Find someone who plays an instrument.",
    "Find someone who has attended ISMIR for more than 5 years",
    "Find someone for which this is their first ISMIR"]


def generate_text(rows, target_idx, matches, num_outputs=24):
    outputs = []

    for match_idx in matches[target_idx]:
        outputs.append("Talk to %s" % rows[match_idx][1])

    outputs.extend(text_fmts)
    categories = {
        'affiliation': get_affilations(rows),
        'education': get_education(rows),
        'topics': get_topics(rows),
        'country': get_countries(rows)
    }

    while len(outputs) < num_outputs:
        fmt, key = random.choice(text_1fmts)
        value = random.choice(categories[key])
        outputs.append(fmt % value)

    return outputs
