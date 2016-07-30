from __future__ import print_function

import argparse
import csv
import logging
import numpy as np
import networkx as nx
import random
import sys

logger = logging.getLogger(name=__file__)


def read_responses(csv_file):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    reader = csv.reader(open(csv_file))
    hdr = reader.next()
    rows = [_ for _ in reader]
    return hdr, rows


def parse_interests(value):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    return [_.strip() for _ in value.split(",")]


def get_topics(data):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    return filter(None, enumerate_interests(data).keys())


def get_education(data):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    levels = set()
    for row in data:
        levels.add(row[4])
    levels = list(levels)
    levels.sort()
    return levels[:-1]


def get_countries(data):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    countries = set()
    for row in data:
        countries.add(row[5].split("/")[-1].strip())
    return list(countries)


def get_affilations(data):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    affilations = set()
    for row in data:
        for aff in row[3].split("/"):
            affilations.add(aff.strip())
    return list(affilations)


def enumerate_interests(data, col_idx=7):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    interests = set()
    for row in data:
        for i in parse_interests(row[col_idx]):
            interests.add(i)
    return dict([(k, n) for n, k in enumerate(interests)])


def interest_bitmap(data, enum_map, col_idx=7):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    N = len(data)
    K = len(enum_map)

    bitmap = np.zeros([N, K], dtype=bool)
    for idx, row in enumerate(data):
        for i in parse_interests(row[col_idx]):
            bitmap[idx, enum_map[i]] = True
    return bitmap


def index_seniority(data, col_idx=6):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    tiers = list(set([_[col_idx] for _ in data]))
    tiers.sort()
    enum_map = dict([(k, n) for n, k in enumerate(tiers)])
    return [enum_map[row[col_idx]] for row in data]


def index_group(data, col_idx=8):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    return np.array([int(row[col_idx]) for row in data])


def categorical_sample(pdf):
    """Randomly select a categorical index of a given PDF.

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    pdf = pdf / pdf.sum()
    return int(np.random.multinomial(1, pdf).nonzero()[0])


def build_graph(data, forced_edges=None, null_edges=None, interest_func='l0',
                seniority_func='l0', combination_func=np.sum):
    """writeme

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

    Returns
    -------
    """
    N = len(data)
    seniority = index_seniority(data, 6)
    interests_to_idx = enumerate_interests(data, 7)
    interests = interest_bitmap(data, interests_to_idx, 7)

    if null_edges is None:
        null_edges = np.zeros([N] * 2, dtype=bool)

    null_edges[np.eye(N, dtype=bool)] = True

    if forced_edges is None:
        forced_edges = np.zeros([N] * 2, dtype=bool)

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
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    return np.power(np.prod(values), 1.0 / len(values))


def select_matches(data, k_matches):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
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
            if row not in matches:
                matches[row] = list()
            matches[row].append(col)

    catch_count = 0
    for row in matches:
        while len(matches[row]) != k_matches:
            col = categorical_sample(1.0 - null_edges[row])
            matches[row].append(col)
            null_edges[row, col] = True
            catch_count += 1

    print("backfilled %d" % catch_count)
    return matches


def select_topic(row_a, row_b):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    topics_a = parse_interests(row_a[7])
    topics_b = parse_interests(row_b[7])

    topics = list(set(topics_a).intersection(set(topics_b)))
    if topics:
        return topics[categorical_sample(np.ones(len(topics)))]

TEXT_FMTS = [
    ("Find someone from %s.", 'affiliation'),
    ("Find someone currently located in %s.", 'country'),
    ("Find someone who is an expert on %s", 'topics'),
    ("Find someone in academia at the %s level", 'education')]


TEXT = [
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

    outputs.extend(TEXT)
    categories = {
        'affiliation': get_affilations(rows),
        'education': get_education(rows),
        'topics': get_topics(rows),
        'country': get_countries(rows)
    }

    while len(outputs) < num_outputs:
        fmt, key = random.choice(TEXT_FMTS)
        value = random.choice(categories[key])
        outputs.append(fmt % value)

    return outputs


def make_card(name, contents, outfile):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    tex_lines = []

    tex_lines.append(r'\documentclass[10pt, a4paper]{article}')
    tex_lines.append(r'\usepackage{tikz}')
    tex_lines.append(r'\usepackage{fullpage}')
    tex_lines.append(r'\usetikzlibrary{positioning,matrix}')
    tex_lines.append(r'\renewcommand*{\familydefault}{\sfdefault}')
    tex_lines.append(r'\usepackage{array}')

    tex_lines.append(r'\begin{document}')
    tex_lines.append(r'\pagestyle{empty}')
    tex_lines.append(r'\begin{center}')

    tex_lines.append(r'\Huge ISMIR 2014 Mixer Bingo\\')
    tex_lines.append(r"\bigskip \huge \emph{%s} \\" % name)
    tex_lines.append(r'\normalsize')
    tex_lines.append(r'')
    tex_lines.append(r'\bigskip')

    random.shuffle(contents)
    c = contents[0:12] + [r'FREE'] + contents[12:24]

    tex_lines.append(r'\begin{tikzpicture}')

    tex_lines.append(r"""\tikzset{square matrix/.style={
    matrix of nodes,
    column sep=-\pgflinewidth, row sep=-\pgflinewidth,
    nodes={draw,
      text height=#1/2-2.5em,
      text depth=#1/2+2.5em,
      text width=#1,
      align=center,
      inner sep=0pt
    },
  },
  square matrix/.default=3.2cm
}""")

    tex_lines.append(r'\matrix [square matrix]')
    tex_lines.append(r'(shi)')
    tex_lines.append(r'{')

    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[0], c[1], c[2], c[3], c[4]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[5], c[6], c[7], c[8], c[9]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[10], c[11], c[12], c[13], c[14]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[15], c[16], c[17], c[18], c[19]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[20], c[21], c[22], c[23], c[24]))
    tex_lines.append(r'};')

    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-1-\i.north east) -- (shi-5-\i.south east);')
    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-1-\i.north west) -- (shi-5-\i.south west);')

    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-\i-1.north west) -- (shi-\i-5.north east);')
    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-\i-1.south west) -- (shi-\i-5.south east);')

    tex_lines.append(r'\end{tikzpicture}')
    tex_lines.append('')
    tex_lines.append(r'\pagebreak')
    tex_lines.append('')

    tex_lines.append(r'\end{center}')
    tex_lines.append(r'\end{document}')

    with open(outfile, 'w') as f:
        for line in tex_lines:
            f.write("%s\n" % line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a',
                        help='writeme')
    parser.add_argument('--b', type=str,
                        default='apple',
                        help='writeme')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to the console.')
    args = parser.parse_args()
    sys.exit(0)
