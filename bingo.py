import random


def make_card(name, contents, outfile):

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

    tex_lines.append(r"%s & %s & %s & %s & %s\\" % (c[0], c[1], c[2], c[3], c[4]))
    tex_lines.append(r"%s & %s & %s & %s & %s\\" % (c[5], c[6], c[7], c[8], c[9]))
    tex_lines.append(r"%s & %s & %s & %s & %s\\" % (c[10], c[11], c[12], c[13], c[14]))
    tex_lines.append(r"%s & %s & %s & %s & %s\\" % (c[15], c[16], c[17], c[18], c[19]))
    tex_lines.append(r"%s & %s & %s & %s & %s\\" % (c[20], c[21], c[22], c[23], c[24]))
    tex_lines.append(r'};')

    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(r'\draw[line width=2pt] (shi-1-\i.north east) -- (shi-5-\i.south east);')
    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(r'\draw[line width=2pt] (shi-1-\i.north west) -- (shi-5-\i.south west);')

    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(r'\draw[line width=2pt] (shi-\i-1.north west) -- (shi-\i-5.north east);')
    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(r'\draw[line width=2pt] (shi-\i-1.south west) -- (shi-\i-5.south east);')

    tex_lines.append(r'\end{tikzpicture}')
    tex_lines.append('')
    tex_lines.append(r'\pagebreak')
    tex_lines.append('')

    tex_lines.append(r'\end{center}')
    tex_lines.append(r'\end{document}')

    with open(outfile, 'w') as f:
        for line in tex_lines:
            f.write("%s\n" % line)
