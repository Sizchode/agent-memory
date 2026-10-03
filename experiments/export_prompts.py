"""Export only AMOR's added relation normalization prompts for the ACL appendix."""

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/oscar/scratch/zliu328/agent-memory-outputs')
PREAMBLE = r'''% Add to the preamble. Keep the ACL document in two-column mode.
\usepackage{fvextra}
\usepackage{needspace}
\DefineVerbatimEnvironment{AMORPrompt}{Verbatim}{%
  fontsize=\footnotesize,
  breaklines=true,
  breakanywhere=true,
  frame=single,
  framesep=2mm,
  xleftmargin=0pt,
  xrightmargin=0pt}
'''


def collect():
    records = []
    for key, title, directory, code in [
        ('relation_names', 'Relation Naming', 'optimization_relation_schema_seed42_20260913', 'relation_schema.py'),
        ('relation_groups', 'Relation Equivalence', 'optimization_canonical_schema_seed42_20260913', 'canonicalize_schema.py')]:
        settings_path = BASE / directory / 'settings.json'
        source = ROOT / 'optimization/graph_construction' / code
        prompt = json.loads(settings_path.read_text())['prompt']
        node, = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in node.targets)]
        assert prompt == ast.literal_eval(node.value)
        records.append(dict(id=key, title=title, source=str(source), settings=str(settings_path),
                            system=prompt, user='<JSON array of relation IDs, labels, and example triples>'))
    return records


def export(output):
    records = collect()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'prompts.json').write_text(json.dumps(dict(
        scope='Only the two relation normalization prompts introduced for AMOR.',
        prompts=records, saved_run_settings_match=True), indent=2) + '\n')
    appendix = [PREAMBLE, r'''% Place the following after \appendix, which should occur only once.
\section{AMOR Prompts}
\label{app:prompts}
We provide AMOR's two offline relation normalization prompts.
Angle brackets indicate runtime inputs.
''']
    for record in records:
        appendix.append('\\Needspace{12\\baselineskip}\n\\subsection{' + record['title'] + '}\n')
        if record['id'] == 'relation_names':
            appendix.append('AMOR uses the canonical names; the requested cardinality and role fields do not control fact selection.\n')
        else:
            appendix.append('The second prompt compares relation labels within each embedding cluster.\n')
        appendix.append('\\begin{AMORPrompt}\n[SYSTEM]\n' + record['system']
                        + '\n\n[USER]\n' + record['user'] + '\n\\end{AMORPrompt}\n')
    (output / 'appendix_prompts.txt').write_text('\n'.join(appendix))
    print('Exported two AMOR prompts, each verified against saved run settings.')


def preview(output):
    """Compile in the unmodified ACL style, leaving no temporary TeX sources."""
    source = (output / 'appendix_prompts.txt').read_text()
    preamble, body = source.split('% Place the following after', 1)
    body = body[body.index('\\section{AMOR Prompts}'):]
    document = (r'\documentclass[11pt]{article}' + '\n'
                + r'\usepackage[preprint]{acl}' + '\n'
                + r'\usepackage[T1]{fontenc}\usepackage{times}' + '\n'
                + preamble + '\n' + r'\begin{document}\appendix' + '\n'
                + body + '\n' + r'\end{document}' + '\n')
    with tempfile.TemporaryDirectory(prefix='amor-acl-prompts-') as temporary:
        directory = Path(temporary)
        url = 'https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty'
        with urlopen(url, timeout=30) as response:
            (directory / 'acl.sty').write_bytes(response.read())
        (directory / 'prompts.tex').write_text(document)
        for _ in range(2):
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'prompts.tex'],
                                    cwd=directory, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-6000:])
        log = (directory / 'prompts.log').read_text()
        warnings = [line for line in log.splitlines() if 'Overfull' in line or 'Missing character' in line]
        shutil.copyfile(directory / 'prompts.pdf', output / 'appendix_prompts_preview.pdf')
        (output / 'prompt_layout_check.json').write_text(json.dumps(dict(
            style=url, columns=2, prompt_width='columnwidth', engine='pdflatex',
            warnings=warnings, source_files=['appendix_prompts.txt']), indent=2) + '\n')
        print('ACL preview compiled; layout warnings:', warnings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    export(args.output)
    if args.preview:
        preview(args.output)
