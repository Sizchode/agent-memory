"""Read-only vector validation and the selected vLLM runtime's tool lookup."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from experiments.encoder_robustness import digest, json_save, validate_bundle
from experiments.encoder_robustness_service import server_environment


def main(root):
    runtime_bin = Path('/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin')
    environment = server_environment(runtime_bin / 'vllm')
    resolved = shutil.which('ninja', path=environment['PATH'])
    if resolved != str(runtime_bin / 'ninja'):
        raise RuntimeError(f'Wrong runtime tool: {resolved}')
    command = [str(runtime_bin / 'python'), '-c',
               "import subprocess; subprocess.run(['ninja', '--version'], check=True)"]
    child = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    print('selected_runtime_ninja', child.stdout.strip(), flush=True)
    sources = json.loads((root / 'sources.json').read_text())
    assert sources['complete']
    audits = []
    for entry in sources['entries']:
        source_path = Path(entry['directory']) / 'source.json'
        source = json.loads(source_path.read_text())
        for encoder in ('qwen', 'bge', 'nv'):
            directory = root / 'vectors' / encoder / entry['task'] / entry['group']
            marker = json.loads((directory / 'complete.json').read_text())
            source_hash, vector_hash = digest(source_path), digest(directory / 'embeddings.npz')
            assert marker['source_sha256'] == source_hash
            assert marker['embeddings_sha256'] == vector_hash
            with np.load(directory / 'embeddings.npz', allow_pickle=False) as bundle:
                validate_bundle(source, bundle)
                dimension = bundle['fact'].shape[1]
            audit = dict(encoder=encoder, task=entry['task'], group=entry['group'],
                         source_sha256=source_hash, embeddings_sha256=vector_hash,
                         dimension=dimension, questions=len(source['query_ids']))
            audits.append(audit)
            print(json.dumps(audit), flush=True)
    target = root / 'jobs' / os.environ['SLURM_JOB_ID'] / 'recovery_preflight.json'
    json_save(target, dict(complete=True, vectors=audits, ninja=resolved,
                          ninja_version=child.stdout.strip(),
                          service_sha256=digest(Path(__file__).with_name('encoder_robustness_service.py'))))
    print('recovery_preflight_complete', len(audits), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
