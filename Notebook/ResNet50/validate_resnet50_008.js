const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const { spawnSync } = require('node:child_process');

const dir = __dirname;
const read = name => JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'));
const old = read('resnet50_5seed_006.ipynb');
const reference = read('resnet50_5seed_reference_recipe.ipynb');
const next = read('resnet50_5seed_008.ipynb');
const main = read('resnet50_5seed.ipynb');
const source = (nb, index) => nb.cells[index].source.join('');
const encoded = source(next, 35).match(/NOTEBOOK_SOURCE_ZLIB_B64 = "([A-Za-z0-9+/=]+)"/)[1];
const fallback = JSON.parse(zlib.inflateSync(Buffer.from(encoded, 'base64')));

const checks = {
  previousPreserved: source(old, 3).includes('NOTEBOOK_VERSION = "006"'),
  referencePreserved: source(reference, 4).includes('NOTEBOOK_VERSION = "007"'),
  same008Copies: JSON.stringify(next) === JSON.stringify(main),
  fiveSeeds: source(next, 4).includes('SEEDS = [42, 123, 2026, 3407, 7777]'),
  dense011Recipe: source(next, 4).includes('HEAD_DROPOUT = 0.30') &&
    source(next, 4).includes('INITIAL_LR = 1e-4') &&
    source(next, 4).includes('BATCH_SIZE = 16') &&
    source(next, 12).includes('"rotation_factor": 0.04') &&
    source(next, 12).includes('"zoom_factor": 0.05') &&
    source(next, 12).includes('weights=None'),
  resnetModelUnchanged: source(next, 12) === source(reference, 12),
  outputFree: next.cells.every(c => c.cell_type !== 'code' ||
    (c.outputs.length === 0 && c.execution_count === null)),
  curvesAndMatrices: source(next, 21).includes('rolling(3, center=True, min_periods=1)') &&
    source(next, 21).includes('fontweight="bold", color="black"') &&
    source(next, 21).includes('validation_per_class_f1_5seeds_panel.png') &&
    source(next, 26).includes('test_per_class_f1_5seeds_panel.png'),
  sourceInBothZips: source(next, 35).includes('NOTEBOOK_COPY.write_bytes') &&
    source(next, 35).includes('".ipynb"'),
  embeddedFallback: fallback.cells.length === next.cells.length &&
    source(fallback, 4).includes('NOTEBOOK_VERSION = "008"'),
};
for (const [name, pass] of Object.entries(checks)) console.log(`${pass ? 'PASS' : 'FAIL'} ${name}`);
if (Object.values(checks).some(pass => !pass)) process.exitCode = 1;

const python = path.join(dir, '..', '..', '.venv', 'Scripts', 'python.exe');
if (fs.existsSync(python)) {
  const checkPython = `import json,sys
nb=json.load(open(sys.argv[1],encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    code=''.join(c['source'])
    code='\\n'.join('pass  # notebook shell command' if line.lstrip().startswith('!') else line for line in code.split('\\n'))
    try: compile(code, f'cell_{i}', 'exec')
    except SyntaxError as exc:
        print(f'FAIL Python syntax: cell {i}: {exc}')
        sys.exit(1)
print('PASS Python syntax: every code cell')`;
  const result = spawnSync(python, ['-c', checkPython,
    path.join(dir, 'resnet50_5seed_008.ipynb')], { encoding: 'utf8' });
  process.stdout.write(result.stdout);
  process.stderr.write(result.stderr);
  if (result.status !== 0) process.exitCode = 1;
}
