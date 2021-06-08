import sys
sys.path.append('/voice/preprocessing')
# sys.path.insert(0, '/voice')
from kor_transliterator import Transliterator
import yaml

with open('config.yaml') as f:
    config = yaml.load(f)

output_txt = config['output_txt']

with open(output_txt, 'r') as f:
    text = f.read()

result = Transliterator(text)
result = result.transliterate()

with open(output_txt, 'w') as f:
    f.write(result)

