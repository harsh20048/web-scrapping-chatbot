import shutil
import os

chroma_dir = os.path.join('data', 'chroma')
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)
    print('Deleted data/chroma directory.')
else:
    print('data/chroma directory does not exist.')
