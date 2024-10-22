import hashlib

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

def generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False

tqdm = tqdm_notebook if is_notebook() else tqdm