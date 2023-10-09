Search.setIndex({"docnames": ["README", "intro", "lecture10", "lecture3", "lecture4", "lecture5", "lecture6", "lecture7", "lecture8", "lecture9", "mandatory1"], "filenames": ["README.md", "intro.md", "lecture10.ipynb", "lecture3.ipynb", "lecture4.ipynb", "lecture5.ipynb", "lecture6.ipynb", "lecture7.ipynb", "lecture8.ipynb", "lecture9.ipynb", "mandatory1.ipynb"], "titles": ["matmek4270-book", "Lecture Notes MATMEK-4270", "Lecture 9b", "Lecture 3", "Lecture 4", "Lecture 5", "Lecture 6", "Lecture 7", "Lecture 8", "Lecture 9", "<span class=\"section-number\">1. </span>Mandatory assignment due 16/10-2023"], "terms": {"link": [0, 9], "note": [0, 2, 3, 4, 5, 6, 7, 8, 9], "notebook": [0, 5, 7, 10], "us": [0, 2, 3, 5, 6, 7, 8, 9, 10], "creat": [0, 2, 3, 4, 6, 7, 8, 9], "thi": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ar": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "found": [0, 4, 7], "repositori": [0, 10], "abov": [0, 2, 3, 4, 5, 6, 7, 8, 9], "howev": [0, 2, 3, 4, 5, 6, 7, 8, 9], "all": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "store": [0, 5, 6, 7], "non": [0, 2, 4], "execut": 0, "clear": 0, "state": [0, 7], "so": [0, 2, 3, 4, 5, 6, 7, 8, 9], "order": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "you": [0, 2, 6, 7, 8, 9, 10], "should": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "either": [0, 8, 10], "clone": 0, "fork": [0, 10], "somewher": 0, "can": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "collect": [1, 2, 8], "3": [1, 2, 4, 5, 6, 7, 8, 9, 10], "4": [1, 2, 3, 5, 6, 7, 8, 9, 10], "5": [1, 2, 3, 4, 6, 7, 8, 9, 10], "6": [1, 2, 3, 4, 7, 8, 9, 10], "7": [1, 2, 4, 5, 6, 8, 9, 10], "8": [1, 2, 3, 4, 6, 7, 9], "due": [1, 8], "16": [1, 3, 4, 6, 7, 8, 9], "10": [1, 2, 3, 5, 6, 7, 8, 9], "2023": 1, "begin": [2, 3, 4, 5, 6, 7, 8, 9, 10], "u": [2, 3, 4, 5, 6, 7, 8, 9, 10], "omega": [2, 3, 4, 6, 7, 8, 9, 10], "2": [2, 3, 4, 5, 6, 7, 8, 9, 10], "0": [2, 3, 4, 5, 6, 7, 8, 9, 10], "i": [2, 3, 4, 5, 6, 7, 8, 9, 10], "end": [2, 3, 4, 6, 7, 8, 9, 10], "implement": [2, 3, 4, 5, 6, 7, 8, 9], "align": [3, 4, 5, 6, 7, 8, 9], "1": [2, 3, 4, 5, 6, 7, 8, 9, 10], "frac": [2, 3, 4, 5, 6, 7, 8, 9, 10], "delta": [2, 3, 4, 5, 6, 7, 9, 10], "t": [3, 4, 5, 6, 7, 9, 10], "n": [2, 3, 4, 5, 6, 7, 8, 9, 10], "2u": [3, 4, 5, 7, 10], "ldot": [2, 3, 4, 5, 6, 7, 8, 9, 10], "n_t": [3, 4, 5, 7, 10], "import": [2, 3, 4, 5, 6, 7, 8, 9], "numpi": [2, 3, 4, 5, 6, 7, 8, 9], "np": [2, 3, 4, 5, 6, 7, 8, 9], "matplotlib": [2, 3, 4, 5, 6, 7, 8, 9], "pyplot": [2, 3, 4, 5, 6, 7, 8, 9], "plt": [2, 3, 4, 5, 6, 7, 8, 9], "def": [3, 5, 6, 7, 8, 9], "dt": [3, 4, 5, 7], "w": [3, 5, 9, 10], "35": 3, "solv": [2, 3, 5, 7, 8, 9, 10], "eq": [3, 4, 6, 7, 8, 9, 10], "paramet": [3, 5, 7, 8], "float": [3, 7, 8, 9], "time": [2, 3, 4, 5, 6, 8, 9, 10], "step": [3, 4, 5, 7], "option": [3, 5], "model": [3, 4], "return": [3, 5, 6, 7, 8, 9], "array_lik": 3, "discret": [2, 3, 7, 9, 10], "The": [2, 3, 5, 9], "nt": [3, 5, 7], "int": [3, 5], "round": 3, "zero": [2, 3, 4, 5, 6, 7, 8, 9], "linspac": [2, 3, 4, 5, 6, 7, 8, 9], "rang": [2, 3, 4, 5, 6, 7, 8, 9], "u_exact": 3, "exact": [3, 4, 6, 7, 8, 9], "arrai": [2, 3, 4, 5, 6, 7, 8, 9], "comput": [2, 3, 4, 5, 7, 8, 9, 10], "ue": [3, 6, 7, 9, 10], "co": [2, 3, 4, 6, 7, 8, 9, 10], "we": [2, 3, 4, 5, 6, 7, 8, 9, 10], "now": [2, 3, 4, 5, 6, 7, 8, 9], "want": [2, 3, 7, 8, 9], "converg": [3, 8, 9, 10], "rate": 3, "ell": [3, 7, 8, 10], "norm": [3, 4, 7, 8, 9, 10], "e_i": 3, "sqrt": [3, 4, 6, 7, 8, 9, 10], "t_i": [3, 9], "sum_": [2, 3, 4, 5, 6, 7, 8, 9, 10], "_t": 3, "u_": [2, 3, 6, 7, 8, 9, 10], "t_n": [3, 4, 5, 7, 9, 10], "given": [3, 4, 5, 10], "uniform": [2, 3, 5, 7, 8, 9, 10], "mesh": [2, 3, 4, 5, 8, 9, 10], "level": 3, "For": [2, 3, 4, 5, 6, 7, 8, 9, 10], "exampl": [2, 3, 4, 5, 6, 7, 8, 9], "0_t": 3, "2_t": 3, "32": [3, 9], "etc": [3, 8], "t_": [3, 4, 9], "t_2": [3, 9], "interv": [2, 3, 5, 7, 8], "point": [2, 3, 4, 5, 6, 7, 8, 9, 10], "assum": [2, 3, 5, 7, 9, 10], "error": [3, 4, 8, 9, 10], "written": [3, 4, 5, 6, 7, 8, 9], "c": [2, 3, 4, 5, 6, 7, 9, 10], "r": [3, 4, 6, 7, 8, 9], "where": [2, 3, 4, 5, 6, 7, 8, 9, 10], "constant": [3, 4, 6, 7, 8, 9], "wai": [2, 3, 7, 9], "have": [2, 3, 4, 5, 6, 7, 8, 9], "two": [2, 3, 4, 5, 8, 10], "e_": [3, 10], "left": [3, 4, 5, 6, 7, 8, 9, 10], "right": [3, 4, 5, 6, 7, 8, 9, 10], "isol": [3, 4], "log": 3, "differ": [2, 3, 7, 8, 9, 10], "find": [2, 3, 6, 7, 8, 9, 10], "let": [2, 3, 4, 5, 6, 7, 8, 9], "s": [3, 4, 7, 8, 9], "first": [3, 5, 6, 7, 8, 9], "write": [3, 5, 6, 7, 8, 9], "function": [3, 4, 5, 10], "defin": [2, 3, 5, 6, 7, 8, 9, 10], "l2_error": [3, 7, 9], "sol": [3, 5], "l2": [3, 7, 8, 10], "result": [3, 5, 6, 7, 8, 9, 10], "from": [2, 3, 4, 5, 6, 7, 8, 9, 10], "callabl": 3, "sum": [3, 6, 7, 8, 9], "convergence_r": 3, "m": [2, 3, 4, 6, 7, 8, 9], "dt0": 3, "30": [3, 6, 9], "num_period": 3, "empir": 3, "estim": 3, "base": [3, 7, 9], "simul": [3, 5], "halv": 3, "each": [2, 3, 5, 6, 7, 8, 9, 10], "number": [3, 5, 6, 7, 8, 9, 10], "per": [3, 9], "period": 3, "coarsest": 3, "size": [3, 4, 6, 7], "domain": [2, 3, 4, 5, 7, 8, 9, 10], "2pi": 3, "p": [3, 6, 8, 9], "pi": [2, 3, 4, 5, 6, 8, 9, 10], "dt_valu": 3, "e_valu": 3, "e": [2, 3, 4, 6, 7, 8, 10], "append": [2, 3, 7, 8, 9], "same": [2, 3, 4, 5, 6, 7, 8, 9, 10], "test": [3, 7, 8, 9], "0036366687367093": 3, "000949732819596": 3, "000240106059534": 3, "13526035155205302": 3, "033729955960525936": 3, "008426939670390836": 3, "0021063843253281167": 3, "5983986006837702": 3, "2991993003418851": 3, "14959965017094254": 3, "07479982508547127": 3, "These": [2, 3, 4, 5, 6, 7, 8], "last": [3, 4, 5, 6, 8, 9], "list": [3, 5, 6], "timestep": [3, 5, 10], "af": 3, "plot": [2, 3, 4, 5, 6, 7, 8, 9], "show": [3, 4, 5, 6, 10], "scheme": [3, 4, 5, 7], "second": [3, 4, 5, 6, 7, 8, 9], "loglog": [3, 9], "titl": 3, "finit": [3, 7, 8, 9, 10], "legend": [2, 3, 4, 7, 8, 9], "mathcal": [3, 4, 9], "o": [3, 4, 9], "plotslop": 3, "slope_mark": 3, "see": [3, 5, 6, 7, 8, 9, 10], "ha": [3, 4, 5, 6, 7, 8, 9], "slope": [2, 3], "accuraci": [2, 3, 4, 5, 7, 8, 9], "test_ord": 3, "assert": [3, 9], "allclos": [3, 9], "atol": 3, "1e": [3, 9], "trivial": [3, 4], "verifi": [3, 6, 7, 9, 10], "work": [2, 3, 6, 7, 8, 9, 10], "satisfi": [3, 8, 10], "boundari": [2, 3, 4, 10], "condit": [3, 4, 7, 8, 10], "ne": [2, 3, 6, 7, 8], "need": [2, 3, 4, 5, 6, 7, 8, 9, 10], "modifi": [3, 4, 5, 6, 7, 10], "sourc": [3, 4], "term": [3, 4, 7, 9], "sinc": [2, 3, 5, 6, 7, 8, 9], "case": [2, 3, 6, 7, 8, 9], "get": [2, 3, 4, 5, 6, 7, 8, 9, 10], "also": [2, 3, 4, 5, 6, 7, 8, 9, 10], "know": [2, 3, 4, 6, 7, 8], "accur": [3, 4, 5, 7, 8], "polynomi": [2, 3, 8], "check": [3, 6, 7, 8, 9], "uc": [3, 8, 9], "numer": [3, 4, 5, 7, 8, 9, 10], "initi": [3, 7, 10], "still": [2, 3, 4, 7, 8], "ok": [3, 5, 8, 9], "becom": [3, 6, 7, 8, 9], "12": [3, 4, 5, 6, 7, 8, 9], "fourth": 3, "print": [3, 4, 6, 7, 8, 9], "02": [3, 4, 8], "take": [3, 4, 5, 6, 7, 8, 9, 10], "closer": 3, "look": [2, 3, 4, 5, 6, 7, 9], "differec": 3, "express": [3, 4, 6, 9], "come": [2, 3, 6, 8], "taylor": [3, 4], "expans": [3, 4, 8, 9], "around": [3, 4, 5, 9], "24": [3, 4, 6, 7, 9], "cdot": [3, 4, 5, 6, 7, 8, 9], "ad": [3, 4, 6, 7, 8], "There": [2, 3, 5, 6, 7, 8, 9, 10], "proport": [3, 4], "our": [3, 4, 6, 7, 8], "put": [3, 6], "residu": 3, "go": [3, 7], "back": [3, 6], "correspond": [3, 6, 7, 9], "exactli": [2, 3, 5, 6, 7, 8, 9], "It": [2, 3, 4, 5, 6, 7, 8, 9], "becaus": [2, 3, 4, 5, 6, 7, 8, 9], "next": [2, 3, 5, 7], "sixth": 3, "deriv": [3, 5, 8], "which": [2, 3, 4, 5, 6, 7, 8, 9], "adj_solv": 3, "r2": 3, "e2": [3, 4], "dt2": 3, "divid": [2, 4, 5], "1d": [2, 4, 6, 7, 9], "line": [2, 4, 6, 7, 8, 9], "onli": [2, 4, 5, 6, 7, 8, 9], "specif": [4, 7], "locat": [4, 6, 7], "node": [2, 4, 5, 6, 7], "A": [2, 4, 5, 6, 7, 8, 9, 10], "up": [2, 4, 6, 7, 8], "until": [2, 4, 7, 8], "recurs": [4, 5, 7, 9], "veri": [2, 4, 5, 6, 7, 8, 9], "easi": [4, 6, 7, 9], "intuit": 4, "loop": [4, 5, 6, 7, 9], "most": [4, 5, 6, 7, 9], "common": [4, 7, 8, 9], "through": [4, 5, 6, 7, 9], "explicit": [4, 7, 9], "assembl": [2, 4, 6, 9], "consid": [4, 5, 6, 7, 8, 9, 10], "decai": [4, 9], "au": 4, "solut": [4, 5, 6, 7, 8, 9], "vector": [4, 5, 7, 8, 9], "boldsymbol": [2, 4, 6, 7, 8, 9, 10], "To": [4, 6, 7, 9], "start": [4, 5, 6, 7, 9, 10], "set": [4, 5, 6, 7, 8, 9], "theta": [4, 9], "rearrang": [4, 7, 8], "algorithm": [4, 5, 7], "set_printopt": [4, 9], "precis": [4, 7, 9], "te": 4, "1001": 4, "b": [4, 5, 6, 7, 8, 9], "exp": [4, 5, 6, 7, 8, 9], "k": [4, 5, 6, 7, 8, 9, 10], "text": [2, 4, 5, 6, 7, 8, 9], "82": 4, "never": [4, 9], "linear": [2, 4, 6, 7, 8, 9], "just": [2, 4, 5, 6, 7, 8, 9], "forward": [4, 5, 7, 9], "simpli": [4, 5, 6, 7, 8, 9], "an": [2, 4, 5, 6, 7, 8, 9, 10], "formul": [4, 8, 9], "mathbb": [4, 6, 8, 9], "coeffici": [4, 6, 8, 9], "bmatrix": [4, 5, 6, 7, 9], "algebra": [4, 5, 6, 7, 8, 9], "gaussian": [4, 9], "elimin": 4, "system": [4, 8], "like": [2, 4, 5, 6, 7, 8, 9], "underbrac": [4, 6, 9], "_": [2, 4, 5, 6, 7, 8, 9, 10], "9": [1, 4, 6, 7, 8], "notic": [4, 5, 6, 8, 9], "row": [4, 5, 7, 8, 9], "remain": [2, 4, 9, 10], "scipi": [4, 5, 6, 7, 8, 9], "spars": [2, 4, 5, 7, 9], "packag": [4, 6], "diag": [4, 5, 6, 7, 9], "full": 4, "ones": [4, 6, 9], "csr": 4, "toarrai": [4, 6], "333": 4, "un": [4, 5, 7, 8], "linalg": [4, 6, 9], "spsolve_triangular": 4, "lower": [4, 8], "true": [4, 6, 7, 9], "unit_diagon": 4, "000e": 4, "00": [4, 8], "333e": 4, "01": 4, "111e": 4, "704e": 4, "235e": 4, "115e": 4, "03": [4, 8], "372e": 4, "572e": 4, "04": [4, 8], "524e": 4, "befor": [4, 7, 8], "quad": [2, 4, 5, 6, 7, 8, 9, 10], "central": [4, 5, 6, 7, 10], "triangular": 4, "upper": [4, 7], "especi": [4, 6], "quick": 4, "backward": [4, 9], "substitut": [4, 9], "here": [4, 5, 6, 7, 8, 9, 10], "specifi": [4, 5, 7], "one": [2, 4, 5, 6, 7, 8, 9], "unknown": [2, 4, 5, 6, 7, 8, 9], "more": [2, 4, 5, 6, 7, 8, 9], "orderli": 4, "obtain": [4, 5, 7], "follow": [4, 5, 6, 7, 8, 9], "three": [4, 5, 7, 8, 9, 10], "h": [4, 7, 9, 10], "2h": [4, 7], "3h": 4, "27": [4, 6, 9], "rememb": [4, 9], "simplic": [2, 4, 5, 6, 7, 8, 9], "oper": [4, 7, 9], "product": [4, 5, 7, 8, 9], "d": [4, 5, 6, 7, 8, 9], "repres": [4, 5, 6, 7, 8, 9], "approxim": [4, 10], "vdot": [4, 5, 6, 7, 8, 9], "ddot": [4, 5, 6, 7], "open": [4, 7], "requir": [2, 4, 5, 6, 7, 8, 9], "subtract": 4, "do": [4, 5, 6, 7, 8, 9], "better": [4, 5, 7, 8, 9], "ye": 4, "cours": 4, "add": [4, 6, 8], "both": [2, 4, 5, 6, 7, 8, 9, 10], "don": 4, "worri": 4, "about": [4, 6, 7, 8, 9, 10], "how": [2, 4, 5, 6, 7, 8, 9, 10], "yet": [4, 7, 9], "4u": 4, "5u": 4, "11": [4, 6, 9], "lead": [4, 5, 6, 7, 8, 9, 10], "side": [4, 6, 7, 8, 9, 10], "python": [4, 5, 6, 7, 8, 9, 10], "d2": [4, 5, 6, 7], "lil": [4, 5, 6, 7], "fix": [4, 6, 8], "forget": 4, "If": [4, 5, 6, 7, 8, 9], "appli": [2, 4, 5, 6, 7, 8, 9], "f": [2, 4, 5, 6, 7, 8, 9, 10], "try": [4, 5, 6, 7, 8, 9], "25": [4, 6, 7, 8, 9], "d2f": 4, "edg": [4, 5, 8, 9], "d2e": 4, "what": [4, 5, 6, 7, 9], "happen": [4, 6], "why": [4, 7], "perfect": [4, 8], "reason": [4, 7], "henc": [4, 5, 7, 8], "even": [2, 4, 6, 7, 8, 9], "though": [4, 7, 8, 9], "complex": [4, 8], "would": [2, 4, 6, 7, 9], "sin": [4, 5, 6, 7, 8, 9, 10], "d2fe": 4, "d2f1": 4, "2nd": 4, "1st": 4, "0x117158310": [], "similar": [4, 5, 7, 9], "skew": 4, "again": [4, 6, 7, 8], "mere": [4, 5, 7], "3u": 4, "d1": [4, 7], "d1fe": 4, "d1f": 4, "0x117fee4d0": [], "equal": [4, 5, 6, 7, 8, 9, 10], "d2n": 4, "75": [4, 7], "en": 4, "16216463185914665": 4, "37045090252771984": 4, "shown": [2, 4, 5, 6, 7, 8, 9], "itself": [4, 5, 7], "great": 4, "thei": [4, 6, 8, 9], "depend": [4, 5, 7, 8], "mai": [2, 4, 5, 6, 7, 8, 9], "onc": [4, 6, 7], "reus": [2, 4, 7, 8], "accord": [4, 8], "ident": [4, 6, 7, 8, 9], "item": [4, 5, 6, 7, 9], "id": 4, "ey": [4, 6], "u1": [4, 8], "spsolv": [4, 6], "line2d": 4, "0x118063e90": [], "0x1180b4e50": [], "0x1180b5150": [], "fulli": 4, "implicit": 4, "neighbour": [4, 5, 7], "everi": [4, 5, 7], "stabl": [4, 7], "than": [2, 4, 6, 7, 8, 9, 10], "possibl": [2, 4, 5, 6, 7, 8, 9], "ani": [2, 4, 5, 6, 7, 8, 9], "posit": [4, 9], "neg": 4, "direct": [4, 6, 7, 9, 10], "x": [2, 4, 5, 6, 7, 8, 9, 10], "x_0": [2, 4, 6, 8], "read": [4, 6, 7, 10], "dx": [4, 5, 6, 7, 8, 9], "With": [4, 7, 8, 9], "evalu": [4, 5, 6, 7, 8, 9], "certain": [4, 8], "That": [4, 6, 8], "mh": 4, "integ": [4, 6, 8, 9, 10], "usual": [4, 5, 7, 8, 9], "notat": [4, 5, 6, 8, 9, 10], "indic": [4, 6, 9], "respect": [4, 5, 6], "c_": 4, "mi": 4, "du_i": 4, "neglect": [4, 9], "du": [4, 7], "m_0": 4, "form": [2, 4, 7, 8, 9], "du_": 4, "du_1": 4, "du_2": 4, "du_3": 4, "du_4": 4, "easili": [4, 6, 7, 8, 9, 10], "re": [4, 8], "normal": [2, 4, 6, 9], "interest": [2, 4, 7, 8], "And": [2, 4, 5, 6, 7, 8, 9], "In": [2, 4, 5, 6, 7, 8], "sympi": [2, 4, 5, 6, 7, 8, 9, 10], "sp": [2, 4, 5, 6, 7, 8, 9], "symbol": [2, 4, 5, 6, 7, 8, 9], "displaystyl": [4, 7, 8], "invers": [4, 9], "inv": 4, "out": [4, 5, 6, 7, 8], "2i": [4, 8, 9], "coef": 4, "inli": 4, "thu": [4, 6, 7, 8, 9], "partial": [5, 8, 9, 10], "differenti": [5, 6, 7, 9], "pde": [5, 7, 8], "space": [2, 5, 7, 8, 9, 10], "l": [5, 7, 8, 9, 10], "hyperbol": [5, 7], "ct": 5, "reflect": [5, 7, 10], "chang": [2, 5, 8, 9, 10], "sign": [5, 9], "nonzero": [2, 5], "without": [5, 8, 9, 10], "pass": [5, 10], "undisturb": 5, "unreflect": 5, "illustr": [5, 6, 7], "nice": [5, 9], "turn": [5, 6, 8], "puls": 5, "off": [5, 6, 7], "damp": 5, "press": 5, "green": [5, 6, 7, 10], "button": 5, "ipython": [5, 7, 8], "displai": [5, 7, 8], "ifram": 5, "http": 5, "phet": 5, "colorado": 5, "edu": 5, "sim": 5, "html": [5, 7], "string": 5, "latest": 5, "string_al": 5, "width": 5, "600": 5, "height": 5, "400": [5, 7], "v": [2, 5, 6, 8, 9], "some": [2, 5, 7, 8, 9], "simplest": [2, 5], "x_j": [5, 6, 7, 8, 9], "j": [2, 5, 6, 7, 8, 9, 10], "grid": [5, 7, 8], "below": [2, 5, 6, 7, 8, 9], "xlabel": 5, "ylabel": 5, "ko": [5, 6, 7, 9], "15": [5, 6, 7, 8, 9, 10], "x_2": [2, 5, 7, 8], "t_3": [5, 9], "n_j": 5, "valu": [5, 6, 7, 8, 9], "later": [5, 7, 8], "n_0": 5, "n_1": 5, "n_n": 5, "_j": [2, 5, 7, 8, 9], "n_": [5, 7, 10], "stencil": [5, 7], "make": [2, 5, 6, 7, 8, 9, 10], "fig": [2, 5, 6, 7, 9], "figur": [2, 5, 6, 7, 8, 9], "figsiz": [2, 5, 6, 7, 8, 9], "ax": [2, 5, 6, 7, 8, 9], "gca": [5, 6, 7, 8], "get_xaxi": 5, "set_vis": 5, "fals": [5, 7], "get_yaxi": 5, "axi": [5, 6, 7], "basic": 5, "chosen": [5, 6, 7, 8, 9], "storag": 5, "total": [2, 5, 6, 9], "wherea": [5, 6, 7, 8, 9], "alwai": [2, 5, 8, 9], "known": [5, 6, 7, 9], "underlin": 5, "courant": 5, "stabil": [5, 7], "matrix": [2, 5, 7, 8, 9, 10], "simplifi": [5, 7, 9], "doe": [5, 6, 7, 8, 9], "includ": [5, 8, 10], "formula": [2, 5, 6, 9], "th": 5, "ji": [5, 9], "except": [5, 6, 9], "modif": [5, 8], "inner": [5, 8, 9], "singl": [5, 6, 7], "addit": [5, 7, 9], "simpl": [2, 5, 6, 7, 8, 9], "tricki": 5, "part": [2, 5, 7, 8], "complet": [5, 7, 8, 10], "updat": [5, 7], "move": [5, 6, 7, 10], "leftarrow": [5, 7], "swap": [5, 7, 9], "readi": [5, 6, 7], "At": [5, 6, 9], "class": [2, 5, 7, 8, 9, 10], "wave1d": 5, "assign": 5, "other": [2, 5, 6, 8, 9], "reader": 5, "advis": 5, "studi": 5, "detail": 5, "hold": [5, 9], "100": [2, 5, 7, 8, 9], "spatial": [5, 7, 9, 10], "unp1": [5, 7], "unm1": [5, 7], "altern": [5, 6, 9], "actual": [5, 6, 8, 9], "further": [5, 6, 7], "insert": [5, 7, 8, 9, 10], "__call__": 5, "sever": [5, 7, 8], "seen": [2, 5, 6], "entir": [2, 5, 6, 7, 8], "apply_bc": 5, "bc": 5, "scale": 5, "after": [5, 7], "n_2": 5, "multipli": 5, "scalar": [5, 8], "_0": [5, 8, 9], "_n": [5, 9], "intern": [2, 5, 7, 10], "matter": [2, 5, 6, 7, 8], "whatev": 5, "overwritten": 5, "necessari": [5, 6, 7], "suffici": 5, "them": [5, 6, 7, 8, 9], "bit": 5, "homogen": [5, 10], "outsid": [5, 7], "togeth": [5, 6], "similarili": 5, "valid": [5, 6, 8], "d_n": 5, "word": [2, 5], "anyth": [5, 6, 8], "directli": [5, 6, 9], "approach": [2, 5, 7, 8, 9], "give": [5, 6, 8], "unfortun": [2, 5], "overlin": [5, 7], "mean": [2, 5, 6, 7, 8, 9], "repeat": [5, 6, 9], "indefinit": 5, "sine": [5, 8], "800": 5, "fi": 5, "red": [5, 6, 7], "dot": [5, 7], "earlier": 5, "been": [5, 6, 7], "wrap": [5, 9], "consequ": 5, "noth": [2, 5], "ordinari": [5, 7], "d_p": 5, "_p": 5, "l0": [2, 5], "extent": 5, "c0": [5, 6], "wavespe": 5, "cfl": [5, 7, 10], "u0": [5, 7, 8], "__init__": [5, 10], "self": 5, "200": 5, "paramt": 5, "bake": 5, "rais": 5, "notimplementederror": 5, "elif": 5, "none": [5, 9], "provid": [5, 8, 9], "els": [5, 8, 9], "runtimeerror": 5, "wrong": [5, 8], "properti": [5, 10], "ic": 5, "save_step": 5, "u_t": 5, "save": [5, 6, 7, 10], "dictionari": 5, "kei": 5, "length": [5, 6, 9, 10], "lambdifi": [5, 6, 7, 8, 9], "sub": [5, 7, 8, 9], "plotdata": [5, 7], "copi": [5, 7], "plot_with_offset": 5, "data": [5, 7], "nd": 5, "len": [5, 7, 8, 9], "v0": 5, "ab": 5, "max": [5, 7], "facecolor": 5, "add_subplot": 5, "111": 5, "lw": 5, "zorder": 5, "fill_between": 5, "choos": [5, 6, 7, 8, 9], "uniti": 5, "averi": 5, "offset": 5, "flip": 5, "when": [2, 5, 6, 7, 8, 9, 10], "reach": 5, "within": [2, 6], "nabla": [6, 7, 10], "coordin": [6, 8, 9], "y": [2, 6, 7, 9, 10], "rectangular": 6, "l_x": [6, 7], "l_y": [6, 7], "dirichlet": [6, 7], "rectangl": [2, 6], "color": [2, 6], "blue": [6, 7], "magenta": 6, "likewis": 6, "consist": [6, 8], "corner": 6, "belong": 6, "along": [6, 7], "easiest": 6, "g": [6, 7, 8, 9], "arrow": 6, "head_width": 6, "rotat": 6, "vertic": 6, "x_i": [2, 6, 7, 8, 9, 10], "n_x": [6, 7], "y_j": [6, 7, 9, 10], "n_y": [6, 7], "x_1": [2, 6, 8], "x_": [2, 6], "y_0": [6, 7, 9], "y_1": 6, "y_": 6, "explain": 6, "itertool": 6, "uxv": 6, "contain": [2, 6, 8, 9], "pair": 6, "describ": [6, 7, 8, 10], "mathemat": 6, "well": [6, 7, 8, 9], "combin": 6, "alreadi": [6, 7, 9, 10], "realli": [6, 7, 9], "grei": 6, "dy": [6, 7, 9], "18": [6, 7, 9], "07": 6, "55": [6, 7], "drop": 6, "front": [6, 9], "xy": [6, 9], "present": 6, "shape": [6, 7, 9], "doubl": [6, 7, 9], "natur": [6, 7, 9], "horizontalalign": [6, 7], "center": [6, 7], "wa": [6, 7, 9, 10], "column": [6, 7, 8, 9], "counterintuit": 6, "think": 6, "a_": [6, 8, 9], "ij": [2, 6, 7, 8, 9, 10], "index": [6, 7, 8, 9], "complic": [2, 6, 7, 8], "factor": [6, 8, 9], "awar": 6, "But": [2, 6, 7, 8, 9], "mistak": 6, "luckili": 6, "nx": [6, 7], "ny": [6, 7], "lx": [6, 7], "ly": [6, 7], "extract": 6, "xij": [6, 7, 9], "yij": [6, 7, 9], "split": 6, "hmmm": 6, "seem": [6, 7, 8], "far": [6, 7], "increas": 6, "opposit": 6, "vari": 6, "horizont": 6, "matric": [6, 7], "contour": 6, "expect": 6, "ax0": [6, 7], "ax1": [6, 7], "subplot": [2, 6, 7, 9], "nrow": 6, "ncol": 6, "sharex": 6, "contourf": [6, 7, 9], "c1": 6, "set_titl": 6, "set_ytick": 6, "colorbar": 6, "tick": 6, "unnecessari": [6, 7], "wast": [6, 7], "memori": 6, "recreat": 6, "keyword": [6, 7], "smesh": 6, "sxij": 6, "syij": 6, "someth": [6, 8, 9], "input": [6, 10], "origin": 6, "contrast": 6, "dimens": [2, 6, 7, 8, 9], "extra": [6, 9], "tell": 6, "similarli": [6, 9], "effici": [2, 6, 9], "appropri": [6, 7], "highli": [2, 6], "code": [6, 7, 9], "denot": [2, 6, 9, 10], "dens": 6, "compon": [6, 8, 10], "sometim": [6, 8, 9], "sens": [6, 7, 9], "comma": 6, "between": [2, 6, 7, 8, 9], "signific": 6, "clearli": 6, "seper": 6, "long": [6, 7, 8], "sequenc": 6, "big": [6, 9], "u_i": [6, 7, 8], "arang": [2, 6, 8, 9], "reshap": 6, "laid": 6, "ravel": 6, "flatten": 6, "final": [6, 8], "third": 6, "individu": [6, 10], "skip": 6, "over": [6, 8, 9], "2u_": [2, 6], "f_": 6, "learn": [6, 7], "d_x": 6, "instead": [2, 6, 7], "_y": [6, 7], "unscal": 6, "ik": 6, "cannot": [6, 10], "anymor": 6, "On": [6, 7, 9], "correct": [6, 7, 8], "across": 6, "d_y": 6, "appar": 6, "b_i": [6, 8], "hand": [6, 7, 8, 9, 10], "help": [6, 7], "transform": 6, "regular": [6, 7, 8, 9], "call": [6, 7, 8, 9, 10], "gener": [6, 7, 8, 9], "longrightarrow": 6, "refer": [6, 7, 8, 9], "output": [6, 7, 8], "process": 6, "leav": 6, "special": [6, 9], "i_": 6, "diagon": [2, 6, 8, 9], "n_i": 6, "soon": 6, "tripl": 6, "otim": [6, 7, 9], "meanwhil": 6, "type": [6, 9], "allow": [6, 8], "multipl": [6, 9], "larger": [6, 9, 10], "q": 6, "pr": 6, "qs": 6, "block": 6, "small": [2, 6, 7], "2a": 6, "2b": 6, "2c": 6, "hline": 6, "3a": 6, "3b": 6, "3c": 6, "3d": [6, 7], "4a": 6, "4b": 6, "4c": 6, "4d": 6, "pictur": 6, "simpler": [6, 9], "new": [2, 6, 7, 8, 9], "theori": [2, 6, 7], "d2x": 6, "kron": 6, "d2u": 6, "d2y": 6, "field": [6, 7], "structur": 6, "method": [6, 7, 9, 10], "manufactur": [6, 10], "guess": 6, "diff": [6, 7, 8], "mesh2d": [6, 7], "manipul": 6, "four": [6, 7], "littl": [6, 7], "slice": 6, "trickeri": 6, "elsewher": [2, 6], "dtype": [6, 8], "bool": 6, "imshow": 6, "cmap": [6, 7], "gray_r": 6, "bnd": 6, "13": [6, 8], "14": [6, 8], "17": [6, 8], "19": [6, 9], "20": [6, 7, 8, 9], "21": [6, 7], "22": [6, 9], "23": 6, "26": [6, 8], "28": [6, 9], "29": [6, 9], "31": [6, 9], "61": 6, "62": 6, "92": 6, "93": 6, "123": 6, "124": 6, "154": 6, "155": 6, "185": 6, "186": 6, "216": 6, "217": 6, "247": 6, "248": 6, "278": 6, "279": 6, "309": 6, "310": 6, "340": 6, "341": 6, "371": 6, "372": 6, "402": 6, "403": 6, "433": 6, "434": 6, "464": 6, "465": 6, "495": 6, "496": 6, "526": 6, "527": 6, "557": 6, "558": 6, "588": 6, "589": 6, "619": 6, "620": 6, "650": 6, "651": 6, "681": 6, "682": 6, "712": 6, "713": 6, "743": 6, "744": 6, "774": 6, "775": 6, "805": 6, "806": 6, "836": 6, "837": 6, "867": 6, "868": 6, "898": 6, "899": 6, "929": 6, "930": 6, "931": 6, "932": 6, "933": 6, "934": 6, "935": 6, "936": 6, "937": 6, "938": 6, "939": 6, "940": 6, "941": 6, "942": 6, "943": 6, "944": 6, "945": 6, "946": 6, "947": 6, "948": 6, "949": 6, "950": 6, "951": 6, "952": 6, "953": 6, "954": 6, "955": 6, "956": 6, "957": 6, "958": 6, "959": 6, "960": 6, "main": 6, "tolil": 6, "tocsr": 6, "avail": [6, 7, 8, 9], "solver": [6, 7], "close": [6, 7, 9], "analyt": [6, 7, 10], "0003880705640305097": 6, "voil\u00e0": 6, "varieti": 6, "focuss": 7, "sole": 7, "problem": [2, 7, 8, 9], "talk": 7, "real": [2, 7, 8, 9, 10], "project": [7, 8, 9, 10], "often": [7, 8, 9], "measur": [7, 8], "drag": 7, "flowrat": 7, "surfac": 7, "its": [7, 8], "answer": [7, 10], "question": [7, 10], "xj": [2, 7, 8, 9], "bo": [7, 8], "profil": 7, "ask": 7, "compar": [7, 8, 9], "xl": 7, "1000": [7, 8, 9], "being": 7, "draw": 7, "straight": [7, 8, 9], "x_3": 7, "x_4": 7, "u_3": 7, "u_4": 7, "By": 7, "48": 7, "4x": 7, "uo": 7, "linearli": 7, "curv": 7, "much": [2, 7, 8, 9], "wors": [7, 8], "extrapol": 7, "500": 7, "understand": [7, 10], "your": [7, 10], "hard": 7, "earn": 7, "shouldn": 7, "higher": 7, "rather": [7, 8], "closest": 7, "separ": [7, 9], "superscript": 7, "power": 7, "basi": [7, 8], "cardin": 7, "ell_j": [2, 7, 8], "compactli": 7, "prod_": [2, 7, 8], "substack": [2, 7, 8], "le": [2, 7, 8], "lagrangebasi": [7, 8], "construct": [7, 8], "mul": [7, 8], "numert": [7, 8], "denom": [7, 8], "ell_0": [2, 7, 8], "ell_1": [2, 7, 8], "ell_2": 7, "loc": 7, "delta_": [2, 7, 8, 9], "automat": [7, 8], "lagrangefunct": [7, 8], "tupl": [7, 8], "uj": [7, 8, 9], "enumer": [7, 8], "reproduc": 7, "insid": [7, 9], "perhap": 7, "lp": 7, "500000000000000": 7, "2x": [7, 9], "costli": 7, "involv": 7, "Is": 7, "x_5": 7, "current": 7, "dl": 7, "cartesian": [7, 9], "meshgrid": [7, 9], "u2": 7, "65": 7, "procedur": 7, "perform": [7, 9], "ell_": 7, "scatter": 7, "ro": 7, "xtick": 7, "ytick": 7, "ell_m": 7, "ell_n": 7, "x_6": 7, "y_6": 7, "y_7": 7, "previous": 7, "gave": 7, "argument": [7, 9], "lagrangefunction2d": 7, "basisx": 7, "basisi": 7, "06": 7, "99999999999999": 7, "0525": 7, "0576": 7, "0504": 7, "0300000000000011": 7, "0419999999999998": 7, "0900000000000003": 7, "126": 7, "0551250000000000": 7, "0563062500000000": 7, "x_7": 7, "y_5": 7, "did": [2, 7, 8], "dlx": 7, "dly": 7, "0227500000000000": 7, "minim": 7, "effort": [2, 7], "spline": 7, "topic": [7, 8], "interpn": 7, "055125": 7, "default": [7, 8], "cubic": 7, "05630625": 7, "int_": [7, 8, 9], "int_0": [7, 8], "abl": 7, "midpoint": 7, "rule": 7, "integrand": 7, "middl": 7, "cell": 7, "squar": [2, 7], "averag": 7, "surround": 7, "mark": 7, "um": 7, "ua": 7, "961003963114361e": 7, "machin": [7, 9], "produc": 7, "send": [7, 10], "09552823074589575": 7, "09586332023451888": 7, "simplif": 7, "09980275105614006": 7, "less": [7, 8, 9, 10], "suppos": 7, "nest": 7, "trapz": [7, 9], "simpson": 7, "l2_error_trapz": 7, "09592899344305951": 7, "l2_error_simp": 7, "simp": 7, "09586418448463656": 7, "mani": [7, 8, 9], "lift": 7, "airfoil": 7, "wing": 7, "pressur": 7, "friction": 7, "forc": 7, "geometri": 7, "integrate_x": 7, "02183109": 7, "04131248": 7, "05840408": 7, "07307749": 7, "08531604": 7, "09511478": 7, "10248041": 7, "10743121": 7, "10999687": 7, "11021837": 7, "10814768": 7, "10384757": 7, "09739127": 7, "08886213": 7, "07835326": 7, "06596713": 7, "05181507": 7, "03601686": 7, "01870016": 7, "expens": [7, 8], "done": [7, 8], "integrate_boundary_x": 7, "3u_": 7, "4u_": 7, "trapezoid": 7, "dui": 7, "46011886423778525": 7, "dudy": 7, "45969769413186": 7, "refin": 7, "surprisingli": 7, "consider": 7, "easier": [7, 8, 9], "poisson": 7, "discretis": 7, "steadi": 7, "ellipt": 7, "difficult": [2, 7, 8, 9], "issu": 7, "anywher": 7, "unless": 7, "iter": [2, 7], "consecut": 7, "_x": 7, "rightarrow": [7, 8], "lot": [7, 8], "store_data": 7, "lambda": [2, 7, 8, 9], "40": [7, 8], "vec": [7, 9], "i_i": [7, 9], "i_x": [7, 9], "were": 7, "taken": 7, "care": [7, 8], "As": [2, 7, 8, 9], "significantli": 7, "fifth": 7, "501": 7, "cm": [2, 7], "subplot_kw": 7, "surf": 7, "plot_surfac": 7, "coolwarm": 7, "linewidth": 7, "antialias": 7, "anim": [7, 10], "captur": [7, 9], "otherwis": [7, 8], "frame": 7, "val": 7, "plot_wirefram": 7, "rstride": 7, "cstride": 7, "vmin": 7, "vmax": 7, "artistanim": 7, "blit": 7, "repeat_delai": 7, "wavemovie2d": 7, "apng": 7, "writer": 7, "pillow": 7, "fp": 7, "png": 7, "browser": 7, "to_jshtml": 7, "smaller": [7, 9], "spread": 7, "blow": 7, "c_x": 7, "c_y": 7, "97": 7, "book": [7, 8], "harmon": 8, "exponenti": 8, "psi_j": [2, 8, 9], "hat": [2, 8, 9], "_k": 8, "psi_k": 8, "kind": [8, 9], "functionspac": [8, 9], "v_n": [2, 8, 9], "span": [2, 8, 9], "sai": [8, 9], "u_n": [2, 8, 9], "element": 8, "best": 8, "_1": 8, "interpol": 8, "briefli": 8, "mention": [8, 9], "focu": [2, 8], "slightli": [8, 9], "equat": [2, 8, 9], "clarifi": 8, "conveni": 8, "integr": [2, 8, 9], "achiev": 8, "diplai": [], "err": [8, 9], "eq1": 8, "eq2": 8, "uhat": [8, 9], "38": 8, "orthogon": [2, 8, 9], "foral": [2, 8, 9], "messi": [], "psi_i": [2, 8, 9], "_i": [8, 9], "mass": [2, 8], "orthonorm": 8, "monomi": 8, "good": [8, 9], "ill": 8, "legendr": [8, 9], "p_0": 8, "p_1": 8, "p_2": 8, "3x": 8, "p_": [2, 8, 9], "2j": [8, 9], "xp_": 8, "p_j": [8, 9], "p_i": [8, 9], "map": [8, 9], "physic": [2, 8, 9], "sooner": 8, "sec": 8, "introduct": 8, "variat": [2, 8], "affin": 8, "revers": 8, "variabl": [8, 9], "introduc": [8, 9], "appear": 8, "cancel": 8, "highlight": 8, "ux": [], "strang": 8, "object": 8, "psi_": [2, 8, 9], "lagrang": [2, 8, 9], "ideal": [], "favour": 8, "16x": [8, 9], "attempt": [2, 8], "fine": 8, "yj": 8, "set_ylim": 8, "larg": 8, "under": 8, "shoot": 8, "bad": [8, 9], "rung": 8, "phenomenon": 8, "idea": 8, "cluster": 8, "chebyshev": [2, 8], "agreement": 8, "aors": [], "41": [], "ul": [8, 9], "semilog": [8, 9], "decreas": 8, "seri": [8, 9], "pleas": 10, "organ": 10, "matmek": 10, "4270": 10, "matmek4270": 10, "mandatory1": 10, "badg": 10, "readm": 10, "own": [9, 10], "collabor": 10, "discuss": 10, "among": 10, "student": 10, "encourag": 10, "me": 10, "email": 10, "github": 10, "usernam": 10, "obviou": [2, 10], "who": 10, "necessarili": [9, 10], "few": [9, 10], "report": 10, "place": 10, "section": 10, "com": 10, "download": 10, "dimension": 10, "poisson2d": 10, "toward": [9, 10], "test_convergence_poisson2d": 10, "test_interpol": 10, "admit": 10, "k_x": 10, "k_y": 10, "m_x": 10, "m_y": 10, "arbitrari": 10, "wave2d": 10, "stand": 10, "lectur": 10, "n_e": [2, 10], "test_convergence_wave2d": 10, "wave2d_neumann": 10, "test_convergence_wave2d_neumann": 10, "folder": 10, "pdf": 10, "imaginari": 10, "imath": 10, "unit": 10, "version": [9, 10], "kh": 10, "tild": 10, "test_exact_wave2d": 10, "neumannwav": 10, "gif": 10, "sure": [9, 10], "mb": 10, "x_m": 8, "kx": [], "p_k": [], "regardless": [], "implicitli": 8, "origo": [], "comut": [], "imposs": [2, 8], "rewrit": 8, "almost": [8, 9], "ninner": 8, "quadratur": [8, 9], "uv": 8, "uhatn": [8, 9], "3333333333333335": 8, "000000000000002": 8, "routin": 8, "adapt": [8, 9], "gradual": 8, "finer": 8, "longer": [8, 9], "improv": [8, 9], "toler": 8, "rel": 8, "absolut": 8, "49": 8, "user": 8, "bernstein": 8, "literatur": 8, "trial": 8, "0x1156c0ad0": [], "0x1170ac2d0": [], "0x1170d1910": [], "0x1171a0f50": [], "0x1171a0e50": [], "uh": [8, 9], "ui": 8, "evid": 8, "oscil": [8, 9], "allwai": 8, "w_n": [], "hatusin": [], "minor": [8, 9], "world": 8, "unc": 8, "58012275e": 8, "00000000e": 8, "55601020e": 8, "06409820e": 8, "52222377e": 8, "53926304e": 8, "93848441e": 8, "17438450e": 8, "64480816e": 8, "adjust": 8, "51283542": 8, "18309886": 8, "60209262": 8, "59154943": 8, "99795065": 8, "06103295": 8, "72004323": 8, "79577472": 8, "56234498": 8, "63661977": 8, "46105771": 8, "53051648": 8, "39059163": 8, "45472841": 8, "33876606": 8, "0x115e6ca50": [], "0x115e6cb90": [], "0x116d19750": [], "0x116cf7550": [], "0x116d4d250": [], "0x11ac7fc50": [], "0x11b0b54d0": [], "0x119c17f90": [], "0x11b13ae10": [], "0x11b13b310": [], "0x11841bb90": [], "0x1189c3a50": [], "0x1197f5610": [], "0x1197d7150": [], "0x1197d73d0": [], "experi": [8, 9], "j_0": [8, 9], "bessel": 8, "0x11229f4d0": [], "0x112302e10": [], "0x11238d7d0": [], "0x1123e0b90": [], "0x112400f10": [], "0x11a614b90": [], "0x10f8df550": [], "0x11b7136d0": [], "0x11b71f510": [], "0x11b750950": [], "ii": 8, "80": 8, "roundoff": [8, 9], "60": 8, "visibl": 8, "0x116f28b90": [], "0x118020950": [], "0x1180f3290": [], "0x118101610": [], "0x118102ad0": [], "untradit": 8, "0x11ac742d0": [], "0x11b9c0bd0": [], "0x11bb086d0": [], "0x11bb4b790": [], "0x11bb6cb90": [], "suit": 9, "smooth": 9, "anoth": [2, 9], "t_k": 9, "cosin": 9, "definit": 9, "t_0": 9, "t_1": 9, "2xt_": 9, "arcco": [2, 9], "noteworthi": 9, "maximum": 9, "minimum": 9, "fact": [2, 9], "extrema": 9, "root": 9, "2n": 9, "gauss": 9, "extrem": 9, "disadvantag": [], "weight": 9, "2_": 9, "usag": 9, "t_j": 9, "galerkin": [2, 9], "l_": 9, "c_i": 9, "c_0": 9, "0x112c44bd0": [], "0x11e43c9d0": [], "0x11e46b9d0": [], "0x11d716d90": [], "0x11e548fd0": [], "choic": 9, "boil": 9, "down": 9, "attent": 9, "intimid": 9, "tk": 9, "aco": 9, "cj": 9, "innerw": 9, "vs": [], "uhj": 9, "enough": 9, "c2": 9, "c3": 9, "least": [2, 9], "timeconsum": 9, "innerwn": 9, "maxit": [], "slow": 9, "50": [], "comparison": 9, "semilogarithm": 9, "odd": 9, "fillstyl": 9, "uej": 9, "astyp": 9, "errc": 9, "errl": 9, "hardli": 9, "advantag": [2, 9], "fast": [], "fourier": 9, "vandermond": 9, "cost": 9, "flop": 9, "chebvand": 9, "suppress": 9, "707": 9, "t_5": 9, "y_i": 9, "dct": 9, "y_n": 9, "reformul": 9, "evaluate_cheb_1": 9, "fft": 9, "xi": 9, "uj_fast": 9, "speed": 9, "timeit": 9, "\u00b5s": 9, "std": 9, "dev": 9, "run": 9, "000": [], "36": 9, "52": [], "Not": 9, "must": [], "could": 9, "sent": [], "610": [], "ns": [], "i_m": 9, "767": [], "faster": 9, "fftw": 9, "mpi4pi": 9, "504": [], "46": [], "reli": 9, "popular": [], "bound": [], "high": [], "spectral": 9, "access": [], "seldom": [], "softwar": [], "chebfun": 9, "shenfun": 9, "furthermor": [2, 9], "rq": [], "chebcoeff": [], "532": [], "67": [], "ref_domain": [8, 9], "thing": 9, "denomin": 9, "mind": 9, "ref": 9, "coord": 9, "besid": 9, "slowest": [], "took": [], "fastest": [], "intermedi": [], "cach": [], "33": [], "68": [], "42": [], "mainli": 9, "matlab": 9, "via": 9, "sf": 9, "testfunct": 9, "importerror": [], "traceback": [], "recent": [], "file": [], "mysoftwar": [], "py": [], "mpi": [], "37": [], "config": [], "dumpconfig": [], "39": [], "chebyshevu": [], "la": [], "optim": [], "runtimeoptim": [], "sparsematrixsolv": [], "matrixbas": [], "tpmatrix": [], "spectralmatrix": [], "extract_bc_matric": [], "get_simplified_tpmatric": [], "importlib": [], "functool": [], "cython": [], "tdma_lu": [], "tdma_solv": [], "pdma_lu": [], "pdma_solv": [], "lu_helmholtz": [], "solve_helmholtz": [], "lu_biharmon": [], "biharmonic_factor_pr": [], "biharmonic_solv": [], "tdma_o_solv": [], "tdma_o_lu": [], "poisson_solve_add": [], "fdma_solv": [], "twodma_solv": [], "threedma_solv": [], "fdma_lu": [], "diagma_solv": [], "tdma_inner_solv": [], "tdma_o_inner_solv": [], "diagma_inner_solv": [], "pdma_inner_solv": [], "fdma_inner_solv": [], "twodma_inner_solv": [], "threedma_inner_solv": [], "heptadma_inner_solv": [], "heptadma_solv": [], "heptadma_lu": [], "solvergeneric1nd_solve_data": [], "matvec": [], "helmholtz_matvec": [], "helmholtz_neumann_matvec": [], "biharmonic_matvec": [], "outer": 9, "outer2d": [], "outer3d": [], "name": [], "mikaelmortensen": [], "cpython": [], "darwin": [], "shortcut": 9, "throughout": 9, "typic": 9, "uhc": 9, "89": [], "81": [], "47": [], "83": [], "63": [], "160": [], "99": [], "51": [], "08": [], "0x111c0e510": [], "0x1129b40d0": [], "0x1129cb810": [], "0x112a81a50": [], "0x112a949d0": [], "34": [], "54": [], "0x1150c7d90": [], "0x115ed19d0": [], "0x115e7f490": [], "0x10a5c0490": [], "0x115f80510": [], "72": [], "79": [], "explicitli": 9, "96": [], "53": [], "78": [], "myst_nb": [], "glue": [], "caption": [], "71": [], "88": [], "58": [], "nameerror": [], "94": [], "74": [], "ipython3": [], "sym": [], "z": [], "example_eq": [], "math": [], "label": [], "syntaxerror": [], "invalid": [], "syntax": [], "jupytext": [], "format": [], "md": [], "myst": [], "text_represent": [], "extens": [], "format_nam": [], "kernelspec": [], "display_nam": [], "languag": [], "python3": [], "45": [], "43": [], "77": [], "59": [], "56": [], "69": [], "9b": [], "global": 2, "73": [], "87": [], "aspect": 2, "fit": [], "obstruct": 2, "core": [], "style": [], "tabl": [], "output_png": [], "5050000000000001": [], "405": [], "colloc": [2, 9], "everywher": 2, "local": 2, "imagin": 2, "laplacian": 2, "2d": [2, 9], "dolfin": 2, "design": 2, "triangul": 2, "piecewis": 2, "piesewis": [], "support": 2, "previou": 2, "chapter": 2, "continu": 2, "nonoverlap": 2, "rare": 2, "femmesh": [], "constain": [], "psi_0": [2, 9], "psi_1": [2, 9], "psi_2": 2, "rainbow": 2, "fem": 2, "One": [], "psi_5": 2, "flexibl": 2, "major": 2, "70": [], "varphi_j": 9, "straightforward": 9, "v_x": 9, "v_y": 9, "varphi_i": 9, "tensor": 9, "dyadic": 9, "kroneck": 9, "relat": 9, "varphi_0": 9, "varphi_1": 9, "understood": 9, "vx": 9, "vy": 9, "l1": 2, "aij": 2, "04166667": 2, "02083333": 2, "08333333": 2, "76": [], "dxdy": 9, "86": [], "p_m": 9, "p_n": 9, "independ": 9, "im": 9, "jn": 9, "mn": 9, "trick": 9, "avoid": 9, "dblquad": 9, "half": 9, "uij": 9, "a_inv": 9, "uhat_ij": 9, "legvand": 9, "ueij": 9, "1480214455446e": [], "09": [], "epsab": 9, "519715605065469e": 9, "tensorproductspac": 9, "comm": 9, "local_mesh": 9, "export": [], "pythonpath": [], "trialfunct": [], "fixed_resolut": [], "echo": [], "0x115c2ce10": [], "0x1170a7210": [], "0x117086d90": [], "0x11716f190": [], "0x1171677d0": [], "44": [], "8675821003376366e": [], "valueerror": [], "opt": [], "anaconda3": [], "env": [], "bookmatmek4270": [], "lib": [], "site": [], "260": [], "expr0": [], "expr1": [], "output_arrai": [], "return_matric": [], "258": [], "testm": [], "function_spac": [], "get_refin": [], "259": [], "outm": [], "scalar_product": [], "buffer": [], "dim": [], "261": [], "262": [], "broadcast": [], "log_2": 9, "view": 9, "84": [], "rest": 9, "fairli": 9, "resolut": 9, "descres": 9, "decad": 9, "focus": 9, "operand": [], "leg": [], "106": [], "165842587697159e": 9, "57": 9, "66": [], "105": [], "0x10edd8c10": [], "0x110081850": [], "0x1101cde50": [], "0x110097750": [], "0x1101dd490": [], "101": [], "classifi": 9, "connect": 9, "05": [], "141": [], "admittedli": 9, "innern": 9, "suscept": 9, "som": [], "109": [], "0x1141a8890": [], "0x11531e210": [], "0x11542cb50": [], "0x115453190": [], "0x115408f10": [], "98": [], "0x110474e10": [], "0x11bd1b890": [], "0x11be0a590": [], "0x11be0bb10": [], "0x11be15a50": [], "104": 9, "week": 9, "0x119da4c90": [], "0x11ae05d10": [], "0x119b20810": [], "0x11aeef390": [], "0x11aefd250": [], "0x113dbd450": 4, "0x114b174d0": 4, "0x114bd16d0": 4, "0x114bd1f90": 4, "0x114be4b10": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"matmek4270": 0, "book": 0, "lectur": [1, 2, 3, 4, 5, 6, 7, 8, 9], "note": 1, "matmek": 1, "4270": 1, "The": [1, 4, 6, 7, 8, 10], "finit": [1, 2, 4, 5, 6], "differ": [1, 4, 5, 6], "method": [1, 2, 3, 4, 5, 8], "variat": 1, "mandatori": [1, 10], "assign": [1, 8, 9, 10], "3": 3, "vibrat": [3, 4], "equat": [3, 4, 5, 6, 7, 10], "manufactur": 3, "solut": [3, 10], "adjust": 3, "solver": [3, 5, 10], "4": 4, "matrix": [4, 6], "approach": 4, "problem": [4, 6, 10], "differenti": 4, "matric": 4, "first": 4, "deriv": [4, 6, 7], "solv": [4, 6], "us": 4, "fd": 4, "gener": 4, "stencil": 4, "5": 5, "wave": [5, 7, 10], "dirichlet": [5, 10], "fix": 5, "end": 5, "neumann": [5, 10], "loos": 5, "open": 5, "boundari": [5, 6, 7, 8], "No": 5, "discret": [5, 6], "remain": 5, "complic": 5, "initi": 5, "condit": [5, 6], "period": 5, "6": 6, "two": [6, 7, 9], "dimension": [6, 7, 9], "spatial": 6, "domain": 6, "cartesian": 6, "product": 6, "meshgrid": 6, "spars": 6, "broadcast": 6, "mesh": [6, 7], "function": [2, 6, 7, 8, 9], "row": 6, "major": 6, "comput": 6, "storag": 6, "2d": [6, 7], "form": 6, "poisson": [6, 10], "s": [6, 10], "vector": 6, "vec": 6, "trick": 6, "kroneck": 6, "partial": [6, 7], "laplac": 6, "oper": 6, "7": 7, "postprocess": 7, "interpol": 7, "One": 7, "lagrang": 7, "polynomi": [7, 9], "other": 7, "tool": 7, "error": 7, "integr": 7, "over": 7, "plu": 7, "time": 7, "8": 8, "approxim": [2, 8, 9], "global": [8, 9], "least": 8, "squar": 8, "galerkin": 8, "colloc": 8, "due": 10, "16": 10, "10": 10, "2023": 10, "implement": 10, "stationari": 10, "exact": 10, "dispers": 10, "coeffici": 10, "test": 10, "creat": 10, "movi": 10, "issu": 8, "weekli": [8, 9], "9": 9, "ii": [], "chebyshev": 9, "fast": 9, "transform": 9, "option": 9, "extern": 9, "softwar": 9, "jupytext": [], "format": [], "md": [], "myst": [], "text_represent": [], "extens": [], "format_nam": [], "kernelspec": [], "display_nam": [], "python": [], "languag": [], "name": [], "python3": [], "9b": 2, "element": 2, "basi": [2, 9], "continu": 9, "nodal": 9, "vs": 9, "modal": 9}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})