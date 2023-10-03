Search.setIndex({"docnames": ["README", "intro", "lecture3", "lecture4", "lecture5", "lecture6", "lecture7", "lecture8", "mandatory1"], "filenames": ["README.md", "intro.md", "lecture3.ipynb", "lecture4.ipynb", "lecture5.ipynb", "lecture6.ipynb", "lecture7.ipynb", "lecture8.ipynb", "mandatory1.ipynb"], "titles": ["matmek4270-book", "Lecture Notes MATMEK-4270", "Lecture 3", "Lecture 4", "Lecture 5", "Lecture 6", "Lecture 7", "Lecture 8", "<span class=\"section-number\">1. </span>Mandatory assignment due 16/10-2023"], "terms": {"link": 0, "note": [0, 2, 3, 4, 5, 6, 7], "notebook": [0, 4, 6, 8], "us": [0, 2, 4, 5, 6, 7, 8], "creat": [0, 2, 3, 5, 6, 7], "thi": [0, 1, 2, 3, 4, 5, 6, 7, 8], "ar": [0, 2, 3, 4, 5, 6, 7, 8], "found": [0, 3, 6], "repositori": [0, 8], "abov": [0, 2, 3, 4, 5, 6, 7], "howev": [0, 2, 3, 4, 5, 6, 7], "all": [0, 2, 3, 4, 5, 6, 7, 8], "store": [0, 4, 5, 6], "non": [0, 3], "execut": 0, "clear": 0, "state": [0, 6], "so": [0, 2, 3, 4, 5, 6, 7], "order": [0, 2, 3, 4, 5, 6, 7, 8], "you": [0, 5, 6, 7, 8], "should": [0, 2, 3, 4, 5, 6, 7, 8], "either": [0, 7, 8], "clone": 0, "fork": [0, 8], "somewher": 0, "can": [0, 2, 3, 4, 5, 6, 7, 8], "collect": [1, 7], "3": [1, 3, 4, 5, 6, 7, 8], "4": [1, 2, 4, 5, 6, 7, 8], "5": [1, 2, 3, 5, 6, 7, 8], "6": [1, 2, 3, 6, 7, 8], "7": [1, 3, 4, 5, 7, 8], "8": [1, 2, 3, 5, 6], "due": [1, 7], "16": [1, 2, 3, 5, 6, 7], "10": [1, 2, 4, 5, 6, 7], "2023": 1, "begin": [2, 3, 4, 5, 6, 7, 8], "u": [2, 3, 4, 5, 6, 7, 8], "omega": [2, 3, 5, 6, 7, 8], "2": [2, 3, 4, 5, 6, 7, 8], "0": [2, 3, 4, 5, 6, 7, 8], "i": [2, 3, 4, 5, 6, 7, 8], "end": [2, 3, 5, 6, 7, 8], "implement": [2, 3, 4, 5, 6, 7], "align": [2, 3, 4, 5, 6, 7], "1": [2, 3, 4, 5, 6, 7, 8], "frac": [2, 3, 4, 5, 6, 7, 8], "delta": [2, 3, 4, 5, 6, 8], "t": [2, 3, 4, 5, 6, 8], "n": [2, 3, 4, 5, 6, 7, 8], "2u": [2, 3, 4, 6, 8], "ldot": [2, 3, 4, 5, 6, 7, 8], "n_t": [2, 3, 4, 6, 8], "import": [2, 3, 4, 5, 6, 7], "numpi": [2, 3, 4, 5, 6, 7], "np": [2, 3, 4, 5, 6, 7], "matplotlib": [2, 3, 4, 5, 6, 7], "pyplot": [2, 3, 4, 5, 6, 7], "plt": [2, 3, 4, 5, 6, 7], "def": [2, 4, 5, 6, 7], "dt": [2, 3, 4, 6], "w": [2, 4, 8], "35": 2, "solv": [2, 4, 6, 7, 8], "eq": [2, 3, 5, 6, 7, 8], "paramet": [2, 4, 6, 7], "float": [2, 6, 7], "time": [2, 3, 4, 5, 7, 8], "step": [2, 3, 4, 6], "option": [2, 4], "model": [2, 3], "return": [2, 4, 5, 6, 7], "array_lik": 2, "discret": [2, 6, 8], "The": [2, 4], "nt": [2, 4, 6], "int": [2, 4], "round": 2, "zero": [2, 3, 4, 5, 6, 7], "linspac": [2, 3, 4, 5, 6, 7], "rang": [2, 3, 4, 5, 6, 7], "u_exact": 2, "exact": [2, 3, 5, 6, 7], "arrai": [2, 3, 4, 5, 6, 7], "comput": [2, 3, 4, 6, 7, 8], "ue": [2, 5, 6, 8], "co": [2, 3, 5, 6, 7, 8], "we": [2, 3, 4, 5, 6, 7, 8], "now": [2, 3, 4, 5, 6, 7], "want": [2, 6, 7], "converg": [2, 7, 8], "rate": 2, "ell": [2, 6, 7, 8], "norm": [2, 3, 6, 7, 8], "e_i": 2, "sqrt": [2, 3, 5, 6, 7, 8], "t_i": 2, "sum_": [2, 3, 4, 5, 6, 7, 8], "_t": 2, "u_": [2, 5, 6, 7, 8], "t_n": [2, 3, 4, 6, 8], "given": [2, 3, 4, 8], "uniform": [2, 4, 6, 7, 8], "mesh": [2, 3, 4, 7, 8], "level": 2, "For": [2, 3, 4, 5, 6, 7, 8], "exampl": [2, 3, 4, 5, 6, 7], "0_t": 2, "2_t": 2, "32": 2, "etc": [2, 7], "t_": [2, 3], "t_2": 2, "interv": [2, 4, 6], "point": [2, 3, 4, 5, 6, 7, 8], "assum": [2, 4, 6, 8], "error": [2, 3, 7, 8], "written": [2, 3, 4, 5, 6, 7], "c": [2, 3, 4, 5, 6, 8], "r": [2, 3, 5, 6, 7], "where": [2, 3, 4, 5, 6, 7, 8], "constant": [2, 3, 5, 6, 7], "wai": [2, 6], "have": [2, 3, 4, 5, 6, 7], "two": [2, 3, 4, 7, 8], "e_": [2, 8], "left": [2, 3, 4, 5, 6, 7, 8], "right": [2, 3, 4, 5, 6, 7, 8], "isol": [2, 3], "log": 2, "differ": [2, 6, 7, 8], "find": [2, 5, 6, 7, 8], "let": [2, 3, 4, 5, 6, 7], "s": [2, 3, 6, 7], "first": [2, 4, 5, 6, 7], "write": [2, 4, 5, 6, 7], "function": [2, 3, 4, 8], "defin": [2, 4, 5, 6, 7, 8], "l2_error": [2, 6], "sol": [2, 4], "l2": [2, 6, 7, 8], "result": [2, 4, 5, 6, 7, 8], "from": [2, 3, 4, 5, 6, 7, 8], "callabl": 2, "sum": [2, 5, 6, 7], "convergence_r": 2, "m": [2, 3, 5, 6, 7], "dt0": 2, "30": [2, 5], "num_period": 2, "empir": 2, "estim": 2, "base": [2, 6], "simul": [2, 4], "halv": 2, "each": [2, 4, 5, 6, 7, 8], "number": [2, 4, 5, 6, 7, 8], "per": 2, "period": 2, "coarsest": 2, "size": [2, 3, 5, 6], "domain": [2, 3, 4, 6, 7, 8], "2pi": 2, "p": [2, 5, 7], "pi": [2, 3, 4, 5, 7, 8], "dt_valu": 2, "e_valu": 2, "e": [2, 3, 5, 6, 7, 8], "append": [2, 6, 7], "same": [2, 3, 4, 5, 6, 7, 8], "test": [2, 6, 7], "0036366687367093": 2, "000949732819596": 2, "000240106059534": 2, "13526035155205302": 2, "033729955960525936": 2, "008426939670390836": 2, "0021063843253281167": 2, "5983986006837702": 2, "2991993003418851": 2, "14959965017094254": 2, "07479982508547127": 2, "These": [2, 3, 4, 5, 6, 7], "last": [2, 3, 4, 5, 7], "list": [2, 4, 5], "timestep": [2, 4, 8], "af": 2, "plot": [2, 3, 4, 5, 6, 7], "show": [2, 3, 4, 5, 8], "scheme": [2, 3, 4, 6], "second": [2, 3, 4, 5, 6, 7], "loglog": 2, "titl": 2, "finit": [2, 6, 7, 8], "legend": [2, 3, 6, 7], "mathcal": [2, 3], "o": [2, 3], "plotslop": 2, "slope_mark": 2, "see": [2, 4, 5, 6, 7, 8], "ha": [2, 3, 4, 5, 6, 7], "slope": 2, "accuraci": [2, 3, 4, 6, 7], "test_ord": 2, "assert": 2, "allclos": 2, "atol": 2, "1e": 2, "trivial": [2, 3], "verifi": [2, 5, 6, 8], "work": [2, 5, 6, 7, 8], "satisfi": [2, 7, 8], "boundari": [2, 3, 8], "condit": [2, 3, 6, 7, 8], "ne": [2, 5, 6, 7], "need": [2, 3, 4, 5, 6, 7, 8], "modifi": [2, 3, 4, 5, 6, 7, 8], "sourc": [2, 3], "term": [2, 3, 6], "sinc": [2, 4, 5, 6, 7], "case": [2, 5, 6, 7], "get": [2, 3, 4, 5, 6, 7, 8], "also": [2, 3, 4, 5, 6, 7, 8], "know": [2, 3, 5, 6, 7], "accur": [2, 3, 4, 6, 7], "polynomi": [2, 7], "check": [2, 5, 6, 7], "uc": [2, 7], "numer": [2, 3, 4, 6, 7, 8], "initi": [2, 6, 8], "still": [2, 3, 6, 7], "ok": [2, 4, 7], "becom": [2, 5, 6, 7], "12": [2, 3, 4, 5, 6, 7], "fourth": 2, "print": [2, 3, 5, 6, 7], "02": [2, 3, 7], "take": [2, 3, 4, 5, 6, 7, 8], "closer": 2, "look": [2, 3, 4, 5, 6], "differec": 2, "express": [2, 3, 5], "come": [2, 5, 7], "taylor": [2, 3], "expans": [2, 3, 7], "around": [2, 3, 4], "24": [2, 3, 5, 6], "cdot": [2, 3, 4, 5, 6, 7], "ad": [2, 3, 5, 6, 7], "There": [2, 4, 5, 6, 7, 8], "proport": [2, 3], "our": [2, 3, 5, 6, 7], "put": [2, 5], "residu": 2, "go": [2, 6], "back": [2, 5], "correspond": [2, 5, 6], "exactli": [2, 4, 5, 6, 7], "It": [2, 3, 4, 5, 6, 7], "becaus": [2, 3, 4, 5, 6, 7], "next": [2, 4, 6], "sixth": 2, "deriv": [2, 4, 7], "which": [2, 3, 4, 5, 6, 7], "adj_solv": 2, "r2": 2, "e2": [2, 3], "dt2": 2, "divid": [3, 4], "1d": [3, 5, 6], "line": [3, 5, 6, 7], "onli": [3, 4, 5, 6, 7], "specif": [3, 6], "locat": [3, 5, 6], "node": [3, 4, 5, 6], "A": [3, 4, 5, 6, 7, 8], "up": [3, 5, 6, 7], "until": [3, 6, 7], "recurs": [3, 4, 6], "veri": [3, 4, 5, 6, 7], "easi": [3, 5, 6], "intuit": 3, "loop": [3, 4, 5, 6], "most": [3, 4, 5, 6], "common": [3, 6, 7], "through": [3, 4, 5, 6], "explicit": [3, 6], "assembl": [3, 5], "consid": [3, 4, 5, 6, 7, 8], "decai": 3, "au": 3, "solut": [3, 4, 5, 6, 7], "vector": [3, 4, 6, 7], "boldsymbol": [3, 5, 6, 7, 8], "To": [3, 5, 6], "start": [3, 4, 5, 6, 8], "set": [3, 4, 5, 6, 7], "theta": 3, "rearrang": [3, 6, 7], "algorithm": [3, 4, 6], "set_printopt": 3, "precis": [3, 6], "te": 3, "1001": 3, "b": [3, 4, 5, 6, 7], "exp": [3, 4, 5, 6, 7], "k": [3, 4, 5, 6, 7, 8], "text": [3, 4, 5, 6, 7], "82": 3, "never": 3, "linear": [3, 5, 6, 7], "just": [3, 4, 5, 6, 7], "forward": [3, 4, 6], "simpli": [3, 4, 5, 6, 7], "an": [3, 4, 5, 6, 7, 8], "formul": [3, 7], "mathbb": [3, 5, 7], "coeffici": [3, 5, 7], "bmatrix": [3, 4, 5, 6], "algebra": [3, 4, 5, 6, 7], "gaussian": 3, "elimin": 3, "system": [3, 7], "like": [3, 4, 5, 6, 7], "underbrac": [3, 5], "_": [3, 4, 5, 6, 7, 8], "9": [3, 5, 6, 7], "notic": [3, 4, 5, 7], "row": [3, 4, 6, 7], "remain": [3, 8], "scipi": [3, 4, 5, 6, 7], "spars": [3, 4, 6], "packag": [3, 5], "diag": [3, 4, 5, 6], "full": 3, "ones": [3, 5], "csr": 3, "toarrai": [3, 5], "333": 3, "un": [3, 4, 6, 7], "linalg": [3, 5], "spsolve_triangular": 3, "lower": [3, 7], "true": [3, 5, 6], "unit_diagon": 3, "000e": 3, "00": [3, 7], "333e": 3, "01": 3, "111e": 3, "704e": 3, "235e": 3, "115e": 3, "03": [3, 7], "372e": 3, "572e": 3, "04": [3, 7], "524e": 3, "befor": [3, 6, 7], "quad": [3, 4, 5, 6, 7, 8], "central": [3, 4, 5, 6, 8], "triangular": 3, "upper": [3, 6], "especi": [3, 5], "quick": 3, "backward": 3, "substitut": 3, "here": [3, 4, 5, 6, 7, 8], "specifi": [3, 4, 6], "one": [3, 4, 5, 6, 7], "unknown": [3, 4, 5, 6, 7], "more": [3, 4, 5, 6, 7], "orderli": 3, "obtain": [3, 4, 6], "follow": [3, 4, 5, 6, 7], "three": [3, 4, 6, 7, 8], "h": [3, 6, 8], "2h": [3, 6], "3h": 3, "27": [3, 5], "rememb": 3, "simplic": [3, 4, 5, 6, 7], "oper": [3, 6], "product": [3, 4, 6, 7], "d": [3, 4, 5, 6], "repres": [3, 4, 5, 6, 7], "approxim": [3, 8], "vdot": [3, 4, 5, 6, 7], "ddot": [3, 4, 5, 6], "open": [3, 6], "requir": [3, 4, 5, 6, 7], "subtract": 3, "do": [3, 4, 5, 6, 7], "better": [3, 4, 6, 7], "ye": 3, "cours": 3, "add": [3, 5, 7], "both": [3, 4, 5, 6, 7, 8], "don": 3, "worri": 3, "about": [3, 5, 6, 7, 8], "how": [3, 4, 5, 6, 7, 8], "yet": [3, 6], "4u": 3, "5u": 3, "11": [3, 5], "lead": [3, 4, 5, 6, 7, 8], "side": [3, 5, 6, 7, 8], "python": [3, 4, 5, 6, 7, 8], "d2": [3, 4, 5, 6], "lil": [3, 4, 5, 6], "fix": [3, 5, 7], "forget": 3, "If": [3, 4, 5, 6, 7], "appli": [3, 4, 5, 6, 7], "f": [3, 4, 5, 6, 7, 8], "try": [3, 4, 5, 6, 7], "25": [3, 5, 6, 7], "d2f": 3, "edg": [3, 4, 7], "d2e": 3, "what": [3, 4, 5, 6], "happen": [3, 5], "why": [3, 6], "perfect": [3, 7], "reason": [3, 6], "henc": [3, 4, 6, 7], "even": [3, 5, 6, 7], "though": [3, 6, 7], "complex": [3, 7], "would": [3, 5, 6], "sin": [3, 4, 5, 6, 7, 8], "d2fe": 3, "d2f1": 3, "2nd": 3, "1st": 3, "0x117158310": [], "similar": [3, 4, 6], "skew": 3, "again": [3, 5, 6, 7], "mere": [3, 4, 6], "3u": 3, "d1": [3, 6], "d1fe": 3, "d1f": 3, "0x117fee4d0": [], "equal": [3, 4, 5, 6, 7, 8], "d2n": 3, "75": [3, 6], "en": 3, "16216463185914665": 3, "37045090252771984": 3, "shown": [3, 4, 5, 6, 7], "itself": [3, 4, 6], "great": 3, "thei": [3, 5, 7], "depend": [3, 4, 6, 7], "mai": [3, 4, 5, 6, 7], "onc": [3, 5, 6], "reus": [3, 6, 7], "accord": [3, 7], "ident": [3, 5, 6, 7], "item": [3, 4, 5, 6], "id": 3, "ey": [3, 5], "u1": [3, 7], "spsolv": [3, 5], "line2d": 3, "0x118063e90": [], "0x1180b4e50": [], "0x1180b5150": [], "fulli": 3, "implicit": 3, "neighbour": [3, 4, 6], "everi": [3, 4, 6], "stabl": [3, 6], "than": [3, 5, 6, 7, 8], "possibl": [3, 4, 5, 6, 7], "ani": [3, 4, 5, 6, 7], "posit": 3, "neg": 3, "direct": [3, 5, 6, 8], "x": [3, 4, 5, 6, 7, 8], "x_0": [3, 5, 7], "read": [3, 5, 6, 8], "dx": [3, 4, 5, 6, 7], "With": [3, 6, 7], "evalu": [3, 4, 5, 6], "certain": [3, 7], "That": [3, 5, 7], "mh": 3, "integ": [3, 5, 7, 8], "usual": [3, 4, 6, 7], "notat": [3, 4, 5, 7, 8], "indic": [3, 5], "respect": [3, 4, 5], "c_": 3, "mi": 3, "du_i": 3, "neglect": 3, "du": [3, 6], "m_0": 3, "form": [3, 6, 7], "du_": 3, "du_1": 3, "du_2": 3, "du_3": 3, "du_4": 3, "easili": [3, 5, 6, 7, 8], "re": [3, 7], "normal": [3, 5], "interest": [3, 6, 7], "And": [3, 4, 5, 6, 7], "In": [3, 4, 5, 6, 7], "sympi": [3, 4, 5, 6, 7, 8], "sp": [3, 4, 5, 6, 7], "symbol": [3, 4, 5, 6, 7], "displaystyl": [3, 6, 7], "invers": 3, "inv": 3, "out": [3, 4, 5, 6, 7], "2i": [3, 7], "coef": 3, "inli": 3, "thu": [3, 5, 6, 7], "partial": [4, 7, 8], "differenti": [4, 5, 6], "pde": [4, 6, 7], "space": [4, 6, 7, 8], "l": [4, 6, 7, 8], "hyperbol": [4, 6], "ct": 4, "reflect": [4, 6, 8], "chang": [4, 7, 8], "sign": 4, "nonzero": 4, "without": [4, 7, 8], "pass": [4, 8], "undisturb": 4, "unreflect": 4, "illustr": [4, 5, 6], "nice": 4, "turn": [4, 5, 7], "puls": 4, "off": [4, 5, 6], "damp": 4, "press": 4, "green": [4, 5, 6, 8], "button": 4, "ipython": [4, 6, 7], "displai": [4, 6, 7], "ifram": 4, "http": 4, "phet": 4, "colorado": 4, "edu": 4, "sim": 4, "html": [4, 6], "string": 4, "latest": 4, "string_al": 4, "width": 4, "600": 4, "height": 4, "400": [4, 6], "v": [4, 5, 7], "some": [4, 6, 7], "simplest": 4, "x_j": [4, 5, 6, 7], "j": [4, 5, 6, 7, 8], "grid": [4, 6, 7], "below": [4, 5, 6, 7], "xlabel": 4, "ylabel": 4, "ko": [4, 5, 6], "15": [4, 5, 6, 7, 8], "x_2": [4, 6, 7], "t_3": 4, "n_j": 4, "valu": [4, 5, 6, 7], "later": [4, 6, 7], "n_0": 4, "n_1": 4, "n_n": 4, "_j": [4, 6, 7], "n_": [4, 6, 8], "stencil": [4, 6], "make": [4, 5, 6, 7, 8], "fig": [4, 5, 6], "figur": [4, 5, 6, 7], "figsiz": [4, 5, 6, 7], "ax": [4, 5, 6, 7], "gca": [4, 5, 6, 7], "get_xaxi": 4, "set_vis": 4, "fals": [4, 6], "get_yaxi": 4, "axi": [4, 5, 6], "basic": 4, "chosen": [4, 5, 6, 7], "storag": 4, "total": [4, 5], "wherea": [4, 5, 6, 7], "alwai": [4, 7], "known": [4, 5, 6], "underlin": 4, "courant": 4, "stabil": [4, 6], "matrix": [4, 6, 7, 8], "simplifi": [4, 6], "doe": [4, 5, 6, 7], "includ": [4, 7, 8], "formula": [4, 5], "th": 4, "ji": 4, "except": [4, 5], "modif": [4, 7], "inner": [4, 7], "singl": [4, 5, 6], "addit": [4, 6], "simpl": [4, 5, 6, 7], "tricki": 4, "part": [4, 6, 7], "complet": [4, 6, 8], "updat": [4, 6], "move": [4, 5, 6, 8], "leftarrow": [4, 6], "swap": [4, 6], "readi": [4, 5, 6], "At": [4, 5], "class": [4, 6, 8], "wave1d": 4, "assign": 4, "other": [4, 5, 7], "reader": 4, "advis": 4, "studi": 4, "detail": 4, "hold": 4, "100": [4, 6, 7], "spatial": [4, 6, 8], "unp1": [4, 6], "unm1": [4, 6], "altern": [4, 5], "actual": [4, 5, 7], "further": [4, 5, 6], "insert": [4, 6, 7, 8], "__call__": 4, "sever": [4, 6, 7], "seen": [4, 5], "entir": [4, 5, 6, 7], "apply_bc": 4, "bc": 4, "scale": 4, "after": [4, 6], "n_2": 4, "multipli": 4, "scalar": [4, 7], "_0": [4, 7], "_n": 4, "intern": [4, 6, 8], "matter": [4, 5, 6, 7], "whatev": 4, "overwritten": 4, "necessari": [4, 5, 6], "suffici": 4, "them": [4, 5, 6, 7], "bit": 4, "homogen": [4, 8], "outsid": [4, 6], "togeth": [4, 5], "similarili": 4, "valid": [4, 5, 7], "d_n": 4, "word": 4, "anyth": [4, 5, 7], "directli": [4, 5], "approach": [4, 6, 7], "give": [4, 5, 7], "unfortun": 4, "overlin": [4, 6], "mean": [4, 5, 6, 7], "repeat": [4, 5], "indefinit": 4, "sine": [4, 7], "800": 4, "fi": 4, "red": [4, 5, 6], "dot": [4, 6], "earlier": 4, "been": [4, 5, 6], "wrap": 4, "consequ": 4, "noth": 4, "ordinari": [4, 6], "d_p": 4, "_p": 4, "l0": 4, "extent": 4, "c0": [4, 5], "wavespe": 4, "cfl": [4, 6, 8], "u0": [4, 6, 7], "__init__": [4, 8], "self": 4, "200": 4, "paramt": 4, "bake": 4, "rais": 4, "notimplementederror": 4, "elif": 4, "none": 4, "provid": [4, 7], "els": [4, 7], "runtimeerror": 4, "wrong": [4, 7], "properti": [4, 8], "ic": 4, "save_step": 4, "u_t": 4, "save": [4, 5, 6, 8], "dictionari": 4, "kei": 4, "length": [4, 5, 8], "lambdifi": [4, 5, 6, 7], "sub": [4, 6, 7], "plotdata": [4, 6], "copi": [4, 6], "plot_with_offset": 4, "data": [4, 6], "nd": 4, "len": [4, 6, 7], "v0": 4, "ab": 4, "max": [4, 6], "facecolor": 4, "add_subplot": 4, "111": 4, "lw": 4, "zorder": 4, "fill_between": 4, "choos": [4, 5, 6, 7], "uniti": 4, "averi": 4, "offset": 4, "flip": 4, "when": [4, 5, 6, 7, 8], "reach": 4, "within": 5, "nabla": [5, 6, 8], "coordin": [5, 7], "y": [5, 6, 8], "rectangular": 5, "l_x": [5, 6], "l_y": [5, 6], "dirichlet": [5, 6], "rectangl": 5, "color": 5, "blue": [5, 6], "magenta": 5, "likewis": 5, "consist": [5, 7], "corner": 5, "belong": 5, "along": [5, 6], "easiest": 5, "g": [5, 6, 7], "arrow": 5, "head_width": 5, "rotat": 5, "vertic": 5, "x_i": [5, 6, 7, 8], "n_x": [5, 6], "y_j": [5, 6, 8], "n_y": [5, 6], "x_1": [5, 7], "x_": 5, "y_0": [5, 6], "y_1": 5, "y_": 5, "explain": 5, "itertool": 5, "uxv": 5, "contain": [5, 7], "pair": 5, "describ": [5, 6, 7, 8], "mathemat": 5, "well": [5, 6, 7], "combin": 5, "alreadi": [5, 6, 8], "realli": [5, 6], "grei": 5, "dy": [5, 6], "18": [5, 6], "07": 5, "55": [5, 6], "drop": 5, "front": 5, "xy": 5, "present": 5, "shape": [5, 6], "doubl": [5, 6], "natur": [5, 6], "horizontalalign": [5, 6], "center": [5, 6], "wa": [5, 6, 8], "column": [5, 6, 7], "counterintuit": 5, "think": 5, "a_": [5, 7], "ij": [5, 6, 7, 8], "index": [5, 6, 7], "complic": [5, 6, 7], "factor": [5, 7], "awar": 5, "But": [5, 6, 7], "mistak": 5, "luckili": 5, "nx": [5, 6], "ny": [5, 6], "lx": [5, 6], "ly": [5, 6], "extract": 5, "xij": [5, 6], "yij": [5, 6], "split": 5, "hmmm": 5, "seem": [5, 6, 7], "far": [5, 6], "increas": 5, "opposit": 5, "vari": 5, "horizont": 5, "matric": [5, 6], "contour": 5, "expect": 5, "ax0": [5, 6], "ax1": [5, 6], "subplot": [5, 6], "nrow": 5, "ncol": 5, "sharex": 5, "contourf": [5, 6], "c1": 5, "set_titl": 5, "set_ytick": 5, "colorbar": 5, "tick": 5, "unnecessari": [5, 6], "wast": [5, 6], "memori": 5, "recreat": 5, "keyword": [5, 6], "smesh": 5, "sxij": 5, "syij": 5, "someth": [5, 7], "input": [5, 8], "origin": 5, "contrast": 5, "dimens": [5, 6, 7], "extra": 5, "tell": 5, "similarli": 5, "effici": 5, "appropri": [5, 6], "highli": 5, "code": [5, 6], "denot": [5, 8], "dens": 5, "compon": [5, 7, 8], "sometim": [5, 7], "sens": [5, 6], "comma": 5, "between": [5, 6, 7], "signific": 5, "clearli": 5, "seper": 5, "long": [5, 6, 7], "sequenc": 5, "big": 5, "u_i": [5, 6, 7], "arang": [5, 7], "reshap": 5, "laid": 5, "ravel": 5, "flatten": 5, "final": [5, 7], "third": 5, "individu": [5, 8], "skip": 5, "over": [5, 7], "2u_": 5, "f_": 5, "learn": [5, 6], "d_x": 5, "instead": [5, 6], "_y": [5, 6], "unscal": 5, "ik": 5, "cannot": [5, 8], "anymor": 5, "On": [5, 6], "correct": [5, 6, 7], "across": 5, "d_y": 5, "appar": 5, "b_i": [5, 7], "hand": [5, 6, 7, 8], "help": [5, 6], "transform": 5, "regular": [5, 6, 7], "call": [5, 6, 7, 8], "gener": [5, 6, 7], "longrightarrow": 5, "refer": [5, 6, 7], "output": [5, 6, 7], "process": 5, "leav": 5, "special": 5, "i_": 5, "diagon": [5, 7], "n_i": 5, "soon": 5, "tripl": 5, "otim": [5, 6], "meanwhil": 5, "type": 5, "allow": [5, 7], "multipl": 5, "larger": [5, 8], "q": 5, "pr": 5, "qs": 5, "block": 5, "small": [5, 6], "2a": 5, "2b": 5, "2c": 5, "hline": 5, "3a": 5, "3b": 5, "3c": 5, "3d": [5, 6], "4a": 5, "4b": 5, "4c": 5, "4d": 5, "pictur": 5, "simpler": 5, "new": [5, 6, 7], "theori": [5, 6], "d2x": 5, "kron": 5, "d2u": 5, "d2y": 5, "field": [5, 6], "structur": 5, "method": [5, 6, 8], "manufactur": [5, 8], "guess": 5, "diff": [5, 6, 7], "mesh2d": [5, 6], "manipul": 5, "four": [5, 6], "littl": [5, 6], "slice": 5, "trickeri": 5, "elsewher": 5, "dtype": [5, 7], "bool": 5, "imshow": 5, "cmap": [5, 6], "gray_r": 5, "bnd": 5, "13": [5, 7], "14": [5, 7], "17": 5, "19": 5, "20": [5, 6], "21": [5, 6], "22": 5, "23": 5, "26": [5, 7], "28": 5, "29": 5, "31": 5, "61": 5, "62": 5, "92": 5, "93": 5, "123": 5, "124": 5, "154": 5, "155": 5, "185": 5, "186": 5, "216": 5, "217": 5, "247": 5, "248": 5, "278": 5, "279": 5, "309": 5, "310": 5, "340": 5, "341": 5, "371": 5, "372": 5, "402": 5, "403": 5, "433": 5, "434": 5, "464": 5, "465": 5, "495": 5, "496": 5, "526": 5, "527": 5, "557": 5, "558": 5, "588": 5, "589": 5, "619": 5, "620": 5, "650": 5, "651": 5, "681": 5, "682": 5, "712": 5, "713": 5, "743": 5, "744": 5, "774": 5, "775": 5, "805": 5, "806": 5, "836": 5, "837": 5, "867": 5, "868": 5, "898": 5, "899": 5, "929": 5, "930": 5, "931": 5, "932": 5, "933": 5, "934": 5, "935": 5, "936": 5, "937": 5, "938": 5, "939": 5, "940": 5, "941": 5, "942": 5, "943": 5, "944": 5, "945": 5, "946": 5, "947": 5, "948": 5, "949": 5, "950": 5, "951": 5, "952": 5, "953": 5, "954": 5, "955": 5, "956": 5, "957": 5, "958": 5, "959": 5, "960": 5, "main": 5, "tolil": 5, "tocsr": 5, "avail": [5, 6, 7], "solver": [5, 6], "close": [5, 6], "analyt": [5, 6, 8], "0003880705640305097": 5, "voil\u00e0": 5, "varieti": 5, "focuss": 6, "sole": 6, "problem": [6, 7], "talk": 6, "real": [6, 7, 8], "project": [6, 7, 8], "often": [6, 7], "measur": [6, 7], "drag": 6, "flowrat": 6, "surfac": 6, "its": [6, 7], "answer": [6, 8], "question": [6, 8], "xj": [6, 7], "bo": [6, 7], "profil": 6, "ask": 6, "compar": [6, 7], "xl": 6, "1000": [6, 7], "being": 6, "draw": 6, "straight": [6, 7], "x_3": 6, "x_4": 6, "u_3": 6, "u_4": 6, "By": 6, "48": 6, "4x": 6, "uo": 6, "linearli": 6, "curv": 6, "much": [6, 7], "wors": [6, 7], "extrapol": 6, "500": 6, "understand": [6, 8], "your": [6, 8], "hard": 6, "earn": 6, "shouldn": 6, "higher": 6, "rather": [6, 7], "closest": 6, "separ": 6, "superscript": 6, "power": 6, "basi": [6, 7], "cardin": 6, "ell_j": [6, 7], "compactli": 6, "prod_": [6, 7], "substack": [6, 7], "le": [6, 7], "lagrangebasi": [6, 7], "construct": [6, 7], "mul": [6, 7], "numert": [6, 7], "denom": [6, 7], "ell_0": [6, 7], "ell_1": [6, 7], "ell_2": 6, "loc": 6, "delta_": [6, 7], "automat": [6, 7], "lagrangefunct": [6, 7], "tupl": [6, 7], "uj": [6, 7], "enumer": [6, 7], "reproduc": 6, "insid": 6, "perhap": 6, "lp": 6, "500000000000000": 6, "2x": 6, "costli": 6, "involv": 6, "Is": 6, "x_5": 6, "current": 6, "dl": 6, "cartesian": 6, "meshgrid": 6, "u2": 6, "65": 6, "procedur": 6, "perform": 6, "ell_": 6, "scatter": 6, "ro": 6, "xtick": 6, "ytick": 6, "ell_m": 6, "ell_n": 6, "x_6": 6, "y_6": 6, "y_7": 6, "previous": 6, "gave": 6, "argument": 6, "lagrangefunction2d": 6, "basisx": 6, "basisi": 6, "06": 6, "99999999999999": 6, "0525": 6, "0576": 6, "0504": 6, "0300000000000011": 6, "0419999999999998": 6, "0900000000000003": 6, "126": 6, "0551250000000000": 6, "0563062500000000": 6, "x_7": 6, "y_5": 6, "did": [6, 7], "dlx": 6, "dly": 6, "0227500000000000": 6, "minim": 6, "effort": 6, "spline": 6, "topic": [6, 7], "interpn": 6, "055125": 6, "default": [6, 7], "cubic": 6, "05630625": 6, "int_": [6, 7], "int_0": [6, 7], "abl": 6, "midpoint": 6, "rule": 6, "integrand": 6, "middl": 6, "cell": 6, "squar": 6, "averag": 6, "surround": 6, "mark": 6, "um": 6, "ua": 6, "961003963114361e": 6, "machin": 6, "produc": 6, "send": [6, 8], "09552823074589575": 6, "09586332023451888": 6, "simplif": 6, "09980275105614006": 6, "less": [6, 7, 8], "suppos": 6, "nest": 6, "trapz": 6, "simpson": 6, "l2_error_trapz": 6, "09592899344305951": 6, "l2_error_simp": 6, "simp": 6, "09586418448463656": 6, "mani": [6, 7], "lift": 6, "airfoil": 6, "wing": 6, "pressur": 6, "friction": 6, "forc": 6, "geometri": 6, "integrate_x": 6, "02183109": 6, "04131248": 6, "05840408": 6, "07307749": 6, "08531604": 6, "09511478": 6, "10248041": 6, "10743121": 6, "10999687": 6, "11021837": 6, "10814768": 6, "10384757": 6, "09739127": 6, "08886213": 6, "07835326": 6, "06596713": 6, "05181507": 6, "03601686": 6, "01870016": 6, "expens": 6, "done": [6, 7], "integrate_boundary_x": 6, "3u_": 6, "4u_": 6, "trapezoid": 6, "dui": 6, "46011886423778525": 6, "dudy": 6, "45969769413186": 6, "refin": 6, "surprisingli": 6, "consider": 6, "easier": [6, 7], "poisson": 6, "discretis": 6, "steadi": 6, "ellipt": 6, "difficult": [6, 7], "issu": 6, "anywher": 6, "unless": 6, "iter": 6, "consecut": 6, "_x": 6, "rightarrow": [6, 7], "lot": [6, 7], "store_data": 6, "lambda": [6, 7], "40": [6, 7], "vec": 6, "i_i": 6, "i_x": 6, "were": 6, "taken": 6, "care": [6, 7], "As": 6, "significantli": 6, "fifth": 6, "501": 6, "cm": 6, "subplot_kw": 6, "surf": 6, "plot_surfac": 6, "coolwarm": 6, "linewidth": 6, "antialias": 6, "anim": [6, 8], "captur": 6, "otherwis": [6, 7], "frame": 6, "val": 6, "plot_wirefram": 6, "rstride": 6, "cstride": 6, "vmin": 6, "vmax": 6, "artistanim": 6, "blit": 6, "repeat_delai": 6, "wavemovie2d": 6, "apng": 6, "writer": 6, "pillow": 6, "fp": 6, "png": 6, "browser": 6, "to_jshtml": 6, "smaller": 6, "spread": 6, "blow": 6, "c_x": 6, "c_y": 6, "97": 6, "book": [6, 7], "harmon": 7, "exponenti": 7, "psi_j": 7, "hat": 7, "_k": 7, "psi_k": 7, "kind": 7, "functionspac": 7, "v_n": 7, "span": 7, "sai": 7, "u_n": 7, "element": 7, "best": 7, "_1": 7, "interpol": 7, "briefli": 7, "mention": 7, "focu": 7, "slightli": 7, "equat": 7, "clarifi": 7, "conveni": 7, "integr": 7, "achiev": 7, "diplai": [], "err": 7, "eq1": 7, "eq2": 7, "uhat": 7, "38": 7, "orthogon": 7, "foral": 7, "messi": [], "psi_i": 7, "_i": 7, "mass": 7, "orthonorm": 7, "monomi": 7, "good": 7, "ill": 7, "legendr": 7, "p_0": 7, "p_1": 7, "p_2": 7, "3x": 7, "p_": 7, "2j": 7, "xp_": 7, "p_j": 7, "p_i": 7, "map": 7, "physic": 7, "sooner": 7, "sec": 7, "introduct": 7, "variat": 7, "affin": 7, "revers": 7, "variabl": 7, "introduc": 7, "appear": 7, "cancel": 7, "highlight": 7, "ux": [], "strang": 7, "object": 7, "psi_": 7, "lagrang": 7, "ideal": [], "favour": 7, "16x": 7, "attempt": 7, "fine": 7, "yj": 7, "set_ylim": 7, "larg": 7, "under": 7, "shoot": 7, "bad": 7, "rung": 7, "phenomenon": 7, "idea": 7, "cluster": 7, "chebyshev": 7, "agreement": 7, "aors": [], "41": [], "ul": 7, "semilog": 7, "decreas": 7, "seri": 7, "pleas": 8, "organ": 8, "matmek": 8, "4270": 8, "matmek4270": 8, "mandatory1": 8, "badg": 8, "readm": 8, "own": 8, "collabor": 8, "discuss": 8, "among": 8, "student": 8, "encourag": 8, "me": 8, "email": 8, "github": 8, "usernam": 8, "obviou": 8, "who": 8, "necessarili": 8, "few": 8, "report": 8, "place": 8, "section": 8, "com": 8, "download": 8, "dimension": 8, "poisson2d": 8, "toward": 8, "test_convergence_poisson2d": 8, "test_interpol": 8, "admit": 8, "k_x": 8, "k_y": 8, "m_x": 8, "m_y": 8, "arbitrari": 8, "wave2d": 8, "stand": 8, "lectur": 8, "n_e": 8, "test_convergence_wave2d": 8, "wave2d_neumann": 8, "test_convergence_wave2d_neumann": 8, "folder": 8, "pdf": 8, "imaginari": 8, "imath": 8, "unit": 8, "version": 8, "kh": 8, "tild": 8, "test_exact_wave2d": 8, "neumannwav": 8, "gif": 8, "sure": 8, "mb": 8, "x_m": 7, "kx": [], "p_k": [], "regardless": [], "implicitli": 7, "origo": [], "comut": [], "imposs": 7, "rewrit": 7, "almost": 7, "ninner": 7, "quadratur": 7, "uv": 7, "uhatn": 7, "3333333333333335": 7, "000000000000002": 7, "routin": 7, "adapt": 7, "gradual": 7, "finer": 7, "longer": 7, "improv": 7, "toler": 7, "rel": 7, "absolut": 7, "49": 7, "user": 7, "bernstein": 7, "literatur": 7, "trial": 7, "0x1156c0ad0": [], "0x1170ac2d0": [], "0x1170d1910": [], "0x1171a0f50": [], "0x1171a0e50": [], "uh": 7, "ui": 7, "evid": 7, "oscil": 7, "allwai": 7, "w_n": 7, "hatusin": [], "minor": 7, "world": 7, "unc": 7, "58012275e": 7, "00000000e": 7, "55601020e": 7, "06409820e": 7, "52222377e": 7, "53926304e": 7, "93848441e": 7, "17438450e": 7, "64480816e": 7, "adjust": 7, "51283542": 7, "18309886": 7, "60209262": 7, "59154943": 7, "99795065": 7, "06103295": 7, "72004323": 7, "79577472": 7, "56234498": 7, "63661977": 7, "46105771": 7, "53051648": 7, "39059163": 7, "45472841": 7, "33876606": 7, "0x115e6ca50": [], "0x115e6cb90": [], "0x116d19750": [], "0x116cf7550": [], "0x116d4d250": [], "0x11ac7fc50": [], "0x11b0b54d0": [], "0x119c17f90": [], "0x11b13ae10": [], "0x11b13b310": [], "0x11841bb90": [], "0x1189c3a50": [], "0x1197f5610": [], "0x1197d7150": [], "0x1197d73d0": [], "experi": 7, "j_0": 7, "bessel": 7, "0x11229f4d0": [], "0x112302e10": [], "0x11238d7d0": [], "0x1123e0b90": [], "0x112400f10": [], "0x11a614b90": 3, "0x10f8df550": 3, "0x11b7136d0": 3, "0x11b71f510": 3, "0x11b750950": 3}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"matmek4270": 0, "book": 0, "lectur": [1, 2, 3, 4, 5, 6, 7], "note": 1, "matmek": 1, "4270": 1, "The": [1, 3, 5, 6, 7, 8], "finit": [1, 3, 4, 5], "differ": [1, 3, 4, 5], "method": [1, 2, 3, 4, 7], "variat": 1, "mandatori": [1, 8], "assign": [1, 7, 8], "3": 2, "vibrat": [2, 3], "equat": [2, 3, 4, 5, 6, 8], "manufactur": 2, "solut": [2, 8], "adjust": 2, "solver": [2, 4, 8], "4": 3, "matrix": [3, 5], "approach": 3, "problem": [3, 5, 8], "differenti": 3, "matric": 3, "first": 3, "deriv": [3, 5, 6], "solv": [3, 5], "us": 3, "fd": 3, "gener": 3, "stencil": 3, "5": 4, "wave": [4, 6, 8], "dirichlet": [4, 8], "fix": 4, "end": 4, "neumann": [4, 8], "loos": 4, "open": 4, "boundari": [4, 5, 6, 7], "No": 4, "discret": [4, 5], "remain": 4, "complic": 4, "initi": 4, "condit": [4, 5], "period": 4, "6": 5, "two": [5, 6], "dimension": [5, 6], "spatial": 5, "domain": 5, "cartesian": 5, "product": 5, "meshgrid": 5, "spars": 5, "broadcast": 5, "mesh": [5, 6], "function": [5, 6, 7], "row": 5, "major": 5, "comput": 5, "storag": 5, "2d": [5, 6], "form": 5, "poisson": [5, 8], "s": [5, 8], "vector": 5, "vec": 5, "trick": 5, "kroneck": 5, "partial": [5, 6], "laplac": 5, "oper": 5, "7": 6, "postprocess": 6, "interpol": 6, "One": 6, "lagrang": 6, "polynomi": 6, "other": 6, "tool": 6, "error": 6, "integr": 6, "over": 6, "plu": 6, "time": 6, "8": 7, "approxim": 7, "global": 7, "least": 7, "squar": 7, "galerkin": 7, "colloc": 7, "due": 8, "16": 8, "10": 8, "2023": 8, "implement": 8, "stationari": 8, "exact": 8, "dispers": 8, "coeffici": 8, "test": 8, "creat": 8, "movi": 8, "issu": 7, "weekli": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})