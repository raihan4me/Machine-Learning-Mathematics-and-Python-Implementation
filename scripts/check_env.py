import sys
import numpy, scipy, pandas, sklearn, matplotlib, joblib
import mlmath
print("python      ", sys.version.split()[0])
for mod in (numpy, scipy, pandas, sklearn, matplotlib, joblib):
    print(f"{mod.__name__:12s}", mod.__version__)
print("mlmath from ", mlmath.__file__)
print("mpl backend ", matplotlib.get_backend())
