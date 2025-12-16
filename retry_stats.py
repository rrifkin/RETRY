#! /usr/bin/env python

import pandas as pd

def retry_stats (file):
      df = pd.read_csv(file)
      df.head()



if __name__ == "__main__":
    import sys
    retry_stats (str(sys.argv[1]), str(sys.argv[2]))