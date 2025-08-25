import pandas


def read_monitor_file(filename: str) -> pandas.DataFrame:
    df = pandas.read_csv(filename, sep=R'\s*\|\s*', header=0, skiprows=[1], index_col=False, engine='python')
    df = df.drop([df.columns[0], df.columns[-1]], axis=1)
    return df