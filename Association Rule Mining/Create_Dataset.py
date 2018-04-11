import pandas as pd

FILE_NAME = 'World_Happiness_2015.csv'


def get_quantile(val, quant, num_quants):
    for i in range(1, num_quants + 1):
        if val <= quant[i / float(num_quants)]:
            return i
    return 0


# returns the dataset as a dataframe
def load_data(filename):

    quantiles = 5

    df = pd.read_csv(filename)
    quant = df.quantile([x/float(quantiles) for x in range(1, quantiles + 1)])

    for name in ['Happiness Score', 'GDP Per Capita', 'Life Expectancy', 'Freedom', 'Gov Trust', 'Generosity']:
        df[name] = df[name].apply(lambda x: get_quantile(x, quant[name], quantiles))

    return df[['Region', 'Happiness Score', 'GDP Per Capita', 'Life Expectancy', 'Freedom', 'Gov Trust', 'Generosity']]


if __name__ == "__main__":

    df = load_data(FILE_NAME)

    for name in ['Happiness Score', 'GDP Per Capita', 'Life Expectancy', 'Freedom', 'Gov Trust', 'Generosity']:
        print("{}\n".format(df[name].value_counts().sort_values()))

    df.to_csv('World_Happiness_Discrete', sep=',')