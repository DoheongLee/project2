import pandas as pd

df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')


# 1) Print the top 10 players in hits, batting average, homerun, and on-base percentage for each year from 2015 to 2018.
def getTopPlayers(year, stat):
    return df[(df['year'] == year)][['batter_name', stat]].nlargest(10, stat)


print("1)")


def printTopPlayersByYears(stats):
    for year in range(2015, 2019):
        print(f"<Year: {year}>")
        for stat in stats:
            topPlayers = getTopPlayers(year, stat)
            print(f"Top 10 ({stat})")
            print(topPlayers.to_string(index=False))


stats = ['H', 'avg', 'HR', 'OBP']
printTopPlayersByYears(stats)

# 2) Print the player with the highest war by position in 2018.
print("2)")


def getTopWarByPosition(df, year, positions):
    dfFiltered = df[(df['year'] == year) & (df['cp'].isin(positions))]
    topWarByPosition = dfFiltered.loc[dfFiltered.groupby('cp')['war'].idxmax()][['cp', 'batter_name', 'war']]
    topWarByPosition['cp'] = pd.Categorical(topWarByPosition['cp'], categories=positions, ordered=True)
    topWarByPositionSorted = topWarByPosition.sort_values('cp')

    return topWarByPositionSorted


positions_of_interest = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
topWarByPosition2018 = getTopWarByPosition(df, 2018, positions_of_interest)
print(topWarByPosition2018.to_string(index=False))

# 3) Among R, H, HR, RBI, SB, war, avg, OBP, SLG, which has the highest correlation with salary?
print("3)")
correlationStats = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
correlations = correlationStats.corr()['salary'].sort_values(ascending=False)
correlationsWithoutSalary = correlations.drop(labels=['salary'])
print(correlationsWithoutSalary.to_string())
highestCorrelationStat = correlationsWithoutSalary.idxmax()
print(f"The highest correlation with salary: {highestCorrelationStat}")
