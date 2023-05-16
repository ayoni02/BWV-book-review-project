import random
import operator
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic

def opendata(a, nrows):
    df = pd.read_csv(a, nrows=nrows, sep=';', encoding='ISO-8859-1')
    return df

def split(df):
    n=len(df)
    N=list(range(n))
    random.seed(2023)
    random.shuffle(N)
    train=df.iloc[N[0:(n*4)//5]]
    test=df.iloc[N[(n*4)//5:]]
    return train, test
    
def red(df):
    reader = Reader(rating_scale=(1,10)) # rating scale range
    trainset = Dataset.load_from_df(df[['User-ID','ISBN','Book-Rating']],reader).build_full_trainset()
    items = trainset.build_anti_testset()
    return trainset, items

def rando(a):
    random.seed()
    rd = random.randint(0,len(a))
    return a[rd]

def mod(df, user, items):
    algo = KNNBasic()
    algo.fit(df)
    user_items = list(filter(lambda x: x[0] == user, items))
    recommendations = algo.test(user_items)
    recommendations.sort(key=operator.itemgetter(3), reverse=True)
    return recommendations
    
if __name__ == '__main__':
    data = opendata('Book reviews\BX-Book-Ratings.csv', nrows=20_000)
    books = opendata('Book reviews\BX_Books.csv', nrows=None)
    mapping_dict = books.set_index("ISBN")["Book-Title"].to_dict()
    train, test = split(data)
    users=test['User-ID'].tolist()
    trainset, items  = red(train)
    user = rando(users)
    recommendations = mod(trainset, user, items)
    print(f"User {user} Recommendation Top 5:")
    for r in recommendations[0:5]:
        try:  
            print(f" [Item] {mapping_dict[r[1]]}, [Estimated Rating] {round(r[3],3)}")
        except:
            continue