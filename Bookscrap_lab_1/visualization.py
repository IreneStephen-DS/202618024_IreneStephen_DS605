import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df=pd.read_csv("cleaned_books.csv") #Reading the cleaned csv file into a pandas dataframe
print(df.head()) #Verify the data is loaded correctly

print("Dataset: ")
print(df.info()) #Verify the data types

print("Summary:")
print(df.describe()) #Get summary statistics of the dataset

#Four PLots to visualize the data

#Histogram of Book Prices (continous variable)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Book Prices')
plt.xlabel('Price')
plt.ylabel('No of Books')

plt.show()

#Countplot of Book Ratings (categorical variable)
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="rating")

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Books")

plt.show()             

#Average Price by Category (categorical variable)
category_price = (
    df.groupby("category")["price"]
    .mean()
    .sort_values(ascending=False)
)
print(category_price)
plt.figure(figsize=(12,6))

category_price.plot(
    kind="bar"
)

plt.title("Average Price by Category")
plt.xlabel("Category")
plt.ylabel("Average Price")

plt.xticks(rotation=90)

plt.show()

#Relationship between Price and Rating (continuous vs categorical variable)
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="rating",
    y="price"
)

plt.title("Price vs Rating Relationship")
plt.xlabel("Rating")
plt.ylabel("Price")

plt.show()

#Word Cloud of Book Descriptions (text data)
text = " ".join(
    df["description"]
    .dropna()
    .astype(str)
)
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)
plt.axis("off")
plt.title("Most Common Words in Book Descriptions")
plt.show()

#Category patterns
category_count = (
    df["category"]
    .value_counts()
)
print("Books per Category:")
print(category_count)

#Highest Rated Books
high_rated = df[df["rating"] >= 4]
print("Highly Rated Books:")
print(
    high_rated[
        ["title","rating","category"]
    ]
)

#Stock patterns
print("Stock Statistics:")
print(df["stock_count"].describe())

#missing values
print("\nMissing Values:")
print(df.isnull().sum())