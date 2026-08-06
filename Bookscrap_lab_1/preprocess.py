import pandas as pd 

df = pd.read_csv("raw_books.csv") #Reading the csv file into a pandas dataframe

print(df.head()) #Verify the data is loaded correctly
print(df.info()) #Verify the data types 
print(df.isnull().sum()) #Directly Check for missing values

print("Duplicate UPC's: ", df.duplicated(subset='upc').sum()) #Check for duplicate UPC

df= df.drop_duplicates(subset='upc') #Remove duplicate UPC

text_col=['title', 'category', 'price', 'rating', 'availability', 'description'] #List of columns
for col in text_col:
    df[col] = (df[col].fillna('') #Fill missing values with empty string
                .str.strip() #Remove leading and trailing whitespace
                .str.lower() #Convert to lowercase
                .str.replace(r'\s+', ' ', regex=True) #Replace multiple spaces with a single space
            )
df['description'] = df['description'].str.replace(r'\n', ' ', regex=True) #Replace new line characters with a space

df['price'] = df['price'].str.replace('£', '', regex=False).astype(float) #Remove the pound sign and convert to float

rating_map = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5
}
df['rating'] = df['rating'].map(rating_map) #Map the rating text to numerical values

df['stock_count'] = (df["availability"].str.extract(r"\((\d+) available\)") #Extract the number of books in stock
                .astype(int) #Convert to integer
            )

df["description_word_count"]=(df["description"].str.split().str.len()) #Count the number of words in the description

def price_range(price):
    if price < 10:
        return 'LOW PRICE'
    elif 10 <= price < 20:
        return 'REASONABLE'
    elif 20 <= price < 30:
        return 'AFFORDABLE'
    else:
        return 'COSTLY'
df['price_range'] = df['price'].apply(price_range) #Apply the price range function to the price column

df["recommended"] = (df["rating"] >= 4) & (df["price"] < 30) #Create a new column to indicate if the book is recommended

df.to_csv("cleaned_books.csv", index=False) #Save the cleaned dataframe to a new csv file
print(df.head()) #Verify the cleaned data
print(df.info()) #SVerify the data types of the cleaned data