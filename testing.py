import CsvReader as cr

# Read file
df=cr.read_csv('airtravel.csv', dtype=[str, int, int, int])
print('print the first 3 rows of DataFrame')
print(df.head(3))


#Drop None values
df.drop_none()

print('----------------------------------------------------------------')
print('Get the columns of Dataframe')
print(df.columns)


print('----------------------------------------------------------------')
print('Get the shape of Dataframe')
print(df.shape)


print('----------------------------------------------------------------')
print('Get the type of each column of DataFrame')
print(df.dtypes)

print('----------------------------------------------------------------')
print("Filter where the column Month is 'MAR' or the column 1958 is 340")
print(df.filterBy((df.getByColumns('Month') == 'MAR') |(df.getByColumns('1958') == 340) ))


print('----------------------------------------------------------------')
print('Select column Month and 1958 ')
print(df.getByColumns(['Month', '1958']).head())


print('----------------------------------------------------------------')
print('Add new column 1961 which is the sum of 1959 and 1960')
print(df.add_columns('1961', df.getByColumns('1959') + df.getByColumns('1960')))
print(df.head())


print('----------------------------------------------------------------')
print('Change the entry at the first row and column 1958 from 340 to 341')
df.set_value(0, '1958', 341)
print(df.head())


print('----------------------------------------------------------------')
print('AREmove column 1961 ')
df.remove_columns('1961')
print(df.head())






