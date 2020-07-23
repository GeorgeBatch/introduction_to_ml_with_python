import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '/Users/macbookair/Desktop/CompareEuropeGroup/reports')
import datamanipulation as dm
import variable_selection as vs

# ----------------------------------------------------------------------------------------------------------
#                                           Creating df
# ----------------------------------------------------------------------------------------------------------

test_dict = {'A': [1.0, 2.3, 3, 5],
			 'B': ['asd', 'asdf', 'asd', 'qwerty'],
			 'C': [6, 0, 3, 5],
			 'D': [1.1234, 4.3543, 7.234, 10],
			 'E': [3, 0, 16, 45],
			 'F': [4, 5, 56, 34]}

my_df = pd.DataFrame(data=test_dict)
print('Original data frame')
print(my_df)
print()

# ----------------------------------------------------------------------------------------------------------
#                                           Subsetting df
# ----------------------------------------------------------------------------------------------------------

# # subsetting the df
# non_zero_df = my_df[(my_df['E'] != 0) | (my_df['C'] != 0)]
# print('Subsetted original data frame')
# print(non_zero_df)
# print()
#
# # resetting the indices
# non_zero_df.reset_index(drop=True, inplace=True)
# print(non_zero_df)
# print()

# ----------------------------------------------------------------------------------------------------------
#                                           Encoding df
# ----------------------------------------------------------------------------------------------------------

my_df_encoded, numeric_fields, dict_categorical_fields = dm.one_hot_encoding(
	df=my_df, drop=False, return_fields=True)

print('Encoded original data frame')
print(my_df_encoded)
print()

list_of_categorical_fields = [field for field in list(my_df_encoded.columns) if field not in numeric_fields]
print('list_of_categorical_fields: \n{}\n'.format(list_of_categorical_fields))

# ----------------------------------------------------------------------------------------------------------
#                                           Splitting for ml:
# ----------------------------------------------------------------------------------------------------------


X, y = dm.splitting_encoded_data_for_ml(df=my_df_encoded, variable='A')
print('Data: \n{}\n'.format(X))
print('Response: \n{}\n'.format(y))

X_categorical = X[:, -len(list_of_categorical_fields):]
X_numeric = X[:, :-len(list_of_categorical_fields)]



print('{} X_categorical: \n{}\n'.format(X_categorical.shape, X_categorical))
print('{} X_numeric: \n{}\n'.format(X_numeric.shape, X_numeric))

X_product = X_numeric



for field in list_of_categorical_fields:
	X_binned = np.array(my_df_encoded[field])
	X_binned = np.expand_dims(X_binned, axis=1)

	print('Categorical field, binning, shape')
	print(field, X_binned.shape)
	print(X_binned)
	print()

	# see page 225 of Intro to ML with python (element by element multiplication)
	print('Product of X_binned * X_numeric (should preserve the dimension of '
		  'X_numeric)'.format((X_binned * X_numeric).shape))
	print(X_binned * X_numeric)

	X_product = np.hstack([X_product, X_binned * X_numeric])
	print('{} X_product after adding {}: \n{}\n'.format(X_product.shape, field, X_product))

print('{} final X_product: \n{}\n'.format(X_product.shape, X_product))
print('X_product should have {} columns.'.format(X_numeric.shape[1] * (X_categorical.shape[1] + 1)))
print('X_product has {} columns.'.format(X_product.shape[1]))


