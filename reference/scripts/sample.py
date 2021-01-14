import pandas as pd

def do_something(df):
	print('I also ran')
	return df + 1

if __name__ == '__main__':

	df = pd.DataFrame({
		'A': [1,2,3,4],
		'B': [3,4,5,6],
	})

	df2 = do_something(df)

	print('I ran')



