import pandas as pd

sentence ="Please respond with yes or no"
if (sentence == ("Yes" or "Yeah" or "Yup")):
	print (sentence)



'''
df = pd.read_csv("G:\\Git_code\\AudioAnalysis\\TextClassification_Commodus\\Input\\ChrisP_coded_data.csv",usecols=['Document ID','Rule 3: Credit check','Rule 3 Proof Phrase'])
df= df[df['Rule 3: Credit check'] == "Identified"]
print(df)

df_2= pd.read_csv("G:\Git_code\AudioAnalysis\TextClassification_Commodus\Output\Test_ChrisP_data_creditCheck_60.csv",usecols=['ID','Sentence'])
print(df_2)'''