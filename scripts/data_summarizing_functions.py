import pandas as pd
import numpy as np
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class DataSummarizer:
    """
    a class with list of data sumarizing methods.
    """
    def __init__(self) -> None:
        pass
    

    def generateModel(self, x, y):
        """
        a function that generates a regression model
        """
        # Creating regression model and fitting it

        # creating an object of LinearRegression class
        model = LinearRegression()

        # fitting the training data
        model.fit(x,y)
        print("Model created Sucessfully.")
        
        return model


    def clusterGenerator(self, df, selected_features, num_clusters, clust_name):
        """
        a function that generates a K means cluster value to rows.
        """
        selected_metrics = df[selected_features]

        # Creating normalized dataframes

        temp_arr = selected_metrics.loc[:, selected_features].values
        temp_arr = StandardScaler().fit_transform(temp_arr) # normalizing the features

        # Turning the temporary array of normalized values into a dataframe.

        normal_df = pd.DataFrame(temp_arr,columns=selected_features)

        # creating the clusters

        kmeans = KMeans(num_clusters)
        kmeans.fit(temp_arr)

        # Generating cluster value to each row

        identified_clusters = kmeans.fit_predict(temp_arr)
        # Adding the generated array of cluster values to the dataframe as a column

        data_with_clusters = df.copy()
        data_with_clusters[clust_name] = identified_clusters 
        
        return normal_df, data_with_clusters, kmeans

    
    def calcScore(self, df, cent, features, title):
        df[title] = df.apply(lambda df : np.linalg.norm(df[features] - cent), axis = 1)
        df2 = df[title]
        df_norm = ((df2-df2.min())/(df2.max()-df2.min()))*100
        df[title] = df_norm
        return df 


    def show_N_per_col(self, df, main, cols, N, target="top"):
        """
        sorts a column and shows top 10 values.
        """
        asc = False
        if(target == "bottom"):
            asc = True
        for col in cols:
            print("\nTop 10 customers based on "+col+"\n")
            print(df.sort_values(by=col, ascending=asc).loc[:,[main, col]].head(N))
        
    
    def bivariateAnalysis(self, df, cols, colors): 
        """
        it plots a scatter chart and runs correlation test
        """
        for i in range(len(cols)):
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(8, 4)) 
            sns.scatterplot(data = df, x=cols[i][0], y=cols[i][1], s=20, color=colors[i])
            print(self.corrMatrix(df, cols[i]))


    def showDistribution(self, df, cols, colors):
        """
        Distribution plotting function.
        """
        for index in range(len(cols)):
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(8, 4)) 
            sns.displot(data=df, x=cols[index], color=colors[index], kde=True, height=4, aspect=2)
            plt.title(f'Distribution of '+cols[index]+' data volume', size=20, fontweight='bold')
            plt.show()

            
    def plot_box(self, df:pd.DataFrame, col:str)->None:
        """
        Boxplot plotting function.
        """
        plt.boxplot(df[col])
        plt.title(f'Plot of {col}', size=20, fontweight='bold')
        ax = plt.gca()
        #ax.set_ylim(top = df[col].quantile(0.9999))
        #ax.set_ylim(bottom = 0)
        # show plot
        plt.show()

    def plot_box2(self, df:pd.DataFrame, columns, color:str)->None:
        """
        Boxplot plotting function.
        """
        fig = plt.figure(figsize =(10, 7))
        
        for col in columns:
            # Creating plot
            plt.boxplot(df[col])
            plt.title(f'Plot of {col}', size=20, fontweight='bold')
            ax = plt.gca()
            ax.set_ylim(top = df[col].quantile(0.9999))
            ax.set_ylim(bottom = 0)
            # show plot
            plt.show()


    def plot_pie(self, df, col, title):
        """
        pie chart plotting function.
        """
        # Wedge properties
        wp = { 'linewidth' : 1, 'edgecolor' : "black" }

        # Creating autocpt arguments
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} g)".format(pct, absolute)
        
        fig, ax = plt.subplots(figsize =(10, 7))
        wedges, texts, autotexts = ax.pie(df[col[1]],
                                    autopct = lambda pct: func(pct, df[col[1]]),
                                    labels = df[col[0]].to_list(),
                                    startangle = 90,
                                    wedgeprops = wp,)

        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title(title)


    def percent_missing(self, df):
        """
        this function calculates the total percentage of missin values in a dataset.
        """
        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")


    def summ_columns(self, df, unique=True):
        """
        shows columns and their missing values along with data types.
        """
        df2 = df.isna().sum().to_frame().reset_index()
        df2.rename(columns = {'index':'variables', 0:'missing_count'}, inplace = True)
        df2['missing_percent_(%)'] = round(df2['missing_count']*100/df.shape[0])
        data_type_lis = df.dtypes.to_frame().reset_index()
        df2['data_type'] = data_type_lis.iloc[:,1]
        
        if(unique):
            unique_val = []
            for i in range(df2.shape[0]):
                unique_val.append(len(pd.unique(df[df2.iloc[i,0]])))
            df2['unique_values'] = pd.Series(unique_val)
        return df2


    def get_top_n(self, df, colname, num, globalDict):
        """
        a function that groups a column and return the top n groups based on member count
        """
        queryDf = lambda q: sqldf(q, globalDict)
        query = 'SELECT "'+colname+'", count(*) as user_count FROM '+df+' WHERE "'+colname+'" != "undefined" group by "'+colname+'" order by user_count DESC LIMIT '+str(num)
        return queryDf(query)


    def manByHandset(self, lis, dfname, globalDict):
        """
        a function that returns top three handsets from top three manufacturers
        """
        queryDf = lambda q: sqldf(q, globalDict)
        for man in lis:
            query = 'SELECT "Handset Manufacturer", "Handset Type", count(*) as num_users \
            FROM '+dfname+'\
            WHERE "Handset Manufacturer" = "'+man+'" \
            group by "Handset Type" \
            order by num_users DESC \
            LIMIT 5'
            print(queryDf(query),'\n')


    def find_agg(self, df, group_columns, agg_columns, agg_metrics, new_columns):
        """
        a function that returns a new dataframe with aggregate values of specified columns.
        """
        new_column_dict ={}
        agg_dict = {}
        for i in range(len(agg_columns)):
            new_column_dict[agg_columns[i]] = new_columns[i]
            agg_dict[agg_columns[i]] = agg_metrics[i]

        new_df = df.groupby(group_columns).agg(agg_dict).reset_index().rename(columns=new_column_dict)
        return new_df


    def combineColumns(self, df, col1, col2, new_name, rem=False):
        """
        combines two numerical variables and create new variable.
        """
        df[new_name] = df[col1]+df[col2]
        if(rem):
            df.drop([col1, col2], axis = 1, inplace = True)


    def generateFreqTable(self, df, cols, range):
        """
        generate a freqeuncy table
        """
        for col in cols:
            print(df[col].value_counts().iloc[:range,])


    def summary_one(self, df, cols):
        """
        calculate range, max, count, and min.
        """
        df2 = df[cols]
        data_types_dict = {'max': float, 'min':float}
  
        df_sum = df2.max().to_frame().reset_index().rename(columns={"index":"variables",0:"max"})
        df_sum["min"] = df2.min().to_frame().reset_index().iloc[:,1]
        df_sum= df_sum.astype(data_types_dict)
        df_sum['range'] = df_sum['max'] - df_sum['min']
        df_sum["count"] = df2.count().to_frame().reset_index().iloc[:,1]
        return df_sum


    def summary_two(self, df, cols):
        """
        calculate central tendency measures.
        """
        df2 = df[cols]
        df_sum = df2.mean().to_frame().reset_index().rename(columns={"index":"variables",0:"mean"})
        df_sum["median"] = df2.median().to_frame().reset_index().iloc[:,1]
        df_sum["mode"] = df2.mode().iloc[:,1]
        return df_sum


    def summary_three(self, df, cols):
        """
        calculate dispersion measures
        """
        df2 = df[cols]
        df_sum = df2.std().to_frame().reset_index().rename(columns={"index":"variables",0:"std"})
        df_sum["var"] = df2.var().to_frame().reset_index().iloc[:,1]
        return df_sum


    def bar_graph(self, df, cols, x_ax):
        """
        graphical univariate analysis function. bar chart.
        """
        plot_df = df[cols]
        plt.figure(figsize=(25, 12))
        sns.countplot(x= x_ax, data=plot_df)



    def topDecile(self, df, group,deci, cols, metric, name, range):
        """
        function that aggregates based on top n deciles.
        """
        df['Decile'] = pd.qcut(df['session_dur'], 10, labels=False)
        aggr_n = self.find_agg(df, group, cols, metric, name)
        aggr_n = aggr_n.loc[aggr_n['Decile'] < range[1]+1]
        aggr_n = aggr_n.loc[aggr_n['Decile'] > range[0]-1]
        return aggr_n


    def corrMatrix(self, df, cols):
        """
        a function that generates correlation matrix as a table.
        """
        relation = df[cols].corr()
        return relation