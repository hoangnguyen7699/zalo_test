from tempfile import tempdir
import numpy as np
import utils
from tabulate import tabulate

class DataFrame:
    """
    Two-dimensional data

    Data structure contains labeled columns (rows are labeled by integers starting from 0).
    Airthmetic operations  add, subtract,.. and binary operations and, or .. are can be be 
    performed on compatiple data frames with one columns.

    Basic ations on table sucha as : filter table by condition on columns, select data by column,
    add new columns and get the ith row of table are allowed on dataframe.

    Parameters
    ----------
    data: dict() 
        Python dictionary with the keys are the columns and the values are the lists containing entries of 
        the corresponding column.

    dtype: list, default None
        List contains the type of the data in table. The order of element of this list must follow the order of 
        columns in table from left to right. If None is given, all the columns will be considered of type str.

    Examples
    --------

    Constructing DataFrame from a dictionary.

    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data=d, dtype=[int, int])
    >>> df
       col1  col2
       ----  -----
         1     3
         2     4

    Notice that the type is int for both 2 columns
    >>> df.dtypes
    Column    Dtype
    --------  -------------
    col1       <class 'int'>
    col2       <class 'int'>

    Compatiple dataFrames with one 1 column can perform the bacsic airthmetic operations and binary operation. This operation 
    returns a DataFrame with one column of boolean type indicating the result of these operation. This result can be used as 
    the filler for function filterBy
    >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 4, 5]})
    >>> df.getByColumns('col1' > 1)
    col1 > 1
    --------------
    False
    True
    True

    >>> df.getByColumns('col1') + df.getByColumns('col2')
    col1 + col2
    --------------
    4
    6
    8
    """
    def __init__(self, data, dtype=None) -> None:
        if data is None or len(data.keys()) == 0:
            data = {}

        if dtype is not None:
            index_to_col = dict()      
            for index, column in enumerate(data.keys()):
                index_to_col[index] = column
            if len(dtype) != len(list(data.keys())): raise Exception('ERROR: The length of the dtype is not equal the length of columns')
            self._dtype= dict()
            for index, type in enumerate(dtype):
                self._dtype[index_to_col[index]]= type
        else:
            self._dtype= dict()
            for _,column in enumerate(data.keys()):
                self._dtype[column]= str

        self._data=dict()
        for _,column in enumerate(data.keys()):
            self._data[column]= utils.cast(data[column], self._dtype[column])
    



    def _to_numpy_array(self, data):
        lst = []
        for column in self.columns:
            lst.append(data[column])
        return np.array(lst)

    @property
    def columns(self):
        """
        Return a list of columns of the DataFrame.
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.columns
        ['col1', 'col2']
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
        ...                    'col3': [5, 6]})
        >>> df.columns
        ['col1', 'col2', 'col3']
        """
        return list(self._data.keys())

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.shape
        (2, 2)
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
        ...                    'col3': [5, 6]})
        >>> df.shape
        (2, 3)
        """
        return (len(self._data[list(self._data.keys())[0]]), len(self._data.keys()))
    
    @property
    def _num_rows(self):
        return self.shape[0]
    
    @property
    def _num_cols(self):
        return self.shape[1]

    @property
    def _col_to_index(self):
        col_to_index = dict()
        for index, column in enumerate(self._data.keys()):
            col_to_index[column] = index
        return col_to_index

    @property
    def _index_to_col(self):
        index_to_col = dict()      
        for index, column in enumerate(self._data.keys()):
            index_to_col[index] = column
        return index_to_col

    @property
    def _index(self):
        return np.ones(self._num_rows)

    @property
    def _data_np(self):
        lst = []
        for column in self.columns:
            lst.append(self._data[column])
        return np.array(lst)  

    @property
    def dtypes(self):
        """
        Print the dtype for each column.
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.dtypes
        Column    Dtype
        --------  -------------
        col1       <class 'int'>
        col2       <class 'int'>
        """
        temp=[]
        for col in self.columns:
            temp.append([col, self._dtype[col]])
        print(tabulate(temp, headers=['Column', 'Dtype']))
            
    
    def __str__(self, num=None):
        lst=[]
        for i in range( min(self._num_rows, num) if num is not None else self._num_rows):
            temp=[self._data[column][i] for column in self.columns]
            lst.append(temp)
        
        return tabulate(lst, headers=self.columns)
    
    def head(self, num_rows=5):
        """
        Print the the first num_rows row(s) of the DataFrame.

        Parameters
        ----------
        num_rows: int, default 5 
            integer indicates the number of printed rows  
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        """
        return self.__str__(num=num_rows)
    
    def loc(self, i):
        """
        Return the row of DataFrame at passed index .

        Parameters
        ----------
        i: int
            row lable
        
        Examples
        --------
        >>> d = {'col1': [1, 2], 'col2': [3, 4]}
        >>> df.loc(1)
        col1    col2
        --------  -------------
        2       4
        """
        if i > self._num_rows -1 : raise Exception('ERROR: Out of range')
        mask= np.zeros(self._num_rows, dtype=bool)
        mask[i]=True
        return self.filterBy(DataFrame(data={'mask':list(mask)}, dtype=[bool]))
    
    def set_value(self, row, column, value):
        """
        Set the passed value of the an entries at passed column and index row of the DataFrame

        Parameters
        ----------
        row: int
            row lable
        column: str
            column lable
        value: value
            new value
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        >>> df.set_value(0, 'col1', 5)
        >>> df.head()
        col1      col2
        --------  -------------
        5         3
        2         4
        """
        if row > self._num_rows -1 : raise Exception('ERROR: Out of range') 
        if column not in self.columns: raise Exception('ERROR: {} is not a column'.format(column))
        if not isinstance(value, self._dtype[column]) : raise Exception('ERROR: Values must be of type {}'.format(self._dtype[column]))
        self._data[column][row]=value

    def set_column_names(self, new_columns):
        """
        Set the new column names for DataFrame

        Parameters
        ----------
        new_columns: list
            The list of new names for the DataFrame. This list must has the size which is compatible with the 
            size of list of the old names.
        
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.columns
        ['col1', 'col2']
        >>> df.set_column_names(['col1_new', 'col2_new'])
        >>> df.columns
        ['col1_new', 'col2_new']
        """
        
        if not isinstance(new_columns, list):
            raise Exception('ERROR: Columns must be a list of strings')
        if len(new_columns) == 0 :
            raise Exception('ERROR: Columns must not be empty')
        if len(new_columns) != len(self.columns):
            raise Exception(
                'ERROR: New columns must has a same lengh as old columns')
        data = dict()
        dtype =dict()
        for index, column in enumerate(new_columns):
            if not isinstance(column, str):
                raise Exception('ERROR: {} is not a string'.format(column))
            data[column] = self._data[self._index_to_col[index]]
            dtype[column]=self._dtype[self._index_to_col[index]]
        self._data = data
        self._dtype= dtype
        del data
    
    def add_columns(self,column_names, dataframe):
        """
        Add the passed columns and values (dataframe) to DataFrame.If a adding column is identical with a column in Dataframe,
        it simply updates the value of this column

        Parameters
        ----------
        column_names: str or list
            The list of adding column(s) for the DataFrame.
        dataframe : DataFrame
            The DataFrame containing data of the adding column. This dataframe must have the shape compatible with the 
            size of the column_names 
        
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        >>> df.add_columns('col3', df.getByColumns('col1')+df.getByColumns('col2'))
        >>> df.head()
        col1      col2              col3
        --------  -------------     ------------
        5         3                 8
        2         4                 6
        """
        if not isinstance(column_names, list):
            column_names= [column_names]
        if not isinstance(dataframe, DataFrame): raise Exception('ERROR: Unsupport add columns with type(s) {}'.format(type(dataframe)))
        num_rows, num_cols= dataframe.shape
        if len(column_names) != num_cols : raise Exception('ERROR: Incompatible dataframe with names')
        if num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
        for i,  column in enumerate(dataframe.columns):
            self._data[column_names[i]]= dataframe._data[column]
            self._dtype[column_names[i]]= dataframe._dtype[column]
    

    def remove_columns(self,column_names):
        """
        Add the passed columns and values (dataframe) to DataFrame.If a adding column is identical with a column in Dataframe,
        it simply updates the value of this column

        Parameters
        ----------
        column_names: str or list
            The list of adding column(s) for the DataFrame.
        
        
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        >>> df.remove_columns('col2')
        >>> df.head()
        col1                   
        --------  
        5         
        2         
        """
        if not isinstance(column_names, list):
            column_names= [column_names]
    
        if len(column_names) > self._num_cols : raise Exception('ERROR: The number of removing of column is larger than the number of columns in DataFram')
        for column in column_names:
            try:
                del self._data[column]
            except KeyError:
                raise Exception('ERROR: {}'.format(column))
            

    def filterBy(self, filters):
        """
        Filter DataFrame based on the passed conditions.

        Parameters
        ----------
        filters: DataFrame
            The DataFrame containing only 1 column with boolean type. The entries of this DataFrame indicates the mask,
            hence must be compatiple with the origin DataFrame in term of number of rows.
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        3         5
        >>> df.filterBy(df.getByColumns('col1')> 1)
        >>> df.head()
        col1      col2              
        --------  -------------     
        2         4               
        3         5                 
        """
        if not isinstance(filters, DataFrame): raise Exception('ERROR: filters must be a dataframe with 1 column')
        if filters.shape[1] != 1 : raise Exception('ERROR: filters must be a dataframe with 1 column')
        mask= filters._data[filters.columns[0]]
        data_np= self._data_np[:, mask]
        return self._np_to_dataframe(data_np)
        


        

    def getByColumns(self, columns=None):
        """
        Get the DataFrame by the passed columns

        Parameters
        ----------
        columns: str or list, default None
            The returning column(s). If columns is None, return the entire DataFrame
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [3, 4, 5]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
        2         4
        3         5
        >>> df.getByColumns('col1)
        col1                    
        --------      
        1                      
        2
        3                       
        """

        if columns is None :
            columns= self.columns
        if not isinstance(columns, list):
            columns=[columns]
        data = dict()
        dtype = []
        if columns is None or len(columns) == 0:
            raise Exception('ERROR: Columns must not be non-empty')
        for column in columns:
            if column not in self._data:
                raise Exception('ERROR: {}'.format(column))
            data[column] = self._data[column]
            dtype.append(self._dtype[column])
        return DataFrame(data=data, dtype=dtype)
        




    def _np_to_dataframe(self, data_np):
        data = dict()
        for column in self.columns:
            data[column] = list(data_np[self._col_to_index[column]])
        return DataFrame(data=data, dtype=self._dtype)

    
    def drop_none(self):
        """
        Drop the rows that contain None value in any column.
       
        
        Examples
        --------
        >>> df = pd.DataFrame({'col1': [1, None], 'col2': [3, 4]})
        >>> df.head()
        col1      col2
        --------  -------------
        1         3
                  4
        >>> df.drop_none()
        col1      col2
        --------  -------------
        1         3                      
        """
        mask=list()
        for col in self.columns:
            mask_col=list()
            for i, val in enumerate(self._data[col]):
                if val is not None: mask_col.append(True)
                else: mask_col.append(False)
            mask.append(mask_col)
        mask= np.logical_and.reduce(np.array(mask, dtype=bool), axis=0)
        dropp_data= self._data_np[:, mask]
        data = dict()
        for column in self.columns:
            data[column] = list(dropp_data[self._col_to_index[column]])
        
        self._data=data


################# Basic airthmetic and binary operator overloading for DataFrames ####################


    def __lt__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support < for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] >= other:
                mask[i] = False
        return DataFrame(data={'{} < {}'.format(column_name,other): mask}, dtype=[bool])

    def __le__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support <= for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] > other:
                mask[i] = False
        return DataFrame(data={'{} <= {}'.format(column_name,other): mask}, dtype=[bool])

    def __eq__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support == for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] != other:
                mask[i] = False
        return DataFrame(data={'{} == {}'.format(column_name,other): mask}, dtype=[bool])

    def __ne__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support != for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] == other:
                mask[i] = False
        return DataFrame(data={'{} != {}'.format(column_name,other): mask}, dtype=[bool])

    def __gt__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support > for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] <= other:
                mask[i] = False
        return DataFrame(data={'{} > {}'.format(column_name,other): mask}, dtype=[bool])

    def __ge__(self, other):
        if self._num_cols > 1:
            raise Exception('ERROR: Only support >= for 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(self._num_rows):
            if lst[i] < other:
                mask[i] = False
        return DataFrame(data={'{} >= {}'.format(column_name,other): mask}, dtype=[bool])
    
    def __and__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: ERROR: Unsupport & operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            mask = np.ones(self._num_rows, dtype=bool)
            for i in range(self._num_rows):
                mask[i]=lst[i] & other_lst[i]
            return DataFrame(data={'{} & {}'.format(column_name,other_column_name): list(mask)}, dtype=[bool])

        raise Exception('ERROR: Unsupport operand type(s) for &: Dataframe and {}'.format(type(other)))
    
    def __or__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport or operator with dataframe with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            mask = np.ones(self._num_rows, dtype=bool)
            for i in range(self._num_rows):
                mask[i]=lst[i] | other_lst[i]
            return DataFrame(data={'{} | {}'.format(column_name,other_column_name): list(mask)}, dtype=[bool])
        raise Exception('ERROR: Unsupport operand type(s) for |: Dataframe and {}'.format(type(other)))
    
    def __xor__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport xor operator with dataframe with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            mask = np.ones(self._num_rows, dtype=bool)
            for i in range(self._num_rows):
                mask[i]=lst[i] ^ other_lst[i]
            return DataFrame(data={'{} ^ {}'.format(column_name,other_column_name): list(mask)}, dtype=[bool])
        raise Exception('ERROR: Unsupport operand type(s) for ^: Dataframe and {}'.format(type(other)))
    
    def __invert__(self):
        num_rows, num_columns= self.shape
        if num_columns != 1 : raise Exception('ERROR: Unsupport ~ operator with dataframe with more than 1 column')
        column_name = self.columns[0]
        lst = self._data[column_name]
        mask = np.ones(self._num_rows, dtype=bool)
        for i in range(num_rows):
            mask[i]= ~lst[i]
        return DataFrame(data={' ~ {}'.format(column_name): list(mask)}, dtype=[bool])
    

    def __add__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport + operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]+other_lst[i])
            return DataFrame(data={'{} + {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport + operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]+other)
            return DataFrame(data={'{} + {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])
    

    def __sub__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport - operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]-other_lst[i])
            return DataFrame(data={'{} - {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport - operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]-other)
            return DataFrame(data={'{} - {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])

    def __mul__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport * operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]*other_lst[i])
            return DataFrame(data={'{} * {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport * operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]*other)
            return DataFrame(data={'{} & {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])
    
    def __pow__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport ** operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]**other_lst[i])
            return DataFrame(data={'{} ** {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport ** operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]**other)
            return DataFrame(data={'{} ** {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])
    
    def __truediv__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport / operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]/other_lst[i])
            return DataFrame(data={'{} / {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport / operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]/other)
            return DataFrame(data={'{} & {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])
    

    def __floordiv__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport // operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]//other_lst[i])
            return DataFrame(data={'{} // {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport // operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]//other)
            return DataFrame(data={'{} // {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])
    
    def __mod__(self, other):
        if isinstance(other, DataFrame):
            other_num_rows, other_num_columns= other.shape
            if other_num_columns != 1 : raise Exception('ERROR: Unsupport mod operator with dataframes with more than 1 column')
            if other_num_rows != self._num_rows: raise Exception('ERROR: Incompatible dataframes')
            column_name = self.columns[0]
            lst = self._data[column_name]
            other_column_name = other.columns[0]
            other_lst = other._data[other_column_name]
            added_lst=[]
            for i in range(self._num_rows):
                added_lst.append(lst[i]%other_lst[i])
            return DataFrame(data={'{} mod {}'.format(column_name,other_column_name): added_lst}, dtype=[self._dtype[column_name]])
        else:
            _, num_columns= self.shape
            if num_columns != 1 : raise Exception('ERROR: Unsupport mod operator with dataframe with more than 1 column')
            column_name = self.columns[0]
            lst = self._data[column_name]
            added_lst=[]
            for i in range(self.num_rows):
                added_lst.append(lst[i]%other)
            return DataFrame(data={'{} mod {}'.format(column_name,other): added_lst}, dtype=[self._dtype[column_name]])




        
