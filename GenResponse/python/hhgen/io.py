import root_pandas as rpd
import pandas as pd
import numpy as np

## --------------------------------------------------------------------
def read_trees(fname,untagged,tagged,ncats,seed=12345,**kwargs):
    #reading contents of ROOT file into pandas dataframe
    dfs = [ rpd.read_root(fname,untagged,**kwargs) ] 
    #error handling: try and exception
    for icat in range(ncats):
        tname = tagged % icat
        try: 
            dfs.append( rpd.read_root(fname,tname,**kwargs) )
        except:
            dfs.append( pd.DataFrame() )
    
    for icat,idf in enumerate(dfs):  
        idf[ "cat" ] = icat
    df = pd.concat( dfs ) 
    
    df["bdtcat"] = ((df["cat"] -1) // 4) + 1
    df.loc[df["cat"] == 0, "bdtcat" ] = 0
    
    df["mxcat"] = ((df["cat"]- 1) % 4) + 1
    df.loc[df["cat"] == 0, "mxcat" ] = 0

    random_index = np.arange( df.shape[0] ) #df.shape[0] gives number of rows, returns array from 0 to no. of rows
    np.random.shuffle(random_index)
    
    #adding random_index as a column to df, setting it as index and sorting 
    df["random_index"] = random_index 
    df.set_index("random_index",inplace=True)
    df.sort_index(inplace=True)
        
    return df

