from projectFunctions import *

df = dataPrep("data/asx_13Jul20.csv")

data_grouped = df.groupby('Symbol')
for name,group in data_grouped:    
    print(name + '  ' + str(group['Symbol'].count()))

temp = pd.unique(df['Symbol'])[47:]

#data = df[df['Symbol'].isin(['A2M.AX','AGL.AX','ALL.AX','ALQ.AX'])]
data = df[df['Symbol'].isin(temp)]
#data = df

print(pd.unique(data['Symbol']))

syms = data['Symbol'].unique()
CPU_CORES = mp.cpu_count() * 2
counter = 1
for i in syms:
    print(counter, " : ",i)
    start_time = time.time()
    temp = data[data['Symbol'] == i]    
    result = applyParallel_npsplit(np.array_split(temp,CPU_CORES), createXYarrays)
    gc.collect()
    gc.collect()
    saveXYtoDisk(result,folder="data/",fname="Set" + str(i))
    print("--- %s seconds ---" % (time.time() - start_time))
    counter += 1



