from pyspark import SQLContext, SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import pandas
import findspark
findspark.init()
sparkc = SparkContext('local')
sqlc = SQLContext(sparkc)

train_loc = 'data/train.csv'
train = sqlc.read.csv(train_loc, sep='\t', header=True)
test_loc = 'data/test.csv'
test = sqlc.read.csv(test_loc, sep='\t', header=True)
#****************************************REMOVE****************************************#
#Because several columns are not provided by test, they should not be features to be considered, so we simply remove them.
columns_to_drop = ['Row','Step Start time','First Transaction Time','Correct Transaction Time','Step End Time','Step Duration (sec)',\
    'Correct Step Duration (sec)','Error Step Duration (sec)','Incorrects','Hints','Corrects']
train = train.drop(*columns_to_drop)
test = test.drop(*columns_to_drop)
# Divide column 'Problem Hierarchy' into two columns: 'Problem Unit' and 'Problem Section'.
def sep_hier_unit(str): 
    return str.split(',')[0]
    
def sep_hier_section(str):
    return str.split(',')[1]

udfsep_hier_unit = F.udf(sep_hier_unit, StringType())
udfsep_hier_section = F.udf(sep_hier_section, StringType())

train = train.withColumn('Problem Unit', udfsep_hier_unit('Problem Hierarchy'))
train = train.withColumn('Problem Section', udfsep_hier_section('Problem Hierarchy'))
test = test.withColumn('Problem Unit', udfsep_hier_unit('Problem Hierarchy'))
test = test.withColumn('Problem Section', udfsep_hier_section('Problem Hierarchy'))

train = train.drop('Problem Hierarchy')
test = test.drop('Problem Hierarchy')

# category 
def encode(column):
    global train,test
    ref_dict = {}
    ori_temp = train.union(test).select(column).distinct().collect()
    temp = []
    for item in ori_temp:
        temp.append(item[column])
    index = 1
    for item in temp:
        ref_dict[item] = index
        index += 1
    
    def innerencode(str):
        return ref_dict[str]
    
    udfinnerencode = F.udf(innerencode,IntegerType())
    
    train = train.withColumn('New '+column, udfinnerencode(column))
    train = train.drop(column)
    train = train.withColumnRenamed('New '+column, column)
    test = test.withColumn('New '+column, udfinnerencode(column))
    test = test.drop(column)
    test = test.withColumnRenamed('New '+column, column)

column_to_encode = ['Anon Student Id','Problem Name','Problem Unit','Problem Section','Step Name']
for time in range(len(column_to_encode)):
    encode(column_to_encode[time])
    
#***************************************CALCULATE**************************************#
def KC_count(str):
    if not str:
        return 0
    else:
        return str.count('~~')+1

def Opp_avg(str):
    if not str:
        return 0.0
    else:
        sum = 0
        count = 0
        strlist = str.split('~~')
        for item in strlist:
            sum += eval(item)
            count += 1
        return float(sum/count)

udfKC_count = F.udf(KC_count, IntegerType())
udfOpp_avg = F.udf(Opp_avg, FloatType())


train = train.withColumn('KC Count', udfKC_count('KC(Default)'))
test = test.withColumn('KC Count', udfKC_count('KC(Default)'))
train = train.withColumn('Opportunity Average', udfOpp_avg('Opportunity(Default)'))
train = train.drop('Opportunity(Default)')
test = test.withColumn('Opportunity Average', udfOpp_avg('Opportunity(Default)'))
test = test.drop('Opportunity(Default)')