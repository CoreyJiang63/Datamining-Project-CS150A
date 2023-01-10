from pyspark import SQLContext, SparkContext
from pyspark.sql.functions import mean, col, udf
from pyspark.sql.types import *
import numpy as np
import pandas as pd

import findspark
findspark.init()

sc = SparkContext('local')
sqlc = SQLContext(sc)

#DATA ENCODING

train = sqlc.read.csv('data/train.csv', sep='\t', header=True)
test = sqlc.read.csv('data/test.csv', sep='\t', header=True)
dropped = ['Row','Step Start time','First Transaction Time','Correct Transaction Time','Step End Time','Step Duration (sec)',\
    'Correct Step Duration (sec)','Error Step Duration (sec)','Incorrects','Hints','Corrects']

for col_dropped in dropped:
    train = train.drop(col_dropped)
    test = test.drop(col_dropped)

def udf_process(func, typ):
    return(f.udf(func, typ))
def newcol(col_name, operation):
    global train, test
    train = train.withColumn(col_name, operation)
    test = test.withColumn(col_name, operation)
def dropcol(col_name):
    global train, test
    train = train.drop(col_name)
    test = test.drop(col_name)

# Process Hierarchy
def generate_unit(hierac): 
    return hierac.split(',')[0]  
def generate_section(hierac):
    return hierac.split(',')[1]

unit_sep = udf(generate_unit, StringType())
section_sep = udf(generate_section, StringType())
train = train.withColumn('Problem Unit', unit_sep(train['Problem Hierarchy']))
test = test.withColumn('Problem Unit', unit_sep(test['Problem Hierarchy']))
train = train.withColumn('Problem Section', section_sep(train['Problem Hierarchy']))
test = test.withColumn('Problem Section', section_sep(test['Problem Hierarchy']))
train = train.drop('Problem Hierarchy')
test = test.drop('Problem Hierarchy')

# Process KC
def generate_KCnum(kc):
    if kc:
        return kc.count('~~')+1
    else:
        return 0
KCcnt = udf(generate_KCnum, IntegerType())
train = train.withColumn('KC Count', KCcnt(train['KC(Default)']))
test = test.withColumn('KC Count', KCcnt(test['KC(Default)']))

# Process Opportunity
def opportunity_avg(opp):
    if not opp:
        return 0.0
    nums = [int(i) for i in opp.split('~~')]
    return sum(nums)/len(nums)*1.0

oppavg = udf(opportunity_avg, FloatType())
train = train.withColumn('Opportunity Avg', oppavg(train['Opportunity(Default)']))
test = test.withColumn('Opportunity Avg', oppavg(test['Opportunity(Default)']))

# Process Categorical
discrete_cols = ['Anon Student Id','Problem Name','Problem Unit','Problem Section','Step Name']
def vectorize(col):
    global train, test
    sid_dict = {}
    un = train.union(test)
    unique = un.select(col).distinct().collect()
    sids = [i[col] for i in unique]
    for idx, sid in enumerate(sids):
        sid_dict[sid] = idx

    def vect(sid):
        return sid_dict[sid]
    udfvect = udf(vect, IntegerType())
    train = train.withColumn('vectorized'+col, udfvect(col))
    train = train.drop(col)
    train = train.withColumnRenamed('vectorized'+col, col)
    test = test.withColumn('vectorized'+col, udfvect(col))
    test = test.drop(col)
    test = test.withColumnRenamed('vectorized'+col, col)

for c in discrete_cols:
    vectorize(c)

# FEATURE ENGINEERING

cfa = train.filter(train['Correct First Attempt']=='1')

# Group By Person

student_dict = {}
correct_studentGroup = cfa.groupby('Anon Student Id').count()
studentGroup = train.groupby('Anon Student Id').count()
student_joined = correct_studentGroup.join(studentGroup, studentGroup['Anon Student Id'] == correct_studentGroup['Anon Student Id'])

student_correct_rate = correct_studentGroup.join(studentGroup, studentGroup['Anon Student Id'] == correct_studentGroup['Anon Student Id']).drop(studentGroup['Anon Student Id'])
personal_rate = student_correct_rate.select('Anon Student Id', (correct_studentGroup['count']/studentGroup['count']))
personal_rate = personal_rate.withColumn('Personal Correct Rate', personal_rate['(count / count)']).drop('(count / count)')

tmp_sum, cnt = 0, 0
for row in personal_rate.collect():
    tmp_sum += row['Personal Correct Rate']
    cnt += 1
personal_mean = tmp_sum/cnt

for row in personal_rate.collect():
    student_dict[row['Anon Student Id']] = row['Personal Correct Rate']
def get_rate_from_id(idx):
    if idx in student_dict.keys():
        return float(student_dict[idx])
    else:
        return personal_mean
        
udf_getRate = udf(get_rate_from_id, FloatType())
train = train.withColumn('Personal Rate', udf_getRate('Anon Student Id'))
test = test.withColumn('Personal Rate', udf_getRate('Anon Student Id'))

# Group By Problem
def group_by_problem():   
    global train, test
    problem_dict = {}
    correct_problemGroup = cfa.groupby('Problem Name').count()
    problemGroup = train.groupby('Problem Name').count()
    problem_joined = correct_problemGroup.join(problemGroup, problemGroup['Problem Name'] == correct_problemGroup['Problem Name'])

    problem_correct_rate = correct_problemGroup.join(problemGroup, problemGroup['Problem Name'] == correct_problemGroup['Problem Name']).drop(problemGroup['Problem Name'])
    problem_rate = problem_correct_rate.select('Problem Name', (correct_problemGroup['count']/problemGroup['count']))
    problem_rate = problem_rate.withColumn('Problem Correct Rate', problem_rate['(count / count)']).drop('(count / count)')

    tmp_sum, cnt = 0, 0
    for row in problem_rate.collect():
        tmp_sum += row['Problem Correct Rate']
        cnt += 1
    problem_mean = tmp_sum/cnt

    for row in problem_rate.collect():
        problem_dict[row['Problem Name']] = row['Problem Correct Rate']
    def get_rate_from_prb(idx):
        if idx in problem_dict.keys():
            return float(problem_dict[idx])
        else:
            return problem_mean
            
    udf_getRate = udf(get_rate_from_prb, FloatType())
    train = train.withColumn('Problem Rate', udf_getRate('Problem Name'))
    test = test.withColumn('Problem Rate', udf_getRate('Problem Name'))

    
# Group By Step
def group_by_step():   
    global train, test
    step_dict = {}
    correct_stepGroup = cfa.groupby('Step Name').count()
    stepGroup = train.groupby('Step Name').count()
    step_joined = correct_stepGroup.join(stepGroup, stepGroup['Step Name'] == correct_stepGroup['Step Name'])

    step_correct_rate = correct_stepGroup.join(stepGroup, stepGroup['Step Name'] == correct_stepGroup['Step Name']).drop(stepGroup['step Name'])
    step_rate = step_correct_rate.select('Step Name', (correct_stepGroup['count']/stepGroup['count']))
    step_rate = step_rate.withColumn('Step Correct Rate', step_rate['(count / count)']).drop('(count / count)')

    tmp_sum, cnt = 0, 0
    for row in step_rate.collect():
        tmp_sum += row['Step Correct Rate']
        cnt += 1
    step_mean = tmp_sum/cnt

    for row in step_rate.collect():
        step_dict[row['Step Name']] = row['Step Correct Rate']
    def get_rate_from_stp(idx):
        if idx in step_dict.keys():
            return float(step_dict[idx])
        else:
            return step_mean
            
    udf_getRate = udf(get_rate_from_stp, FloatType())
    train = train.withColumn('Step Rate', udf_getRate('Step Name'))
    test = test.withColumn('Step Rate', udf_getRate('Step Name'))

# Group By KC
def group_by_kc():   
    global train, test
    kc_dict = {}
    correct_kcGroup = cfa.groupby('KC(Default)').count()
    kcGroup = train.groupby('KC(Default)').count()
    kc_joined = correct_kcGroup.join(kcGroup, kcGroup['KC(Default)'] == correct_kcGroup['KC(Default)'])

    kc_correct_rate = correct_kcGroup.join(kcGroup, kcGroup['KC(Default)'] == correct_kcGroup['KC(Default)']).drop(kcGroup['KC(Default)'])
    kc_rate = kc_correct_rate.select('KC(Default)', (correct_kcGroup['count']/kcGroup['count']))
    kc_rate = kc_rate.withColumn('KC Correct Rate', kc_rate['(count / count)']).drop('(count / count)')

    tmp_sum, cnt = 0, 0
    for row in kc_rate.collect():
        tmp_sum += row['KC Correct Rate']
        cnt += 1
    kc_mean = tmp_sum/cnt

    for row in kc_rate.collect():
        kc_dict[row['KC(Default)']] = row['KC Correct Rate']
    def get_rate_from_kc(idx):
        if idx in kc_dict.keys():
            return float(kc_dict[idx])
        else:
            return kc_mean
            
    udf_getRate = udf(get_rate_from_kc, FloatType())
    train = train.withColumn('KC Rate', udf_getRate('KC(Default)'))
    test = test.withColumn('KC Rate', udf_getRate('KC(Default)'))


group_by_problem()
group_by_step()
group_by_kc()

# OUTPUT

train = train.drop('KC(Default)')
test = test.drop('KC(Default)')
train.toPandas().to_csv('data/train_preprocessed.csv', sep='\t', header=True, index = False)
test.toPandas().to_csv('data/test_preprocessed.csv', sep='\t', header=True, index = False)