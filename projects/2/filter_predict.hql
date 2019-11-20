insert into hw2_pred select transform(id, if1, if2, if3, if4, 
if5, if6, if7, if8, if9, if10, if11, if12, if13, if14, cf2, cf3, cf4) 
using 'predict.py' as (id, pred) from hw2_test where 20 < if1 and if1 < 40;
