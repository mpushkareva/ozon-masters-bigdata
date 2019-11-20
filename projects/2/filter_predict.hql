insert into hw2_pred select transform(id, if1, if2, if3, if4, 
if5, if6, if7, if8, if9, if10, if11, if12, if13, if14, cf2, cf3, cf4) 
using 'predict.py' as (id int, pred float) from hw2_test where if1 != '\\N' and 20 < int(if1) and int(if1) < 40;
