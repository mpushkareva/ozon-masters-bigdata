insert into hw2_pred select transform(id, pred) 
using 'predict.py' as (id int, pred float) from hw2_test where 20 < int(if1) and int(if1) < 40;
