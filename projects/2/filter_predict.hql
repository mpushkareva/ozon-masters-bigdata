add file projects/2/model.py;
add file projects/2/train.py;
add file 2.joblib;
add file projects/2/predict.py;
insert into hw2_pred select transform(*) 
using 'predict.py' as (id, pred) from hw2_test where 20 < if1 and if1 < 40;
