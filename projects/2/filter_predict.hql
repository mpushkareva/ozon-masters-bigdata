add file projects/2/model.py;
add file projects/2/train.py;
add file 2.joblib;
add file projects/2/predict.py;
insert into my_hw2_pred select transform(*) 
using 'predict.py' as (id, pred) from my_hw2_test where if1 > 20 and if1 < 40;
