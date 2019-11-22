create table hw2_pred (id int, pred float)
row format delimited fields terminated by '\t' 
lines terminated by '\n'
stored as textfile
location 'mpushkareva_hw2_pred';
