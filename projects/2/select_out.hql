INSERT OVERWRITE LOCAL DIRECTORY 'mpushkareva_hiveout'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
SELECT * FROM hw2_pred;
