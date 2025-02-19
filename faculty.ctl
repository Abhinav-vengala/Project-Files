load data
infile "C:\Users\abhin\OneDrive\Desktop\faculty.txt"
replace
into table Faculty
fields terminated by ','
(fid,fname,deptid)