load data
infile "C:\Users\abhin\OneDrive\Desktop\student.txt"
replace
into table Student
fields terminated by ','
(snum,sname,major,standing,age)