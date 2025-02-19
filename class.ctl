load data
infile "C:\Users\abhin\OneDrive\Desktop\class.txt"
replace
into table Class
fields terminated by ','
(name,meets_at,room,fid)