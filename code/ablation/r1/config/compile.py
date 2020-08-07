import re
rec_data="+MIPLOBSERVE:0,68220,1,3303,0,-1"
msgidRegex = re.compile(r',(\d)+,')
mo = msgidRegex.search(rec_data)
print(mo.group())