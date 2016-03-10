import csv

def combineProducts(lines):
	prevPid="pid"
	names="names"
	value="values"
	f=open("../../data/attributes_join.csv", "wb")
	writer=csv.writer(f)
	writer.writerow(("pid", "names", "values"))
	for line in lines:
		if prevPid==line[0]:
			names=names+" "+line[1]
			value=value+" "+line[2]
		else :
			print prevPid
			writer.writerow((prevPid, names, value))
			names=line[1]
			value=line[2]
			prevPid=line[0]
	f.close()

lines = csv.reader(open("../../data/attributes.csv", "rb"))
lines.next()
lines=list(lines)
combineProducts(lines)