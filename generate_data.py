import os


total = 0

i = 100
while True:
    i += 100
    command = "run-example SparkPi {} &> ./log/log_{}.txt".format(i,i)
    os.system(command)

    grep = "grep -c \"speculatable\" ./log/log_{}.txt".format(i)

    p = os.popen(grep,'r')
    lines = p.readlines()

    if not lines:
        break

    print lines
    data = int(lines[0].strip("\n"))


    total += data

    print total

    if total > 2000:
        break

print "Stragger collected : {}".format(total)
