#!/usr/bin/bash
import json

def process_label(i):
	res = dict()
	log = open("./yan/log_pi/log_{}.txt".format(i), 'rb')
#	name_line = log.readline()
#	name_line = name_line.split('/')
#	name = name_line[len(name_line) - 1].strip()
	lines = log.readlines()
	for l in lines:
		l = l.split(' ')
		if (l.count("app") > 0):
			l = l[len(l) - 1].split('/')
			name = l[len(l) - 1].strip()
		elif (l.count("speculatable") > 0):
			#label.write("{},{}\n".format(name, l[6]))
			tid = int(float(l[6]))
			sid = int(float(l[9]))
			if (res.has_key(sid)):
				res[sid][tid] = 1
			else:
				res[sid] = {tid:1}
	log.close()
	return name,res

def process_metric(raw_name, metric_file, label):
	raw_file = open("./yan/event_log/{}".format(raw_name), 'rb')
	#json.load(metric_file)
	metric_lines = raw_file.readlines()
	time_all = []
	write_lines = []
	for l in metric_lines:
		l = json.loads(l)
		if (l["Event"] == "SparkListenerTaskEnd"):
			task_metrics = l["Task Metrics"].values()
			tid = l["Task Info"]["Task ID"]
			sid = l["Stage ID"]
			index = l["Task Info"]["Index"]
			runtime = l["Task Info"]["Finish Time"] - l["Task Info"]["Launch Time"]
			if (len(time_all) < sid + 1):
				time_all.append([runtime])
			else:
				time_all[sid].append(runtime)
			write_lines.append(["{},{},{},{},".format(raw_name, tid, sid, index) + ','.join(str(e) for e in task_metrics[0:3]) + \
				',' + ','.join(str(e) for e in (task_metrics[3].values())) + ',' + ','.join(str(e) for e in (task_metrics[5].values()))\
				+ ',{},'.format(task_metrics[6]) + ','.join(str(e) for e in (task_metrics[7].values() + task_metrics[8].values()))\
				+ ',' + ','.join(str(e) for e in task_metrics[9:]) + ",{}".format(runtime), runtime, sid, index])
	for i in range(len(time_all)):
		time_all[i].sort()
		tot = 3 * len(time_all[i]) / 4
		time_all[i] = [float(sum(time_all[i][0:tot])) / tot, float(time_all[i][tot/2 + 4])]
	print time_all
	for l in write_lines:
		#print l[1]
		metric_file.write('{},{},{},{}\n'.format(l[0], l[1] / time_all[l[2]][0], l[1] / time_all[l[2]][1], int(label.has_key(l[2]) and label[l[2]].has_key(l[3]))))
			#task_metrics = l["Task Metrics"].values()
			#metric_file.write("{},{},{},{},".format(raw_name, l["Task Info"]["Task ID"], l["Task Metrics"]["Executor CPU Time"], l["Task Metrics"]["Result Size"]))
			#metric_file.write("{},{},".format(l["Task Metrics"]["Input Metrics"]))
			#metric_file.write("{},".format(l["Task Metrics"]["Updated Blocks"]))
			#metric_file.write("{},{},{},{},{},{},\n".format(for _ in l["Task Metrics"]["Shuffle Read Metrics"].values()))

def find_event(file_name):
	file = open("./test_data/metric_log/{}".format(file_name), 'rb')
	event = []
	metric_lines = file.readlines()
	for l in metric_lines:
		e = json.loads(l)["Event"]
		e = json.dumps(e)
		if (event.count(e) == 0):
			event.insert(0, e)
	return event

def main():
	#label = open("labels.csv", 'wb')
	#label.write("File Name,Task ID\n")
	metric = open("metrics_4.csv", 'wb')
	metric.write("File Name,Task ID,Stage ID,Index,Executor CPU Time,\
		Executor Run Time,Result Size,Bytes Read,Records Read,\
		Local Bytes Read,Total Records Read,Fetch Wait Time,\
		Remote Blocks Fetched,Remote Bytes Read,\
		Local Blocks Fetched,Memory Bytes Spilled,Records Written,\
		Bytes Written,Shuffle Write Time,Shuffle Records Written,Shuffle Bytes Written,\
		Result Serialization Time,Executor Deserialize Time,Disk Bytes Spilled,\
		JVM GC Time,Executor Deserialize CPU Time,Duration,Average Ratio,Medium Ratio,Label\n")
	for i in range(37, 38):
		metric_raw, label= process_label(i)
		print metric_raw
		process_metric(metric_raw, metric, label)
	#print dic[0]["Event"]
	#print json.dumps(dic[0], indent=4)
	#print find_event(metric_file_name)
	metric.close()
	return 0

main()

