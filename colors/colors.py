import json
from pprint import pprint

def load_colors(filename):
	with open(filename) as data_file:    
	    data = json.load(data_file)

	colors = {}
	for i in range(len(data)):
		colors[str(data[i]['name'])] = ((data[i]['rgb']['r'], data[i]['rgb']['g'], data[i]['rgb']['b']), 500)
		# print(str(data[i]['name']), (data[i]['rgb']['r'], data[i]['rgb']['g'], data[i]['rgb']['b']))

	return colors