import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import datetime
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib.dates import DateFormatter
print 'Last submitted results (updated at {}):'.format(time.strftime('%X %x'))

def read_data():
	df = pd.DataFrame.from_csv('Apps/PlainText 2/538.csv')
	dfbetas = pd.DataFrame.from_csv('Apps/PlainText 2/betas538.csv')
	dfexitpolls = pd.DataFrame.from_csv('Apps/PlainText 2/538-exitpolls.csv')
	dfstates = df[df.index != 'US']
	dfstates = pd.concat([dfstates, dfbetas], axis=1, join='inner')
	
	dfexitpolls = dfexitpolls.iloc[::-1]
	for index, row in dfexitpolls.iterrows():
		dfstates.set_value(index,'Hreal',row['Hreal'])
		dfstates.set_value(index,'Treal',row['Treal'])
		dfstates.set_value(index,'PercIn',row['PercIn'])
	dfstates = dfstates.fillna(0)

	dfexitpolls = dfexitpolls.iloc[::-1]
	dfexitpolls.columns = ['Percentage in', 'Hillary', 'Trump']
	return (df, dfbetas, dfstates, dfexitpolls.iloc[::-1])

def election2016(df, dfbetas, dfstates, n_sims):
	national_average = df[df.index == 'US']\
		[['Hvotes','Tvotes','Jvotes','Mvotes','Ovotes']].as_matrix()[0]
	national_beta = dfbetas.get_value('US','Beta')
	dfstates["Hadv"] = dfstates.\
		apply(lambda row: row["Hreal"] - row["Hvotes"], axis=1)
	dfstates["Tadv"] = dfstates.\
		apply(lambda row: row["Treal"] - row["Tvotes"], axis=1)
	dfstates["Weight"] = dfstates["PercIn"] * dfstates["Votes"]
	dfstates["Hadv_weighted"] = dfstates["Hadv"] * dfstates["Weight"]
	dfstates["Tadv_weighted"] = dfstates["Tadv"] * dfstates["Weight"]
	if sum(dfstates["Weight"]) == 0:
		hadv = 0
		tadv = 0
	else:
		hadv = sum(dfstates["Hadv_weighted"])/sum(dfstates["Weight"])
		tadv = sum(dfstates["Tadv_weighted"])/sum(dfstates["Weight"])
	advantage = [hadv, tadv, 0, 0, -hadv-tadv]
	weight = sum(dfstates["Weight"])/538.
	draw_national_average = np.random.dirichlet(national_average*national_beta,n_sims)
	poll_offset = (draw_national_average - national_average)*(1-weight) + np.asarray(advantage)*weight
	sims = []
	for index, row in dfstates.iterrows():
		ratios = row.loc[['Hvotes','Tvotes','Jvotes','Mvotes','Ovotes']].as_matrix()
		real_ratios = np.concatenate((row.loc[['Hreal','Treal']].as_matrix(),\
									 np.asarray([0,0,1-sum(row.loc[['Hreal','Treal']].as_matrix())])))
		perc_in = row.get_value('PercIn')
		votes = row.loc[['Votes']][0]
		state_beta = row.loc[['Beta']][0]
		code = index
		state = row.loc[['State']][0]
		for sim in range(0,n_sims):
			ratios_offset = np.clip(ratios + poll_offset[sim],0,1)
			ratios_offset = ratios_offset / sum(ratios_offset)
			random_result = np.random.dirichlet((ratios_offset*state_beta).tolist(),1)[0]
			result = random_result*(1-perc_in) + perc_in*real_ratios
			electoral_votes = (result == max(result))*votes
			wins = (result == max(result))
			if perc_in > 0:
				real_ratios_to_store = list(real_ratios)[0:2]
			else:
				real_ratios_to_store = [np.nan,np.nan]
			sims.append([sim,state,code,votes] + real_ratios_to_store + \
						[perc_in] + result.tolist() + electoral_votes.tolist() + wins.tolist())
	sims = pd.DataFrame(sims)
	sims.columns = ['Simulation','State','Code','Votes','Hreal','Treal','PercIn',\
					'Hperc','Tperc','Jperc','Mperc','Operc','Hvotes','Tvotes','Jvotes',\
					'Mvotes','Ovotes','Hwins','Twins','Jwins','Mwins','Owins']
	agg_bysim = sims.groupby(['Simulation'])[['Hvotes','Tvotes','Jvotes','Mvotes']].sum()
	agg_bysim['Hwin'] = agg_bysim.apply(lambda row: (row[0] == max(row)) & (row[0] >= 270), axis=1)
	agg_bysim['Twin'] = agg_bysim.apply(lambda row: (row[1] == max(row)) & (row[1] >= 270), axis=1)
	hwinperc = len(agg_bysim[agg_bysim['Hwin'] == True])/float(len(agg_bysim))
	twinperc = len(agg_bysim[agg_bysim['Twin'] == True])/float(len(agg_bysim))
	helec = agg_bysim['Hvotes'].mean()
	telec = agg_bysim['Tvotes'].mean()
	agg_bystate = sims.groupby(['State'])[['Votes','PercIn','Hreal','Treal','Hwins','Twins','Mwins']].mean()
	
	winprobs = pd.DataFrame.from_csv('Apps/PlainText 2/winprobs.csv')
	newrow = pd.DataFrame([[datetime.datetime.now(), hwinperc, twinperc, helec, telec]])
	newrow.columns = ['time','hwins','twins','helec','telec']
	newrow = newrow.set_index(['time'])
	winprobs = winprobs.append(newrow)
	winprobs.to_csv('Apps/PlainText 2/winprobs.csv')
	return (agg_bysim,agg_bystate)

def win_histogram(agg_bysim):
	plt.figure(num=None, figsize=(16,6), dpi=80, facecolor='w', edgecolor='k')
	agg_bysim['Tvotes'].plot.hist(bins=range(0,540,2),alpha=0.5)
	agg_bysim['Hvotes'].plot.hist(bins=range(0,540,2),alpha=0.5)
	matplotlib.rc('font', **{'size': 20})
	plt.xlabel('Electoral votes')
	plt.ylabel('Number of simulations')
	plt.axvline(270,color='k')
	
def plot_winprobs_time():
	plt.figure(num=None, figsize=(16,6), dpi=80, facecolor='w', edgecolor='k')
	winprobs = pd.DataFrame.from_csv('Apps/PlainText 2/winprobs.csv')
	x = winprobs.index
	yh = winprobs['hwins']
	yt = winprobs['twins']
	axes = plt.gca()
	axes.set_ylim([0,100])
	plt.plot_date(x, yt*100, linestyle='solid', ms=4, mew=0, lw=3)
	plt.plot_date(x, yh*100, linestyle='solid', ms=4, mew=0, lw=3)
	timefmt = DateFormatter('%H:%M')
	axes.xaxis.set_major_formatter(timefmt)
	matplotlib.rc('font', **{'size': 20})
	plt.xlabel('Time')
	plt.ylabel('Win probability (%)')
	plt.axhline(50,color='k')
	plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.text(x[-1]-(x[-1]-x[0])*0.01, yh[-1]*100+6, 'Hillary: {:.1f}%'.format(yh[-1]*100), ha='right', va= 'bottom')
	plt.text(x[-1]-(x[-1]-x[0])*0.01, yt[-1]*100+6, 'Trump: {:.1f}%'.format(yt[-1]*100), ha='right', va= 'bottom')
	
def plot_elecvotes_time():
	plt.figure(num=None, figsize=(16,6), dpi=80, facecolor='w', edgecolor='k')
	winprobs = pd.DataFrame.from_csv('Apps/PlainText 2/winprobs.csv')
	x = winprobs.index
	yh = winprobs['helec']
	yt = winprobs['telec']
	axes = plt.gca()
	axes.set_ylim([0,500])
	plt.plot_date(x, yt, linestyle='solid', ms=4, mew=0, lw=3)
	plt.plot_date(x, yh, linestyle='solid', ms=4, mew=0, lw=3)
	timefmt = DateFormatter('%H:%M')
	axes.xaxis.set_major_formatter(timefmt)
	matplotlib.rc('font', **{'size': 20})
	plt.xlabel('Time')
	plt.ylabel('Expected electoral votes')
	plt.axhline(270,color='k')
	plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.text(x[-1]-(x[-1]-x[0])*0.01, yh[-1]+15, 'Hillary: {}'.format(int(yh[-1])), ha='right', va= 'bottom')
	plt.text(x[-1]-(x[-1]-x[0])*0.01, yt[-1]+15, 'Trump: {}'.format(int(yt[-1])), ha='right', va= 'bottom')

def plot_states(agg_bystate):
	agg_bystate['competitive'] = agg_bystate.apply(lambda row: abs(row['Hwins']-0.5), axis=1)
	agg_bystate = agg_bystate.sort_values(['competitive'], ascending=True)
	plt.figure(num=None, figsize=(16,32), dpi=80, facecolor='w', edgecolor='k')
	colors = plt.cm.coolwarm(agg_bystate['Twins'])
	X = 538-agg_bystate['Votes'].cumsum()
	Y = agg_bystate['Hwins']-0.5
	state = agg_bystate.index
	matplotlib.rc('font', **{'size': 20})
	plt.barh(X, Y, height=agg_bystate['Votes'], color=colors)
	plt.xlim(-0.5,0.5)
	plt.ylim(0,538)
	plt.yticks([])
	for x,y,s,v in zip(X+agg_bystate['Votes']/2. - 3,Y/2.,state,agg_bystate['Votes']):
		plt.text(y, x, '{} ({}%, {}EV)'.format(s,int(round((y+0.5)*100)), int(v)), ha='center', va= 'bottom')

def plot_map(agg_bystate):
	plt.figure(num=None, figsize=(16,12), dpi=80, facecolor='w', edgecolor='k')
	# Lambert Conformal map of lower 48 states.
	m = Basemap(llcrnrlon=-119,llcrnrlat=19,urcrnrlon=-64,urcrnrlat=49,
			projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
	# draw state boundaries.
	# data from U.S Census Bureau
	# http://www.census.gov/geo/www/cob/st2000.html
	shp_info = m.readshapefile('cb_2015_us_state_500k','states',drawbounds=True)
	# population density by state from
	# http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
	popdensity = agg_bystate['Hwins'].to_dict()
	# choose a color for each state based on population density.
	colors={}
	statenames=[]
	cmap = plt.cm.coolwarm
	vmin = 0; vmax = 1 # set range.
	for shapedict in m.states_info:
		statename = shapedict['NAME']
		# skip DC and Puerto Rico.
		if statename not in ['District of Columbia','Puerto Rico','American Samoa', \
							'United States Virgin Islands','Guam','Commonwealth of the Northern Mariana Islands']:
			pop = popdensity[statename]
			# calling colormap with value between 0 and 1 returns
			# rgba value.  Invert color range (hot colors are high
			# population), take sqrt root to spread out colors more.
			colors[statename] = cmap(1.-(pop-vmin)/(vmax-vmin))[:3]
		statenames.append(statename)
	# cycle through state names, color each one.
	ax = plt.gca() # get current axes instance
	for nshape,seg in enumerate(m.states):
		# skip DC and Puerto Rico.
		if statenames[nshape] not in ['District of Columbia','Puerto Rico','American Samoa', \
							'United States Virgin Islands','Guam','Commonwealth of the Northern Mariana Islands']:
		# Offset Alaska and Hawaii to the lower-left corner. 
			if statenames[nshape] == 'Alaska':
			# Alaska is too big. Scale it down to 35% first, then transate it. 
				seg = list(map(lambda (x,y): (0.35*x + 1100000, 0.35*y-1300000), seg))
			if statenames[nshape] == 'Hawaii':
				seg = list(map(lambda (x,y): (x + 5400000, y-1700000), seg))

			color = rgb2hex(colors[statenames[nshape]]) 
			poly = Polygon(seg,facecolor=color,edgecolor=color)
			ax.add_patch(poly)