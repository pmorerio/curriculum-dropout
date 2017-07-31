import cPickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns


def average(list_of_lists, last_n=10):
	allValues =  np.array(list_of_lists)
	allValues = np.sort(allValues)
	
	return np.mean(np.mean(allValues[:,-last_n:]))
	
	
fig = plt.figure()
fig.suptitle('Results')
plt.grid(b=True, which='both', color='0.65',linestyle='-')


#colors = ['r','r','b','b','k','g','g']
colors = ['r','b','k','g','k','r','g','m','b']
experiments = sorted(glob.glob('./experiments/*/'))
assert len(experiments) == len(colors)

for exp, color in zip(experiments, colors):
	

	if 'switch_do' in exp:
		continue

	
	#~ print exp.split('/')[-2], color
	runs=sorted(glob.glob(exp + '*accuracies.pkl'))
	accTrSet_m, accValSet_m, Xentr_m, gammaVal_m = [], [], [], []
	
	#print len(runs)
	
	for run in runs:
		f = open(run)
		accTrSet, accValSet, _, Xentr, gammaVal = cPickle.load(f)
		accTrSet_m.append(accTrSet)
		accValSet_m.append(accValSet)
		Xentr_m.append(Xentr)
		gammaVal_m.append(gammaVal)
	
	sns.set(color_codes=True)
	
	plt.subplot(3,1,1)
	sns.tsplot(Xentr_m, err_style=None, color=color)
	plt.ylabel('Xentropy')
	plt.xlabel('Iterations (x500)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	plt.subplot(3,1,2)
	#sns.tsplot(accTrSet_m, err_style=None,color=color)
	sns.tsplot(gammaVal_m, err_style=None,color=color)
	plt.ylabel('Prob')
	plt.xlabel('Iterations (x500)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	plt.subplot(3,1,3)
	sns.tsplot(accValSet_m, err_style=None, color=color)
	plt.ylabel('Acc. Test')
	plt.xlabel('Iterations (x500)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	print len(runs),"\t",exp.split('/')[-2],"\t",color, "\t",average(accValSet_m)
	
	
plt.show()

	
