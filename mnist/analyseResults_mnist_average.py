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
fig.suptitle('Results' )

#colors = ['m','b','r','k','g']
colors = ['r','b','k','g']
experiments = sorted(glob.glob('./experiments/**/'))
assert len(experiments) == len(colors)

for exp, color in zip(experiments, colors):
	
	#~ if '_small' in exp:
	#~ if 'big' in exp:
		#~ continue

	runs=sorted(glob.glob(exp + '*accuracies.pkl'))
	accTrSet_m, accValSet_m, accTestSet_m, Xentr_m = [], [], [], []
	
	#print runs
	#print len(runs)
	
	for run in runs:
		f = open(run)
		accTrSet, accValSet, accTestSet, Xentr = cPickle.load(f)
		accTrSet_m.append(accTrSet)
		accValSet_m.append(accValSet)
		accTestSet_m.append(accTestSet)
		Xentr_m.append(Xentr)
	
	sns.set(color_codes=True)
	
	plt.subplot(3,1,1)
	sns.tsplot(Xentr_m, color=color)
	plt.ylabel('Xentropy')
	plt.xlabel('Iterations (x100)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	plt.subplot(3,1,2)
	sns.tsplot(accTrSet_m, color=color)
	plt.ylabel('Acc. Training')
	plt.xlabel('Iterations (x100)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	plt.subplot(3,1,3)
	sns.tsplot(accTestSet_m, color=color)
	plt.ylabel('Acc. Test')
	plt.xlabel('Iterations (x100)')
	plt.grid(b=True, which='both', color='0.65',linestyle='-')

	
	print len(runs),"\t",exp.split('/')[-2],"\t",color, "\t",average(accValSet_m)


	
plt.show()


	
