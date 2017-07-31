#!/bin/bash

for i in {10..20}; 
	do 	#python evaluate_dm.py switch_do_05 $i ;  
	 	#python evaluate_dm.py switch_do_10 $i ;  
	 	#python evaluate_dm.py switch_do_20 $i ;  
	 	#python evaluate_dm.py switch_do_50 $i ;  
	 	#python evaluate_dm.py switch_do_100 $i ;  
	 	python evaluate_dm.py do $i ;  
		python evaluate_dm.py no_do $i ;  
	 	python evaluate_dm.py sdo $i ;  
	done
