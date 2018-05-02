#!/bin/bash

for i in {10..20}; 
	do 	python evaluate_mnist.py sdo $i ;  
		python evaluate_mnist.py do $i ;  
		python evaluate_mnist.py no_do $i ;  
		python evaluate_mnist.py ann_do $i ;  
	done
