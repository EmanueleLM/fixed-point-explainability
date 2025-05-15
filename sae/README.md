### Instructions

Please make sure to have installed all required libraries.

There is 1 main file to run the recursive SAE (```recursive_sae.py```), and one that takes its output and outputs the distribution of recursive SAE iterations before convergence (```quantify_convergence.py```).

This is an example of how to run the experiment on our numerical questions dataset:
```
python recursive_sae.py --prompts-file q_a/number_questions.txt --answers-file q_a/number_answers.txt --output-dir 8b_number_result 
```

You can navigate to the ```q_a``` folder and see the filenames for the other 2 categories we test our method on (multiple choice and general knowledge), and change the ```--prompts-file``` and ```--answers-file``` and parameters accordingly. The structure of these files is simple: for each line in questions, the corresponding line in answers is the correct token. Keep in mind that the default model we use is Llama 3.1 8b, which can be sensitive to question format.

Once this file is done running (keep in mind you need a lot of VRAM) you will find a log file for each prompt in the ```--output-dir```, and you will see a summary.txt file with the average percentages of correct answers and the average Jaccard similarity between starting and fixed point features.

The outputs are saved to the ```--output-dir```, and then we can use the second file as follows:

```
python quantify_convergence.py
```

Keep in mind that the names of the output dir are hard coded in the ```categories``` variable. 
This file will scan the individual logs to record the distribution of how many iterations it took to reach a fixed point explaination to ```iteration_distributions.csv``` and it will print summary statistics to the command line.
