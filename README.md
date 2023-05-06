# MIMIC-VQA
This is the code for generating MIMIC-VQA dataset

## How to use
1. Enter the 'code' directory
```console
cd code
```
2. Prepare for mimic_all.csv (MIMIC-CXR-JPG needs to be ready)
```angular2html
python get_mimic_all.py
```
3. Extract intermediate KeyInfo json dataset
```angular2html
python question_gen.py -j
```
4. Generate full version of question answer pairs
```angular2html
python question_gen.py -q
```
Or step 3 and step 4 can be executed simultaneously by run:
```angular2html
python question_gen.py -j -q
```
5. Filter to reduce the Yes/No answers and removed answers with less frequent occurrences
```angular2html
python filtering_low_freq.py
```