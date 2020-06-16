# French Coreference Resolution System
French-CRS is a machine-learning based NLP framework for coreference resolution in French language. It is already trained using 25 syntactic/morphological features derived from ANCOR a French Oral Corpus. French-CRS has already pre-trained language models and it is ready to be incorporated for French text. It internaly uses other systems for mention and named entity detections. French-CRS is planned to be enriched by semantic features. This will let it be fitted for other tasks such as nomination detection in social media context.


## Prerequisite:  

1. Python 3.7  

2. Virtualenv (a useful resource for non-familiar ones is `https://python.doctor/page-virtualenv-python-environnement-virtuel`

3. CAREN (Coréférence par Application de RÉseaux de Neurones) for French mention detection (notice that the author permission is required) `https://github.com/LoicGrobol/decofre/`


## Quick Installation Instruction

For a quick start:  

1. Type at the root of your command prompt `git clone https://github.com/mehdi-mirzapour/French-CRS` or download the
   zip from github and unzip it  
2. Create a new environment `virtualenv -p /path/to/python3.7 env4fcrs` and replace `/path/to` with your local path to python using `which python` command; if you are sure that your current python version is 3.7 you can simply use `virtualenv env4fcrs`  
3. Source it `source env4fcrs/bin/activate` (remember you should do this every time you want to use French-CRS)   
4. Change the directory to cloned folder `cd French-CRS`  
5. Let the setup file install autumatically all the components `pip install -e .` (check out if the point sign "." is not accidently removed)  
6. In case you want to use Jupyter notebook, you can add your virtual environment to it by typing `python -m ipykernel install --user --name=env4fcrs`  
6. Install the following language model for spaCy  

    ```bash
    python -m spacy download fr_core_news_md
    ```  
7. Install an additional dependency called CAREN (Coréférence par Application de RÉseaux de Neurones) for French mention detection with the following setup (notice that the author permission is required):  

    ```bash
    git clone https://github.com/Evpok/neural-end-to-end-coref
    python3.7 -m venv .virtenv
    source .virtenv/bin/activate
    cd neural-end-to-end-coref
    git reset --hard 7ceb245fbf8cbf126c18ce29fb9e4746ca0036db
    python -m pip install -e .
    python -m spacy download fr_core_news_sm
    ``` 
8. To deactivate the enviroment run `deactivate` in the commonad prompt or simply close the terminal

## Running a pre-trained model pipelines in command line  

1. Ensure you are in the virtualenv `source env4fcrs/bin/activate`  
2. Modify the configuration file that exists in the root cloned folder.  
3. Run the crs-resolver `crs-resolver --text "..."`. You can get more information by `crs-resolver --help`.  

## Running a pre-trained model pipelines in jupyter notebook  

1. Ensure you are in the virtualenv `source env4fcrs/bin/activate`  
2. Ensure you have run before `python -m ipykernel install --user --name=env4fcrs` at the command prompt 
3. Type at command prompt `jupyter notebook`
4. Open "demo" folder and click on the file "Text2CoreferenceChains.ipynb"  


## Training a new model in jupyter notebook

1. Ensure you are in the virtualenv `source env4fcrs/bin/activate`  
2. Ensure you have run before `python -m ipykernel install --user --name=env4fcrs` at the command prompt 
3. Type at command prompt `jupyter notebook`
4. Open "demo" folder and click on the file "Model_ANCOR_Training.ipynb"  

## Downloading ANCOR and training with it
ANCOR can be downloaded here:  http://www.info.univ-tours.fr/~antoine/parole_publique/  

Notice: Downloading ANCOR corpus is not mandatory for running the CRS system. Pre-trained models are already introduced in the "pre-trained language models" folder. The jupyter notebook "/demo/Text2CoreferenceChains.ipynb" also describes how to actually integrate them.

## Citations
```bibtex
@inproceedings{desoyer2016coreference,
  title={Coreference Resolution for French Oral Data: Machine Learning Experiments with ANCOR},
  author={D{\'e}soyer, Ad{\`e}le and Landragin, Fr{\'e}d{\'e}ric and Tellier, Isabelle and Lefeuvre, Ana{\"\i}s and Antoine, Jean-Yves and Dinarelli, Marco},
  booktitle={International Conference on Intelligent Text Processing and Computational Linguistics},
  pages={507--519},
  year={2016},
  organization={Springer}
}
```

```bibtex
@inproceedings{muzerelle:hal-01075679,
  TITLE = {{ANCOR\_Centre, a Large Free Spoken French Coreference Corpus:  description of the Resource and Reliability Measures}},
  AUTHOR = {Muzerelle, Judith and Lefeuvre, Ana{\"i}s and Schang, Emmanuel and Antoine, Jean-Yves and Pelletier, Aurore and Maurel, Denis and Eshkol, Iris and Villaneau, Jeanne},
  BOOKTITLE = {{LREC'2014, 9th Language Resources and Evaluation Conference.}},
  PAGES = {843-847},
  YEAR = {2014}
}
```

### License
French-CRS is MIT-licensed, as found in the LICENSE file.
