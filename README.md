# French Coreference Resolution System
French-CRS is a machine-learning based NLP framework for coreference resolution in French language. It is already trained using 25 syntactic/morphological features derived from ANCOR a French Oral Corpus. French-CRS has already pre-trained language models and it is ready to be incorporated for French text. It internaly uses other systems for mention and named entity detections. French-CRS is planned to be enriched by semantic features. This will let it be fitted for other tasks such as nomination detection in social media context.

## Boostraping French-CRS

For an easy boostrap

1. Type at command prompt `https://github.com/mehdi-mirzapour/French-CRS` or download the
   zip from github and unzip it  
2. Create a new enviroment `virtualenv env4fcrs`  
3. Source it `source French-CRS/bin/activate` (remember you should do this every time you want to use French-CRS)  
4. Change the directory to cloned folder `cd French-CRS`  
5. Let the setup file install all the components autumatically `pip install -e .` (Don't forget to put the "." point sign)  
6. Install the additional dependencies

    ```bash
    python -m spacy download fr_core_news_md
    ```

## Running a pre-trained model pipelines in command line

1. Ensure you are in the virtualenv `source French-CRS/bin/activate`  
2. Modify the configuration file that exists in the root cloned folder.  
3. Run the crs-resolver `crs-resolver --text "..."`. You can get more information by `crs-resolver --help`.  


## Installing French-CRS
French-CRS requires Python 3.7 or above and it calls internally following framework:  

1- ÚFAL UDPipe.  
2- spaCy for French Named Entitiy Detection.  
3- CAREN (Coréférence par Application de RÉseaux de Neurones) for French mention detection.  
4- JupyterLab Notebook (optional)  

Quick installation guide:
```
  $ git clone https://github.com/mehdi-mirzapour/French-CRS
  $ cd French-CRS
  $ pip install -r requirements.txt
  $ python -m spacy download fr_core_news_md
  $ Jupyter notebook
```

## Running a pre-trained model pipelines in jupyter notebook
The jupyter notebook "/demo/Text2CoreferenceChains.ipynb" describes all the essential steps. This file can be opened via Jupyter notebook and it demonstrates a walkthough with a runnnng example.  


## Training a new model in jupyter notebook
The jupyter notebook "/demo/Model_ANCOR_Training.ipynb" describes all the essential steps. This file can be opened via Jupyter notebook and it demonstrates different possible strategies for training new models. 

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
