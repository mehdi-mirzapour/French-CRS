# French Coreference Resolution System
French-CRS is a machine-learning based NLP framework for coreference resolution in French language. It is already trained using 25 syntactic/morphological features derived from ANCOR a French Oral Corpus. French-CRS has already pre-trained language models and it is ready to be incorporated for French text. It internaly uses other systems for mention and named entity detections. French-CRS is planned to be enriched by semantic features. This will let it be fitted for other tasks such as nomination detection in social media context.






## Installing French-CRS
French-CRS requires Python 3.6.1 or above and it calls internally following framework:
spaCy for French Named Entitiy Detection
CAREN (Coréférence par Application de RÉseaux de Neurones) for French mention detection.

Get the source code:
```
  $ git clone https://github.com/mehdi-mirzapour/French-CRS
  $ cd test
```

