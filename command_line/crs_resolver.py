import click
import json

from french_crs.text2corefchains import udpipe_spacy_lang_model, mentions2chains
from french_crs.pairs2chains import chains_builder


@click.command()
@click.option('--text',
              help='The input text with potential corefernce chains.')

@click.option('--showchains', 
              default='yes',
              help='Whether to show coreference chains results on screen or not. The default value is "yes"')

@click.option('--visualizer',
              default='enable',
              help='Whether to show coreference chains visualizer results on screen or not. The default value is "enable"')

@click.option('--configfile',
                default='coref-config.json',
              help='The path and the name of the json confing file. The default value is "coref-config.json"')

@click.option('--outputdir', 
                default='.',
              help='The directory that pipeline results will be saved. The default value is "."')

def run_pipelines(text, showchains, visualizer, configfile, outputdir):
    
    if (text==None):

        message="""
Usage: crs-resolver.py [OPTIONS]

    Options:
    --text TEXT        The input text with potential corefernce chains.
    --showchains TEXT  Whether to show coreference chains results on screen or
                        not. The default value is "yes"

    --visualizer TEXT  Whether to show coreference chains visualizer results on
                        screen or not. The default value is "enable"

    --configfile TEXT  The path and the name of the json confing file. The
                        default value is "coref-config.json"

    --outputdir TEXT   The directory that pipeline results will be saved. The
                        default value is "."

    --help             Show this message and exit.

        """
        print(message)
        exit()

    try:
        open("./command_line_config.json", "r") as f:
        config_file = json.load(f)

    except FileNotFoundError:
        print("Warning: the file 'command_line_config.json' could not be found. A default internal config is assigned.")
        config_file = {"udpipe_model": "./external language models/french-ud-2.0-170801.udpipe",
                        "spacy_model": "fr_core_news_md",
                        "pretrained_model": "./pre-trained language models/Model_ANCOR_Representative_INDIRECTE.model",
                        "text": "à mettre en œuvre de le protectionnisme intelligent à mettre en avant de le patriotisme économique pour donner un avantage à les entreprises françaises dans la commande publique voilà tout cela . Le patriotisme économique qui n ’ a jamais été mis en œuvre le protectionnisme intelligent la défiscalisation de les heures supplémentaires la suppression de le travail détaché la baisse de les charges mais exclusivement pour les TPE PME . Il met en place un patriotisme économique un protectionnisme intelligent il dit à les constructeurs américains si vous voulez aller faire vos voitures à l ’ étranger construire une voiture à l’ étranger alors vous paierez une taxe en les réimportant à les Etats-Unis . D ’ autant que évidemment ce que fait Trump m ’ intéresse et pour cause puisqu ’ il met en place la politique que j’ appelle de mes vœux depuis très longtemps et notamment la politique de patriotisme économique de protectionnisme intelligent",
                        "path_mention_detector": ".",
                        "text_features_xls_path": "./command_line/crs-output/unseen_data.xlsx",
                        "scorch_json_output_path": "./command_line/crs-output/",
                        "stop_words_file": "./french_crs/StopWords-FR.xlsx",
                        "context_visualized_file": "./command_line/crs-output/context_visualized.xlsx"
        }


    # Language Model Loading

    print("Language Model Loading...")
    model = udpipe_spacy_lang_model(config_file["udpipe_model"],
                                    config_file["spacy_model"])

    # Text Parsing
    print("Text Parsing...")
    model.tokenize(config_file["text"])
    model.parse_sentences_json()
    model.update_json_sents_with_spacy_entities(operation="in")
    model.update_json_sents_with_mentions(
        operation="==", relative_path_mention_detector=config_file["path_mention_detector"])


    # Pairs & Features Building
    print("Pairs & Features Building...")
    chains_generator = mentions2chains(model.mentions_ids_in_json_sents)
    chains_generator.generate_mention_pairs(window_size=30, iteration=3)
    chains_generator.generate_json_mention_pairs(
        model.mentions_ids_in_json_sents, model.sentences_json)
    chains_generator.json_mention_pairs2dataframe(
        save_file=True, file_path=config_file["text_features_xls_path"])

    # Coreference Prediction

    print("Coreference Prediction...")
    chains_generator.json_mention_pairs2dataframe(
        save_file=True, file_path=config_file["text_features_xls_path"])
    chains_generator.init_data_model(model_name=config_file["pretrained_model"],
                                    input_method=".xlsx",
                                    file_path=config_file["text_features_xls_path"],
                                    column_outcome="Prediction",
                                    threshold=0.5)

    chains_generator.apply_model_to_dataset()


    # Coreference Chains Building

    print("Coreference Chains Building...")
    chains = chains_builder(config_file["text_features_xls_path"],
                            config_file["text_features_xls_path"],
                            "Prediction",
                            "Prediction",
                            0.5)

    print("Coreference Chains Saving...")
    chains.generate_gold_model_json_output(
        mode="test", path_json_files=config_file["scorch_json_output_path"])
    chains.coref_chains_pred_json


    # Coreference Chains Visualizing
    print("Coreference Chains Visualizing...")
    chains.context_visualizer(model.sentences_json,
                            context_left_window=3,
                            context_right_window=3,
                            theme_left_window=5,
                            theme_right_window=4,
                            file_path=config_file["context_visualized_file"],
                            path_to_stop_words=config_file["stop_words_file"])

if __name__ == '__main__':
    run_pipelines()
