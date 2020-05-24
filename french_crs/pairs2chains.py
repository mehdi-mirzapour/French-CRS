import json
import pandas as pd
import networkx as nx
import ast
from pathlib import Path

"""

This class reads a given list of pairs and generates two coreference chains
the first one is the gold one, and the second one is the related predicted one
in json format.

"""

class chains_builder():
    

    # initiator for building the chains
    def __init__(self, path_gold_file, path_model_file, gold_column, model_column,threshold):
        self.gold_dataframe = pd.read_excel(str(Path(path_gold_file)), index_col=0)
        self.model_dataframe = pd.read_excel(str(Path(path_model_file)), index_col=0)
        self.gold_column = gold_column
        self.model_column = model_column
        self.threshold = threshold


    def generate_coreference_chains(self, dataframe, column,mode):

        coreference_chain_built = {}
        chain_id = 0
        G = nx.Graph()


        dataframe_ = dataframe[dataframe[column] >= self.threshold]

        for id_chain in range(dataframe_.shape[0]):
            chain_left = dataframe_.iloc[id_chain]["Left_ID"]
            chain_right = dataframe_.iloc[id_chain]["Right_ID"]
            G.add_edge(chain_left, chain_right)


        for subgraph in list(nx.connected_components(G)):
            coreference_chain_built[chain_id] = []
            for node in subgraph:
                if mode=="test":
                    coreference_chain_built[chain_id].append(ast.literal_eval(node))
                elif mode=="train":
                    coreference_chain_built[chain_id].append(node)

            chain_id += 1

        return(coreference_chain_built)

    # Calling this method produces two json files in SCORCH format
    def generate_gold_model_json_output(self, mode="test", path_json_files="./"):

        coref_chains_gold = self.generate_coreference_chains(
            self.gold_dataframe, self.gold_column, mode)
        coref_chains_pred = self.generate_coreference_chains(
            self.model_dataframe, self.model_column, mode)

        coref_chains_gold_json = {"type": "clusters", "clusters": coref_chains_gold}
        coref_chains_pred_json = {"type": "clusters", "clusters": coref_chains_pred}

        with open(str(Path(path_json_files+'coref_chains_gold.json')), 'w') as outfile:
            json.dump(coref_chains_gold_json, outfile)

        with open(str(Path(path_json_files+'coref_chains_pred.json')), 'w') as outfile:
            json.dump(coref_chains_pred_json, outfile)

        self.coref_chains_gold_json = coref_chains_gold
        self.coref_chains_pred_json = coref_chains_pred


    def print2string(self,context_list,sep=" "):
        items=[item[1] for item in context_list]
        items=sep.join(items)
        return(items)


    def return_left_right_context_theme(self, mention, sentences_json,
                                  context_left_window,
                                  context_right_window,
                                  theme_left_window,
                                  theme_right_window, 
                                  path_to_stop_words="./french_crs/StopWords-FR.xlsx"
                                  ):

        df=pd.read_excel(str(Path(path_to_stop_words)))
        stopwords_list = list(df["StopWords"])

        # Return whole context list : "whole_context"
        whole_context = []
        count = 0
        for sent_id in range(len(sentences_json)):
            for token_id in range(len(sentences_json[sent_id]["Content"])):
                whole_context.append([count, sentences_json[sent_id]
                                    ["Content"][token_id]["form"], sent_id+1, token_id+1])
                count += 1

        # Return whole theme (=context without stopwords list) : "whole_context_without_stopwords"
        whole_theme = []
        count = 0
        for sent_id in range(len(sentences_json)):
            for token_id in range(len(sentences_json[sent_id]["Content"])):
                if sentences_json[sent_id]["Content"][token_id]["form"].lower() not in stopwords_list:
                    whole_theme.append([count, sentences_json[sent_id]
                                        ["Content"][token_id]["form"], sent_id+1, token_id+1])
                    count += 1


        current_sent = mention[1]
        current_token_first = mention[2][0]
        current_token_last = mention[2][-1]


        # Seperate context list into before and after list
        len_whole_context = len(whole_context)
        id_left_context = -1
        id_right_context = len_whole_context
        for context_id in range(len(whole_context)):
            if (whole_context[context_id][2], whole_context[context_id][3]) < (current_sent, current_token_first):
                id_left_context = whole_context[context_id][0]

        for context_id in range(len(whole_context)):
            if (whole_context[context_id][2], whole_context[context_id][3]) > (current_sent, current_token_last):
                id_right_context = whole_context[context_id][0]
                break


        # Create left context
        local_left_context = []
        for ctx_id in range(id_left_context-context_left_window+1, id_left_context+1):
            if ctx_id < 0:
                local_left_context.append([-1, '$', -1, -1])
            elif ctx_id >= len_whole_context:
                local_left_context.append([-1, '$', -1, -1])
            else:
                local_left_context.append(whole_context[ctx_id])


        # Create right context
        local_right_context = []
        for ctx_id in range(id_right_context, id_right_context+context_right_window):
            if ctx_id < 0:
                local_right_context.append([-1, '$', -1, -1])
            elif ctx_id >= len_whole_context:
                local_right_context.append([-1, '$', -1, -1])
            else:
                local_right_context.append(whole_context[ctx_id])


        # Seperate theme list into before and after list
        len_whole_theme = len(whole_theme)
        id_left_theme = -1
        id_right_theme = len_whole_theme
        for theme_id in range(len(whole_theme)):
            if (whole_theme[theme_id][2], whole_theme[theme_id][3]) < (current_sent, current_token_first):
                id_left_theme = whole_context[theme_id][0]

        for theme_id in range(len(whole_theme)):
            if (whole_theme[theme_id][2], whole_theme[theme_id][3]) > (current_sent, current_token_last):
                id_right_theme = whole_context[theme_id][0]
                break


        # Create left theme
        local_left_theme = []
        for ctx_id in range(id_left_theme-theme_left_window+1, id_left_theme+1):
            if ctx_id < 0:
                local_left_theme.append([-1, '$', -1, -1])
            elif ctx_id >= len_whole_theme:
                local_left_theme.append([-1, '$', -1, -1])
            else:
                local_left_theme.append(whole_theme[ctx_id])


        # Create right theme
        local_right_theme = []
        for ctx_id in range(id_right_theme, id_right_theme+theme_right_window):
            if ctx_id < 0:
                local_right_theme.append([-1, '$', -1, -1])
            elif ctx_id >= len_whole_theme:
                local_right_theme.append([-1, '$', -1, -1])
            else:
                local_right_theme.append(whole_theme[ctx_id])

        return(local_left_context, local_right_context, local_left_theme, local_right_theme)



    def context_visualizer(self, sentences_json, 
                                        context_left_window=3,
                                        context_right_window=3,
                                        theme_left_window=10,
                                        theme_right_window=10,
                                        file_path="context_visualized.xlsx",
                                        path_to_stop_words="./french_crs/StopWords-FR.xlsx"):

        df = pd.DataFrame(columns=[
                          'ID', 'Thème gauche', 'Contexte_L', 'Contexte_C', 'Contexte_R', 'Thème droit'])

        for chain_id in range(len(self.coref_chains_pred_json)):

            for mention in self.coref_chains_pred_json[chain_id]:

                local_left_context, local_right_context, \
                local_left_theme, local_right_theme = self.return_left_right_context_theme(
                                                                                    mention, sentences_json, 
                                                                                    context_left_window,
                                                                                    context_right_window,
                                                                                    theme_left_window,
                                                                                    theme_right_window,
                                                                                    path_to_stop_words)

                Contexte_L = self.print2string(local_left_context)
                Contexte_C = mention[0]
                Contexte_R = self.print2string(local_right_context)

                Theme_gauche = self.print2string(local_left_theme, ",")
                Theme_droit = self.print2string(local_right_theme, ",")

                df_row = pd.DataFrame([{'ID': chain_id, 'Thème gauche': Theme_gauche, 'Contexte_L': Contexte_L,
                                        'Contexte_C': Contexte_C, 'Contexte_R': Contexte_R, 'Thème droit': Theme_droit}])
                df = df.append(df_row, ignore_index=True)
            df_row = pd.DataFrame([{'ID': "", 'Thème gauche': "", 'Contexte_L': "",
                                    'Contexte_C': "", 'Contexte_R': "", 'Thème droit': ""}])
            df = df.append(df_row, ignore_index=True)
            df = df.append(df_row, ignore_index=True)

        df.to_excel(str(Path(file_path)), index=False)
        self.visualized_context_df = df

        return(self.visualized_context_df)
