import json
import pandas as pd
import networkx as nx
import ast

"""

This class reads a given list of pairs and generates two coreference chains
the first one is the gold one, and the second one is the related predicted one
in json format.

"""

class chains_builder():
    

    # initiator for building the chains
    def __init__(self, path_gold_file, path_model_file, gold_column, model_column,threshold):
        self.gold_dataframe = pd.read_excel(path_gold_file, index_col=0)
        self.model_dataframe = pd.read_excel(path_model_file, index_col=0)
        self.gold_column = gold_column
        self.model_column = model_column
        self.threshold = threshold


    def generate_coreference_chains(self, dataframe, column, threshold,mode):

        coreference_chain_built = {}
        chain_id = 0
        G = nx.Graph()


        dataframe_ = dataframe[dataframe[column] >= threshold]

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
    def generate_gold_model_json_output(self, mode="test"):

        coref_chains_gold = self.generate_coreference_chains(
            self.gold_dataframe, self.gold_column, self.threshold,mode)
        coref_chains_pred = self.generate_coreference_chains(
            self.model_dataframe, self.model_column, self.threshold,mode)

        coref_chains_gold_json = {"type": "clusters", "clusters": coref_chains_gold}
        coref_chains_pred_json = {"type": "clusters", "clusters": coref_chains_pred}


        with open('coref_chains_gold.json', 'w') as outfile:
            json.dump(coref_chains_gold_json, outfile)

        with open('coref_chains_pred.json', 'w') as outfile:
            json.dump(coref_chains_pred_json, outfile)

        self.coref_chains_gold_json = coref_chains_gold
        self.coref_chains_pred_json = coref_chains_pred
