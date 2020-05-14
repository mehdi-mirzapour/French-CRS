import xml.etree.ElementTree as ET
import json
import pandas as pd
import re
import random
import glob
import networkx as nx

"""

This class reads a given ANCOR file located in a sub_corpus 
and returns its related mentions and relations in json format

"""

class read_ancor_file():
    
    # Looks for ./DISTRIB_ANCOR/ as deafult otherwise should be indicated.
    def __init__(self, path_ancor_file=None):
        if path_ancor_file==None:
            self.path_ancor_file= './DISTRIB_ANCOR/'
        else:
            self.path_ancor_file = path_ancor_file

    # sets initial configuration for three different file-formats presented in ANCOR corpus
    def set_file_config(self, sub_corpus, file):

        self.sub_corpus = sub_corpus
        self.file = file

        self.path_ancor_file += sub_corpus+'/'

        self.tree_xml = ET.parse(
            self.path_ancor_file+'annotation_integree/'+file+'.xml')
        self.tree_aa = ET.parse(
            self.path_ancor_file+'aa_fichiers/'+file+'.aa')
        self.data_source = open(
            self.path_ancor_file+'ac_fichiers/'+file + '.ac', "r").read()

        self.root_xml = self.tree_xml.getroot()
        self.root_aa = self.tree_aa.getroot()

    # return mentions in json format
    def generate_json_mentions(self):
        
        json_mentions = {}

        for unit in self.root_xml.iter("unit"):
            data_character = {}
            data_character["TYPE"] = unit.find("./characterisation/type").text

            for anchor in self.root_xml.iter("anchor"):
                if anchor.attrib["id"] == unit.attrib["id"]:
                    data_character["NUM"] = anchor.attrib["num"]

            for feat in unit.findall("./characterisation/featureSet/feature"):
                data_character[feat.attrib["name"]] = feat.text

            for unit_aa in self.root_aa.iter("unit"):
                if unit_aa.attrib["id"] == unit.attrib["id"]:
                    data_character["START_ID"] = int(unit_aa.find(
                        "./positioning/start/singlePosition").attrib["index"])
                    data_character["END_ID"] = int(unit_aa.find(
                        "./positioning/end/singlePosition").attrib["index"])

            json_mentions[unit.attrib["id"]] = data_character

        self.json_mentions = json_mentions

    # return mentions in json format and eliminates all the features that are not presented.
    def remove_empty_json_mentions(self, features_to_check=["CONTENT", "PREVIOUS"]):

        json_mentions_= self.json_mentions.copy()
        for mention in list(json_mentions_):
            for feature in features_to_check:
                if feature not in json_mentions_[mention]:
                    del json_mentions_[mention]
                    break

        self.json_mentions = json_mentions_


    # return relations in json format based on the json mentions
    def generate_json_relations(self):
        
        json_relations={}
        
        for relation in self.root_aa.iter("relation"):
            data_character = {}
            data_character["TYPE"] = relation.find("./characterisation/type").text

            for feat in relation.findall("./characterisation/featureSet/feature"):
                data_character[feat.attrib["name"]] = feat.text

            unit_ids = []
            for feat in relation.findall("./positioning/term"):
                unit_ids.append(feat.attrib["id"])

            if (unit_ids[0] in self.json_mentions.keys()) and (unit_ids[1] in self.json_mentions.keys()):
                if (len(unit_ids) == 2):
                    data_character["LEFT_UNIT"] = {
                        "ID": unit_ids[0], **self.json_mentions[unit_ids[0]]}
                    data_character["RIGHT_UNIT"] = {
                        "ID": unit_ids[1], **self.json_mentions[unit_ids[1]]}

                json_relations[relation.attrib["id"]] = data_character
        
        self.json_relations = json_relations

    # return chains in json format based on the json mentions and relations.
    def generate_json_chains(self, coreference_type=["DIRECTE", "INDIRECTE", "ANAPHORE"]):
        G = nx.Graph()
        chain_id = 0
        json_coreference_chains = {}

        for chain in self.json_relations:
            if self.json_relations[chain]["TYPE"].strip() in coreference_type:
                chain_left = self.json_relations[chain]["LEFT_UNIT"]["ID"]
                chain_right = self.json_relations[chain]["RIGHT_UNIT"]["ID"]
                G.add_edge(chain_left, chain_right)

        for subgraph in list(nx.connected_components(G)):
            json_coreference_chains[chain_id] = []
            for node in subgraph:
                json_coreference_chains[chain_id].append(node)
            chain_id += 1

        self.json_coreference_chains = json_coreference_chains

"""

This class inherits read_ancor_file class which itself reads a given ANCOR 
It introduces some methodes for having csv/excel format file which is built 
on the basis of different strategies of chain building.

"""

class chains2excel(read_ancor_file):

    # Looks for ./DISTRIB_ANCOR/ as deafult otherwise should be indicated.
    def __init__(self, path_ancor_file=None):
        super().__init__(path_ancor_file)


    # Generates positive and negative pairs by applying two strategies
    # "normal" strategy uses "neg_pairs_percentage" parameter for creating the pairs
    # "pairs_from_left_with_window_size" strategy uses "window_size" for creating the pairs

    def generate_pos_neg_pairs(self, neg_pairs_percentage=0.5, strategy="normal",window_size=10):
        
        positive_pairs = []
        negative_pairs = []


        if (strategy=="normal"):

            for chains in list(self.json_coreference_chains.keys()):
                for chain_id in range(len(self.json_coreference_chains[chains])-1):
                    positive_pairs.append(
                        [self.json_coreference_chains[chains][chain_id], self.json_coreference_chains[chains][chain_id+1]])

            num_negative_relations = int(round(
                (len(positive_pairs)*neg_pairs_percentage)/(1-neg_pairs_percentage), 0))+1
            
            count = 0
            no_result_count=0

            while (count <= num_negative_relations and no_result_count<=1000):

                mention_left = random.choice(list(self.json_mentions.keys()))
                mention_right = random.choice(list(self.json_mentions.keys()))

                if mention_right != mention_left:
                    if([mention_left, mention_right] not in negative_pairs) and ([mention_right, mention_left] not in negative_pairs):
                        if ([mention_left, mention_right] not in positive_pairs) and ([mention_right, mention_left] not in positive_pairs):
                            negative_pairs.append(
                                [mention_left, mention_right])
                            count += 1
                        else:
                            no_result_count += 1

            pos_pairs = [[pair[0], pair[1], 1] for pair in positive_pairs]
            neg_pairs = [[pair[0], pair[1], 0] for pair in negative_pairs]
            self.json_pos_neg_pairs = pos_pairs+neg_pairs


        elif (strategy == "pairs_from_left_with_window_size"):

            list_mentions_sorted_seperated=[]
            df_mentions = pd.DataFrame.from_dict(self.json_mentions, orient='index')
            df_mentions = df_mentions.sort_values(by=['START_ID'])
            list_mentions_sorted = list(df_mentions.index)
            
            for index in range(len(list_mentions_sorted)):

                    start_index = index-window_size

                    if (start_index) < 0:
                        start_index = 0

                    for index_left in range(index, start_index, -1):

                        left_m = list_mentions_sorted[index]
                        right_m = list_mentions_sorted[index_left-1]

                        # detecting wether the pairs belongs to chain or not
                        pos_flag = False

                        for index_chain in range(len(self.json_coreference_chains.keys())):
                            if (left_m in self.json_coreference_chains[index_chain]) and (right_m in self.json_coreference_chains[index_chain]):
                                pos_flag = True
                                break

                        if (pos_flag):
                            list_mentions_sorted_seperated.append([left_m, right_m, 1])
                        else:
                            list_mentions_sorted_seperated.append([left_m, right_m, 0])

            self.json_pos_neg_pairs = list_mentions_sorted_seperated
        
    # Extends positive and negative with chain id
    def extend_with_order_pos_neg_pairs(self):

        def sort_list_of_list_3(sub_li):
            l = len(sub_li)
            for i in range(0, l):
                for j in range(0, l-i-1):
                    if (sub_li[j][3] > sub_li[j + 1][3]):
                        tempo = sub_li[j]
                        sub_li[j] = sub_li[j + 1]
                        sub_li[j + 1] = tempo
            return sub_li

        pairs_extended = []
        for pair in self.json_pos_neg_pairs:
            if (pair[2] == 1):
                for chain_key in list(self.json_coreference_chains.keys()):
                    chain_list = self.json_coreference_chains[chain_key]
                    if (pair[0] in chain_list) and (pair[1] in chain_list):
                        pairs_extended.append(
                            [pair[0], pair[1], pair[2], chain_key])
            else:
                pairs_extended.append([pair[0], pair[1], pair[2], -1])

        self.json_pos_neg_pairs_with_orders=sort_list_of_list_3(pairs_extended)


    # Generate training/test dataset
    def generate_training_test_dataset(self, output_path=".",save_files=True):

        pos_rel_num = len(
            [pair for pair in self.json_pos_neg_pairs_with_orders if pair[2] == 1])
        neg_rel_num = len(
            [pair for pair in self.json_pos_neg_pairs_with_orders if pair[2] == 0])

        coreference_pairs_df = pd.DataFrame(columns=['Left_ID', 'Right_ID', 'Sub_Corpus', 'File_Name', 'chain_id', 'm1',
                                                    'm2', 'm1_TYPE', 'm2_TYPE', 'm1_DEF', 'm2_DEF', 'm1_GP', 'm2_GP',
                                                    'm1_GENDER', 'm2_GENDER', 'm1_NUMBER', 'm2_NUMBER', 'm1_EN', 'm2_EN',
                                                    'ID_FORM', 'ID_SUBFORM', 'INCL_RATE', 'COM_RATE', 'ID_DEF', 'ID_GP', 
                                                    'ID_TYPE','ID_EN', 'ID_GENDER', 'ID_NUMBER', 'DISTANCE_MENTION', 
                                                    'DISTANCE_WORD','DISTANCE_CHAR', 'EMBEDDED', 'ID_PREVIOUS', 'ID_NEXT', 
                                                    'IS_CO_REF'])

        tf2yn = {True: "YES", False: "NO"}

        for pair in self.json_pos_neg_pairs_with_orders:
            left_ID = pair[0]
            right_ID = pair[1]
            status = pair[2]
            chain_id = pair[3]

            left_unit = self.json_mentions[left_ID]
            right_unit = self.json_mentions[right_ID]

            left_index = left_unit["START_ID"]
            right_index = right_unit["START_ID"]

            if (left_index < right_index):
                pass
            else:
                temp_json = right_unit
                right_unit = left_unit
                left_unit = temp_json

            if "CONTENT" not in left_unit:
                m1 = ""
            elif left_unit["CONTENT"] is None:
                m1 = ""
            else:
                m1 = left_unit["CONTENT"]

            if "CONTENT" not in right_unit:
                m2 = ""
            elif right_unit["CONTENT"] is None:
                m2 = ""
            else:
                m2 = right_unit["CONTENT"]

            m1_TYPE = left_unit["TYPE"]
            m2_TYPE = right_unit["TYPE"]
            m1_DEF = left_unit["DEF"]
            m2_DEF = right_unit["DEF"]
            m1_GP = left_unit["GP"]
            m2_GP = right_unit["GP"]
            m1_GENDER = left_unit["GENRE"]
            m2_GENDER = right_unit["GENRE"]
            m1_NUMBER = left_unit["NB"]
            m2_NUMBER = right_unit["NB"]
            m1_EN = left_unit["EN"]
            m2_EN = right_unit["EN"]
            m1_PREVIOUS = left_unit["PREVIOUS"]
            m2_PREVIOUS = right_unit["PREVIOUS"]
            m1_NEXT = left_unit["NEXT"]
            m2_NEXT = right_unit["NEXT"]

            # Needed steps for calculating the words and chars distances
            string_text = self.data_source[left_unit["END_ID"]:right_unit["START_ID"]]
            text_between_units = re.sub(
                "<[a-zA-Z0-9\s=\",/.]+>", " ", string_text).strip().split()

            # Needed steps for calculating the words and chars distances
            left_content = str(left_unit["CONTENT"])
            right_content = str(right_unit["CONTENT"])

            left_content_list = left_content.lower().split()
            right_content_list = right_content.lower().split()

            len_of_words_min = min(len(left_content_list), len(right_content_list))
            len_of_words_max = max(len(left_content_list), len(right_content_list))
            len_intersection = len(
                [value for value in left_content_list if value in right_content_list])

            # Calculating the char distance
            dist_chars_acc = 0
            for char in text_between_units:
                dist_chars_acc += len(char)
            dist_chars_acc += len(text_between_units)

            # Calculating the True/False properties
            bool_is_FORM = m1 == m2
            bool_is_SUBFORM = len_intersection > 0

            bool_is_DEF = m1_DEF == m2_DEF
            bool_is_GP = m1_GP == m2_GP
            bool_is_TYPE = m1_TYPE == m2_TYPE
            bool_is_EN = m1_EN == m2_EN
            bool_is_GENDER = m1_GENDER == m2_GENDER
            bool_is_NUMBER = m1_NUMBER == m2_NUMBER

            bool_is_PREVIOUS = m1_PREVIOUS == m2_PREVIOUS
            bool_is_NEXT = m1_NEXT == m2_NEXT

            ID_FORM = tf2yn[bool_is_FORM]
            ID_SUBFORM = tf2yn[bool_is_SUBFORM]

            ID_DEF = tf2yn[bool_is_DEF]
            if m1_DEF == "UNK" or m2_DEF == "UNK":
                ID_DEF = "NA"

            ID_GP = tf2yn[bool_is_GP]
            if m1_GP == "UNK" or m2_GP == "UNK":
                ID_GP = "NA"

            ID_TYPE = tf2yn[bool_is_TYPE]
            if m1_TYPE == "UNK" or m2_TYPE == "UNK":
                ID_TYPE = "NA"

            ID_EN = tf2yn[bool_is_EN]
            if m1_EN == "NO" or m2_EN == "NO":
                ID_EN = "NA"

            ID_GENDER = tf2yn[bool_is_GENDER]
            if m1_GENDER == "UNK" or m2_GENDER == "UNK":
                ID_GENDER = "NA"

            ID_NUMBER = tf2yn[bool_is_NUMBER]
            if m1_NUMBER == "UNK" or m2_NUMBER == "UNK":
                ID_NUMBER = "NA"

            EMBEDDED = tf2yn[((m1 in m2) or (m2 in m1))]

            ID_PREVIOUS = tf2yn[bool_is_PREVIOUS]
            if m1_PREVIOUS == "^" or m2_PREVIOUS == "^":
                ID_PREVIOUS = "NA"

            ID_NEXT = tf2yn[bool_is_NEXT]
            if m1_NEXT == "^" or m2_NEXT == "^":
                ID_NEXT = "NA"

            INCL_RATE = len_intersection/len_of_words_min
            COM_RATE = len_intersection/len_of_words_max

            DISTANCE_MENTION = abs(int(left_unit["NUM"])-int(right_unit["NUM"]))-1
            DISTANCE_WORD = len(text_between_units)
            DISTANCE_CHAR = dist_chars_acc

            row_to_be_added = pd.DataFrame([[left_ID, right_ID, self.sub_corpus, self.file, chain_id, m1, m2, 
                                            m1_TYPE, m2_TYPE, m1_DEF, m2_DEF, m1_GP, m2_GP,
                                            m1_GENDER, m2_GENDER, m1_NUMBER, m2_NUMBER, m1_EN, m2_EN,
                                             ID_FORM, ID_SUBFORM, INCL_RATE, COM_RATE, ID_DEF, ID_GP, ID_TYPE,
                                            ID_EN, ID_GENDER, ID_NUMBER, DISTANCE_MENTION, DISTANCE_WORD,
                                            DISTANCE_CHAR, EMBEDDED, ID_PREVIOUS, ID_NEXT, status]],
                                        columns=coreference_pairs_df.columns)
            coreference_pairs_df = coreference_pairs_df.append(
                row_to_be_added, ignore_index=True, sort=False)

        if save_files:
            coreference_pairs_df.to_excel(
                output_path+"/"+self.file+"_POS_"+str(pos_rel_num)+"_NEG_"+str(neg_rel_num)+".xlsx")
        
        self.coreference_pairs_dataframe=coreference_pairs_df
