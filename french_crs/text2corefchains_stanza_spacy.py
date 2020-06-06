from ufal.udpipe import Model, InputFormat, OutputFormat, ProcessingError, Sentence
import os
import spacy
import stanza
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
import joblib
from pathlib import Path
from spacy.matcher import Matcher
from spacy_stanza import StanzaLanguage



class stanza_spacy_lang_model:

    def __init__(self, spacy_stanza_lang_model="gsd", framework="spacy"):
        
        if framework=="spacy":
            self.nlp = spacy.load(spacy_stanza_lang_model)
        else:
            stanza.download('fr', package=spacy_stanza_lang_model)
            snlp = stanza.Pipeline('fr', package=spacy_stanza_lang_model)
            self.nlp = StanzaLanguage(snlp)

    def run_spacy_pipe_lines(self, text): 
        self.doc=self.nlp(text)

    def find_mentions_in_doc(self):

        matcher = Matcher(self.nlp.vocab)
        
        pattern_mention_1 = [
            {"POS": "DET"}, 
            {"POS": "NOUN"},
            {"POS": "ADJ"}
        ]

        pattern_mention_2 = [
            {"POS": "DET"},
            {"POS": "NOUN"},
            {"POS": "ADJ"},
            {"POS": "ADP"},
            {"POS": "DET"},
            {"POS": "NOUN"},
            {"POS": "ADJ"}
        ]

        pattern_mention_3 = [
            {"POS": "DET","OP": "?"},
            {"POS": "PRON"}
        ]


        pattern_mention_4 = [
            {"POS": "PROPN"}
        ]


        pattern_mention_5 = [
            {"POS": "DET"},
            {"POS": "NOUN"},
            {"POS": "NOUN"}
        ]

        matcher.add("pattern_mention_1", None, pattern_mention_1)
        matcher.add("pattern_mention_2", None, pattern_mention_2)
        matcher.add("pattern_mention_3", None, pattern_mention_3)
        matcher.add("pattern_mention_4", None, pattern_mention_4)
        matcher.add("pattern_mention_5", None, pattern_mention_5)

        matches = matcher(self.doc)

        mentions_list=[]
        
        for _, start, end in matches:
            matched_span = self.doc[start:end]
            if matched_span.text != "’":
                mentions_list.append(matched_span)
            
        self.doc_mentions_list=mentions_list
        
        return(self.doc_mentions_list)
            

class mentions2chains:

    def __init__(self, doc, doc_mentions_list):
        self.doc = doc
        self.doc_mentions_list = doc_mentions_list

    def remove_duplicates(self, oldlist):
        cleanlist = []

        for x in oldlist:
            if x not in cleanlist:
                cleanlist.append(x)
        return(cleanlist)


    def mentions_sort(self):
        n = len(self.doc_mentions_list)

        for i in range(n):
            for j in range(0, n-i-1):
                if (self.doc_mentions_list[j+1].start < self.doc_mentions_list[j].start):
                    self.doc_mentions_list[j], self.doc_mentions_list[j +1] = self.doc_mentions_list[j+1], self.doc_mentions_list[j]


    def remove_included_mentions(self):

        if (len(self.doc_mentions_list)) <= 1:
            return(self.doc_mentions_list)

        removed_list_mentions = []

        for idx in range(len(self.doc_mentions_list)-1):

            first_ment = self.doc_mentions_list[idx]
            second_ment = self.doc_mentions_list[idx+1]

            x = set(range(first_ment.start, first_ment.end))
            y = set(range(second_ment.start, second_ment.end))

            if y.issubset(x):
                removed_list_mentions.append(second_ment)

            if x.issubset(y):
                removed_list_mentions.append(first_ment)

        filtered_mentions_list = [
            ment for ment in self.doc_mentions_list if ment not in removed_list_mentions]

        self.doc_mentions_list = filtered_mentions_list

        return(self.doc_mentions_list)

    def generate_mention_pairs(self, window_size= 20):

        self.doc_mentions_list = self.remove_duplicates(self.doc_mentions_list)
        self.mentions_sort()
        self.remove_included_mentions()

        list_mention_pairs = []
        n = len(self.doc_mentions_list)

        if window_size >= n:
            window_size = n-1

        # Traverse through all array elements
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, i+1):
                if i != j and (i-j) < window_size+1:
                    list_mention_pairs.append(
                        (self.doc_mentions_list[j], self.doc_mentions_list[i]))

        self.doc_mention_pairs_list = list_mention_pairs

        return(self.doc_mention_pairs_list)

    def find_gender(self, mention):

        found_gender = "UNK"

        for idx in range(0, mention.end-mention.start):
            if mention[idx].pos_ in ["NOUN", "PROPN", "DET"]:
                if "Masc" in mention[idx].tag_:
                    found_gender = "M"
                    break
                if "Fem" in mention[idx].tag_:
                    found_gender = "F"
                    break
        return(found_gender)


    def find_number(self, mention):

        found_number = "UNK"
        for idx in range(0, mention.end-mention.start):
            if mention[idx].pos_ in ["NOUN", "PROPN", "DET"]:
                if "Sing" in mention[idx].tag_:
                    found_number = "SG"
                    break
                if "Plur" in mention[idx].tag_:
                    found_number = "PL"
                    break
        return(found_number)


    def find_prep(self, mention):

        found_prep = "NO"
        for idx in range(0, mention.end-mention.start):
            if mention[idx].dep_ in ["case"]:
                    found_prep = "YES"
                    break

        return(found_prep)

    def find_pos(self, mention):

        found_pos = "UNK"
        for idx in range(0, mention.end-mention.start):
            if mention[idx].pos_ in ["NOUN", "PROPN"]:
                    found_pos = "N"
                    break
            if mention[idx].pos_ in ["PRON"]:
                    found_pos = "PR"
                    break

        return(found_pos)

    def find_entity(self, mention):

        found_ent = "UNK"
        for idx in range(0,mention.end-mention.start):

            if mention[idx].ent_type_ is not "":
                if mention[idx].ent_type_ == "PERSON":
                    found_ent = "PERS"
                elif mention[idx].ent_type_ == "LOC":
                    found_ent = "LOC"
                elif mention[idx].ent_type_ == "ORG":
                    found_ent = "ORG"
                elif mention[idx].ent_type_ == "EVENT":
                    found_ent = "EVENT"
                elif mention[idx].ent_type_ == "TIME":
                    found_ent = "TIME"
                elif mention[idx].ent_type_ == "PRODUCT":
                    found_ent = "PROD"
                elif mention[idx].ent_type_ == "QUANTITY":
                    found_ent = "AMOUNT"
                elif mention[idx].ent_type_ == "PRODUCT":
                    found_ent = "PROD"
                elif mention[idx].ent_type_ == "NORP":
                    found_ent = "FONC"
                else:
                    found_ent = "UNK"
                    break

        return(found_ent)


    def check_id_form(self, mention_1, mention_2):
        if mention_1.text.lower() == mention_2.text.lower():
            return("YES")
        else:
            return("NO")

    def check_sub_form(self, mention_1, mention_2):
        if ((mention_1.text.lower() in mention_2.text.lower()) or
                (mention_2.text.lower() in mention_1.text.lower())):
            return("YES")
        else:
            return("NO")

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def check_incl_rate(self, mention_1, mention_2):
        words_list_l = mention_1.text.lower().split()
        words_list_2 = mention_2.text.lower().split()
        len_of_words_min = min(len(words_list_l), len(words_list_2))
        len_intersection = len(self.intersection(words_list_l, words_list_2))
        return(len_intersection/len_of_words_min)

    def check_com_rate(self, mention_1, mention_2):
        words_list_l = mention_1.text.lower().split()
        words_list_2 = mention_2.text.lower().split()
        len_of_words_max = max(len(words_list_l), len(words_list_2))
        len_intersection = len(self.intersection(words_list_l, words_list_2))
        return(len_intersection/len_of_words_max)


    def calc_distance_num_chars(self, mention_1, mention_2):

        distance = 0

        if mention_1 not in self.doc_mentions_list:
            return(distance)

        if mention_2 not in self.doc_mentions_list:
            return(distance)

        if mention_1.start == mention_2.start:
            return(distance)

        if mention_1.start > mention_2.start:
            mention_1, mention_2 = mention_2, mention_1

        for idx in range(mention_1.end, mention_2.start):
            distance += len(self.doc[idx])

        return(distance)

    def calc_distance_num_words(self, mention_1, mention_2):

        distance = 0

        if mention_1 not in self.doc_mentions_list:
            return(distance)

        if mention_2 not in self.doc_mentions_list:
            return(distance)

        if mention_1.start == mention_2.start:
            return(distance)

        if mention_1.start > mention_2.start:
            mention_1, mention_2 = mention_2, mention_1

        return(mention_2.start-mention_1.end)

    def calc_distance_num_mentions(self, mention_1, mention_2):

        distance = 0

        if mention_1 not in self.doc_mentions_list:
            return(distance)

        if mention_2 not in self.doc_mentions_list:
            return(distance)

        if mention_1.start > mention_2.start:
            mention_1, mention_2 = mention_2, mention_1

        for idx, mention in enumerate(self.doc_mentions_list):
            if mention.start == mention_1.start:
                start_index = idx
            if mention.start == mention_2.start:
                end_index = idx
                distance = end_index-start_index-1
                if distance == -1:
                    distance = 0
                return(distance)

        return(distance)


    def find_prev_next(self, mention_1, mention_2):

        tf2yn = {True: "YES", False: "NO"}

        mention_1_prev = mention_1.start-1
        mention_2_prev = mention_2.start-1

        mention_1_next = mention_1.end+1
        mention_2_next = mention_2.end+1


        if (mention_1_prev < 0 or mention_2_prev < 0):
            ID_PREV = "NO"
        else:
            bool_is_prev = self.doc[mention_1_prev].text == self.doc[mention_2_prev].text
            ID_PREV = tf2yn[bool_is_prev]

        max_token_num = self.doc.__len__()

        if (mention_1_next >= max_token_num or mention_2_next >= max_token_num):
            ID_NEXT = "NO"
        else:
            bool_is_next = self.doc[mention_1_next].text == self.doc[mention_2_next].text
            ID_NEXT = tf2yn[bool_is_next]

        return(ID_PREV, ID_NEXT)


    def init_json_mention_pairs(self):

            id = 0
            json_mention_pairs = []

            # m1_TYPE	m2_TYPE
            # m1_GP	m2_GP	m1_GENDER	m2_GENDER	m1_NUMBER	m2_NUMBER	m1_EN	m2_EN	ID_FORM	ID_SUBFORM
            # INCL_RATE	COM_RATE	ID_DEF	ID_GP	ID_TYPE	ID_EN	ID_GENDER	ID_NUMBER	DISTANCE_MENTION
            # DISTANCE_WORD	DISTANCE_CHAR	EMBEDDED	ID_PREVIOUS	ID_NEXT	IS_CO_REF

            for (left_mention, right_mention) in self.doc_mention_pairs_list:
                id += 1
                coref_pair = {}
                coref_pair["coref_pair_id"] = id
                coref_pair["coref_left_mention"] = left_mention
                coref_pair["coref_right_mention"] = right_mention
                coref_pair["coref_mentions_features"] = {
                    "m1_TYPE": "UNK",
                    # {N, PR, UNK, NULL}
                    "m2_TYPE": "UNK",
                    # {N, PR, UNK, NULL}
                    "m1_GP": "UNK",
                    # {N, PR, UNK, NULL}
                    "m2_GP": "UNK",
                    # {N, PR, UNK, NULL}
                    "m1_GENDER": "UNK",
                    # {M, F, UNK, NULL}
                    "m2_GENDER": "UNK",
                    # {SG, PL, UNK, NULL}
                    "m1_NUMBER": "NULL",
                    # {SG, PL, UNK, NULL}
                    "m2_NUMBER": "NULL",
                    # {PERS, FONC, LOC, ORG, PROD, TIME, AMOUNT, EVENT, NO, UNK, NULL}
                    "m1_EN": "UNK",
                    # {PERS, FONC, LOC, ORG, PROD, TIME, AMOUNT, EVENT, NO, UNK, NULL}
                    "m2_EN": "UNK"
                }

                coref_pair["coref_pair_relation_features"] = {
                    "ID_FORM": "NA",         # {YES, NO, NA}
                    "ID_SUBFORM": "NA",      # {YES, NO, NA}
                    "INCL_RATE": 0.0,          # REAL NUM
                    "COM_RATE": 0.0,          # REAL NUM
                    # {YES, NO, NA}
                    "ID_GP": "NA",
                    # {YES, NO, NA}
                    "ID_TYPE": "NA",
                    # {YES, NO, NA}
                    "ID_EN": "NA",
                    # {YES, NO, UNK}
                    "ID_GENDER": "NA",
                    # {YES, NO, UNK}
                    "ID_NUMBER": "NA",
                    "DISTANCE_MENTION": 0.0,  # REAL NUM
                    "DISTANCE_WORD": 0.0,    # REAL NUM
                    "DISTANCE_CHAR": 0.0,   # REAL NUM
                    # {YES, NO, NA}
                    "EMBEDDED": "NA",
                    # {YES, NO, NA}
                    "ID_PREVIOUS": "NA",
                    # {YES, NO, NA}
                    "ID_NEXT": "NA",
                    # {COREF, NOT_COREF}
                    "IS_CO_REF": 0
                }
                json_mention_pairs.append(coref_pair)

            self.json_mention_pairs = json_mention_pairs
            return(self.json_mention_pairs)


    def generate_json_mention_pairs(self):

        self.init_json_mention_pairs()
        tf2yn = {True: "YES", False: "NO"}

        for coref_pair_id in range(len(self.json_mention_pairs)):

            left_mention = self.json_mention_pairs[coref_pair_id]["coref_left_mention"]
            right_mention = self.json_mention_pairs[coref_pair_id]["coref_right_mention"]

            m1_TYPE=self.find_pos(left_mention)
            m2_TYPE = self.find_pos(right_mention)
            m1_GP = self.find_prep(right_mention)
            m2_GP = self.find_prep(left_mention)
            m1_GENDER = self.find_gender(left_mention)
            m2_GENDER = self.find_gender(right_mention)
            m1_NUMBER = self.find_number(left_mention)
            m2_NUMBER = self.find_number(right_mention)
            m1_EN = self.find_entity(right_mention)
            m2_EN = self.find_entity(left_mention)


            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_GENDER"] = m1_GENDER
            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_GENDER"] = m2_GENDER

            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_NUMBER"] = m1_NUMBER
            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_NUMBER"] = m2_NUMBER

            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_GP"] = m1_GP
            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_GP"] = m2_GP

            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_EN"] = m1_EN
            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_EN"] = m2_EN

            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_TYPE"] = m1_TYPE
            self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_TYPE"] = m2_TYPE

            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_FORM"] = self.check_id_form(
                left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_SUBFORM"] = self.check_sub_form(
                left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["INCL_RATE"] = self.check_incl_rate(
                left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["COM_RATE"] = self.check_com_rate(
                left_mention, right_mention)


            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["DISTANCE_WORD"] = self.calc_distance_num_words(
                left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["DISTANCE_CHAR"] = self.calc_distance_num_chars(
                left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["DISTANCE_MENTION"] = self.calc_distance_num_mentions(
                left_mention, right_mention)
            
            bool_is_GP = self.json_mention_pairs[coref_pair_id]["coref_mentions_features"][
                "m1_GP"] == self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_GP"]

            bool_is_TYPE = self.json_mention_pairs[coref_pair_id]["coref_mentions_features"][
                "m1_GENDER"] == self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_GENDER"]

            bool_is_GENDER = self.json_mention_pairs[coref_pair_id]["coref_mentions_features"][
                "m1_TYPE"] == self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_TYPE"]
            bool_is_NUMBER = self.json_mention_pairs[coref_pair_id]["coref_mentions_features"][
                "m1_NUMBER"] == self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m1_NUMBER"]

            bool_is_EN = self.json_mention_pairs[coref_pair_id]["coref_mentions_features"][
                "m1_EN"] == self.json_mention_pairs[coref_pair_id]["coref_mentions_features"]["m2_EN"]
            

            m1 = left_mention[0]
            m2 = right_mention[0]


            ID_GP = tf2yn[bool_is_GP]
            if m1_GP == "UNK" or m2_GP == "UNK":
                ID_GP = "NA"

            ID_TYPE = tf2yn[bool_is_TYPE]
            if m1_TYPE == "UNK" or m2_TYPE == "UNK":
                ID_TYPE = "NA"

            ID_EN = tf2yn[bool_is_EN]
            if m1_EN == "UNK" or m2_EN == "UNK":
                ID_EN = "NA"

            ID_GENDER = tf2yn[bool_is_GENDER]
            if m1_GENDER == "UNK" or m2_GENDER == "UNK":
                ID_GENDER = "NA"

            ID_NUMBER = tf2yn[bool_is_NUMBER]
            if m1_NUMBER == "UNK" or m2_NUMBER == "UNK":
                ID_NUMBER = "NA"

            EMBEDDED = tf2yn[((m1.text in m2.text) or (m2.text in m1.text))]



            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_GP"] = ID_GP
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_TYPE"] = ID_TYPE
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_EN"] = ID_EN
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_GENDER"] = ID_GENDER
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_NUMBER"] = ID_NUMBER
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["EMBEDDED"] = EMBEDDED

            prev_next_pair=self.find_prev_next(left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_PREVIOUS"] = prev_next_pair[0]
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_NEXT"] = prev_next_pair[1]

        return(self.json_mention_pairs)


    def json_mention_pairs2dataframe(self, save_file=True, file_path="./unseen_data.xlsx"):

        df_rows=[]
        for coref_pair in self.json_mention_pairs:
            row=[]
            row.append(coref_pair["coref_pair_id"])
            row.append(coref_pair["coref_left_mention"].text)
            row.append(coref_pair["coref_right_mention"].text)
            row.append(coref_pair["coref_mentions_features"]["m1_TYPE"])
            row.append(coref_pair["coref_mentions_features"]["m2_TYPE"])
            row.append(coref_pair["coref_mentions_features"]["m1_GP"])
            row.append(coref_pair["coref_mentions_features"]["m2_GP"])
            row.append(coref_pair["coref_mentions_features"]["m1_GENDER"])
            row.append(coref_pair["coref_mentions_features"]["m2_GENDER"])
            row.append(coref_pair["coref_mentions_features"]["m1_NUMBER"])
            row.append(coref_pair["coref_mentions_features"]["m2_NUMBER"])
            row.append(coref_pair["coref_mentions_features"]["m1_EN"])
            row.append(coref_pair["coref_mentions_features"]["m2_EN"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_FORM"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_SUBFORM"])
            row.append(coref_pair["coref_pair_relation_features"]["INCL_RATE"])
            row.append(coref_pair["coref_pair_relation_features"]["COM_RATE"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_GP"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_TYPE"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_EN"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_GENDER"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_NUMBER"])
            row.append(coref_pair["coref_pair_relation_features"]["DISTANCE_MENTION"])
            row.append(coref_pair["coref_pair_relation_features"]["DISTANCE_WORD"])
            row.append(coref_pair["coref_pair_relation_features"]["DISTANCE_CHAR"])
            row.append(coref_pair["coref_pair_relation_features"]["EMBEDDED"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_PREVIOUS"])
            row.append(coref_pair["coref_pair_relation_features"]["ID_NEXT"])
            row.append(coref_pair["coref_pair_relation_features"]["IS_CO_REF"])

            df_rows.append(row)

        header = ["coref_pair_id",
                "Left_ID",
                "Right_ID",
                "m1_TYPE",
                "m2_TYPE",
                "m1_GP",
                "m2_GP",
                "m1_GENDER",
                "m2_GENDER",
                "m1_NUMBER",
                "m2_NUMBER",
                "m1_EN",
                "m2_EN",
                "ID_FORM",
                "ID_SUBFORM",
                "INCL_RATE",
                "COM_RATE",
                "ID_GP",
                "ID_TYPE",
                "ID_EN",
                "ID_GENDER",
                "ID_NUMBER",
                "DISTANCE_MENTION",
                "DISTANCE_WORD",
                "DISTANCE_CHAR",
                "EMBEDDED",
                "ID_PREVIOUS",
                "ID_NEXT",
                "IS_CO_REF"]

        self.dataframe_mention_pairs = pd.DataFrame(df_rows)
        self.dataframe_mention_pairs.columns=header

        if save_file:           
            self.dataframe_mention_pairs.to_excel(file_path, index=False)

        return self.dataframe_mention_pairs

    def init_data_model(self, model_name, input_method="dataframe", file_path="./unseen_data.xlsx", column_outcome="Prediction", threshold=0.5):
        
        self.input_method=input_method
        self.file_path = file_path

        if input_method=="dataframe":
            self.model_input_dataframe = self.dataframe_mention_pairs.copy()
        elif input_method == ".xlsx":
            self.model_input_dataframe = pd.read_excel(file_path, index_col=0)
        self.model_name = model_name
        self.column_outcome = column_outcome
        self.values_list = [
            'AMOUNT', 'DEF_DEM', 'DEF_SPLE', 'EVENT',
            'EXPL', 'F', 'FONC', 'INDEF', 'LOC', 'M', 'N',
            'NO', 'ORG', 'PERS', 'PL', 'PR', 'PROD', 'SG',
            'TIME', 'UNK', 'VOID', 'YES'
        ]
        self.values_list_for_replace = range(1, 23)
        self.columns_drop_list = ["Left_ID", "Right_ID", "DISTANCE_MENTION",
                                "DISTANCE_WORD", "DISTANCE_CHAR"]
        self.threshold = threshold

    def apply_model_to_dataset(self):

        model = joblib.load(self.model_name)

        model_input_dataframe_ = self.model_input_dataframe.drop(
            self.columns_drop_list, axis=1)
        model_input_dataframe_.fillna("VOID", inplace=True)
        model_input_dataframe_ = model_input_dataframe_.replace(
            self.values_list, self.values_list_for_replace)

        if "Prediction" in (list(model_input_dataframe_.columns)):
            y_pred = model.predict(model_input_dataframe_.iloc[:, 0:-2])
        else:
            y_pred = model.predict(model_input_dataframe_.iloc[:, 0:-1])

        binary_y_pred = []

        for x in y_pred:
            if x >= self.threshold:
                binary_y_pred.append(1)
            else:
                binary_y_pred.append(0)

        self.model_input_dataframe[self.column_outcome] = y_pred

        if self.input_method == ".xlsx":
            self.model_input_dataframe.to_excel(self.file_path)

        return (self.model_input_dataframe)
