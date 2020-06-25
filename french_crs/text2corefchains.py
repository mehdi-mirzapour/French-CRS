from ufal.udpipe import Model, InputFormat, OutputFormat, ProcessingError, Sentence
import os
import spacy
import json
import pandas as pd
import re
from bs4 import BeautifulSoup
import joblib
from pathlib import Path
from spacy.matcher import Matcher



class udpipe_spacy_lang_model:

    def __init__(self, udpipe_lang_model, spacy_lang_model):
        """Load given model."""
        self.spacy_lang_model = spacy.load(str(Path(spacy_lang_model)))
        self.udpipe_lang_model = Model.load(str(Path(udpipe_lang_model)))
        if not self.udpipe_lang_model:
            raise Exception(
                "Cannot load UDPipe model from file '%s'" % udpipe_lm_path)
        

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.udpipe_lang_model.newTokenizer(
            self.udpipe_lang_model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        
        self.text_tokenized = self._read(text, tokenizer)

        return self.text_tokenized

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ProcessingError()
        sentences = []

        sentence = Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.udpipe_lang_model.tag(sentence, self.udpipe_lang_model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.udpipe_lang_model.parse(sentence, self.udpipe_lang_model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output



    def parse_sentences_json(self):
        """Write given ufal.udpipe.Sentence-s in the json format."""

        for s in self.text_tokenized:
            self.udpipe_lang_model.tag(s, self.udpipe_lang_model.DEFAULT)
            self.udpipe_lang_model.parse(s, self.udpipe_lang_model.DEFAULT)

        output_format = OutputFormat.newOutputFormat("epe")

        parsed_sentences = []
        counter = 0
        for sentence in self.text_tokenized:
            counter += 1
            parsed_sent = {}
            output = ''
            output = output_format.writeSentence(sentence)
            output += output_format.finishDocument()
            jsonformat = json.loads(output)
            parsed_sent["Sent_ID"] = counter
            parsed_sent["Sent_Tokenized"] = ""
            parsed_sent["Sent_Mentions"] = []
            parsed_sent["Sent_Entities_spaCy"] = []
            parsed_sent["Content"] = jsonformat["nodes"]
            parsed_sentences.append(parsed_sent)
        for i in range(0, len(parsed_sentences)):
            words = ""
            for j in range(0, len(parsed_sentences[i]["Content"])):
                words += parsed_sentences[i]["Content"][j]["form"]+" "
            parsed_sentences[i]["Sent_Tokenized"] = words


        self.sentences_json=parsed_sentences
        return parsed_sentences

    def remove_duplicates(self,oldlist):
        cleanlist = []

        for x in oldlist:
            if x not in cleanlist:
                cleanlist.append(x)
        return(cleanlist)


    def n_gram_word_forms_json(self, sent_json, n_size):
            n_grams = []
            forms_lenght = len(sent_json)

            if n_size >= forms_lenght:
                n_size = forms_lenght

            for form_id in range(0, forms_lenght-n_size+1):
                words = ""
                for entity_id in range(0, n_size):
                    words += sent_json[form_id+entity_id]["form"]+" "
                n_grams.append(words[0:-1])
            return(n_grams)


    def find_entities_json_sents_using_spacy(self, operation="in"):

        list_entities = []

        for sent_id in range(0, len(self.sentences_json)):

            text = self.sentences_json[sent_id]["Sent_Tokenized"]
            doc = self.spacy_lang_model(text)

            for entity in doc.ents:

                count = 0
                sent = self.sentences_json[sent_id]["Content"]

                entity_length = len(entity.text.strip().split(" "))
                n_grams = self.n_gram_word_forms_json(sent, entity_length)

                for n_gram in n_grams:
                    count += 1

                    if (operation == "in"):
                        if (entity.text in n_gram):
                            list_entities.append([entity.text, entity.label_, (sent_id+1),
                                                [id for id in range(count, count+entity_length)]])
                    elif (operation == "=="):
                        if entity.text == n_gram:
                            list_entities.append([entity.text, entity.label_, (sent_id+1),
                                                [id for id in range(count, count+entity_length)]])
        
        self.spacy_detected_entities = self.remove_duplicates(list_entities)
        return(self.spacy_detected_entities)



    def update_json_sents_with_spacy_entities(self,operation="in"):

        list_entities = self.find_entities_json_sents_using_spacy(
            operation="in")
        
        for sent_id in range(0, len(self.sentences_json)):
            self.sentences_json[sent_id]["Sent_Entities_spaCy"] = []
            for entity_id in range(0, len(list_entities)):
                if (self.sentences_json[sent_id]["Sent_ID"] == list_entities[entity_id][2]):
                    self.sentences_json[sent_id]["Sent_Entities_spaCy"].append(
                        list_entities[entity_id])


    
    def find_mentions_in_tokenized_text(self, relative_path=".."):

        udpipe_parsed_text = ""
        for index in range(0, len(self.sentences_json)):
            sent=self.sentences_json[index]["Sent_Tokenized"]
            udpipe_parsed_text += sent

        file_raw_text = open(str(Path(relative_path+"/mention_detector_files/raw_text.txt")), 'w')
        file_raw_text.write(udpipe_parsed_text)
        file_raw_text.close()

        os.system(str(Path(relative_path+"/french_crs/mention_detector.sh "+relative_path)))

        with open(str(Path(relative_path+"/mention_detector_files/mentions.json"))) as json_file:
            mentions = json.load(json_file)

        filtered_mentions = []

        for i in range(0, len(mentions)):
            if mentions[i]["sys_tag"] != None:
                filtered_mentions.append(mentions[i])
        
        self.json_mentions=filtered_mentions

        list_mentions = []

        for mentions in self.json_mentions:
            words = []
            for word in mentions["content"]:
                words.append(word)
            if words != []:
                list_mentions.append(words)
        
        self.list_mentions = self.remove_duplicates(list_mentions)
        
        self.list_mentions_merged_words = [" ".join(self.list_mentions[id])
                         for id in range(len(self.list_mentions))]

        return (self.list_mentions_merged_words)



    def remove_included_mentions(self, iteration):

        for rep in range(iteration):

            removed_list_mentions = []

            for idx in range(len(self.mentions_ids_in_json_sents)-1):
                first_ment = self.mentions_ids_in_json_sents[idx][2]
                second_ment = self.mentions_ids_in_json_sents[idx+1][2]

                if self.mentions_ids_in_json_sents[idx][1] == self.mentions_ids_in_json_sents[idx+1][1]:
                    x = set(range(first_ment[0], first_ment[-1]+1))
                    y = set(range(second_ment[0], second_ment[-1]+1))
                    if y.issubset(x):
                        removed_list_mentions.append(
                            self.mentions_ids_in_json_sents[idx+1])

            new_list_mentions = [
                ment for ment in self.mentions_ids_in_json_sents if ment not in removed_list_mentions]

            self.mentions_ids_in_json_sents = new_list_mentions

        return(self.mentions_ids_in_json_sents)


    def find_mention_ids_in_sents_json(self, operation="==", relative_path_mention_detector = ".."):

        self.find_mentions_in_tokenized_text(
            relative_path=str(Path(relative_path_mention_detector)))
        mentions_ids_in_json_sents = []
        for sent_id in range(0, len(self.sentences_json)):
            for mention in self.list_mentions_merged_words:
                mention_length = len(mention.strip().split(" "))
                n_grams = self.n_gram_word_forms_json(
                    self.sentences_json[sent_id]["Content"], mention_length)
                count = 0
                for n_gram in n_grams:
                    count += 1
                    if (operation == "in"):
                        if mention in n_gram:
                            mentions_ids_in_json_sents.append([mention, sent_id+1,
                                                              [id for id in range(count, count+mention_length)]])
                    elif (operation == "=="):
                        if mention == n_gram:
                            mentions_ids_in_json_sents.append([mention, sent_id+1,
                                                               [id for id in range(count, count+mention_length)]])
                self.mentions_ids_in_json_sents = self.remove_duplicates(
                    mentions_ids_in_json_sents)

        self.remove_included_mentions(iteration=3)

        return(self.mentions_ids_in_json_sents)

    def update_json_sents_with_mentions(self, operation="==", relative_path_mention_detector=".."):

        self.find_mention_ids_in_sents_json(
            operation, str(Path(relative_path_mention_detector)))
        for sent_id in range(0, len(self.sentences_json)):
            self.sentences_json[sent_id]["Sent_Mentions"] = []
            for mention_id in range(0, len(self.mentions_ids_in_json_sents)):
                if (self.sentences_json[sent_id]["Sent_ID"] == self.mentions_ids_in_json_sents[mention_id][1]):
                    self.sentences_json[sent_id]["Sent_Mentions"].append(
                        self.mentions_ids_in_json_sents[mention_id])

        return(self.sentences_json)




class mentions2chains:

    def __init__(self, mentions_ids_in_json_sents):
       self.mentions_ids_in_json_sents = mentions_ids_in_json_sents


    def remove_included_mentions(self,iteration):

        for rep in range(iteration):
                
            removed_list_mentions = []

            for idx in range(len(self.mentions_ids_in_json_sents)-1):
                first_ment = self.mentions_ids_in_json_sents[idx][2]
                second_ment = self.mentions_ids_in_json_sents[idx+1][2]

                if self.mentions_ids_in_json_sents[idx][1] == self.mentions_ids_in_json_sents[idx+1][1]:
                    x = set(range(first_ment[0], first_ment[-1]+1))
                    y = set(range(second_ment[0], second_ment[-1]+1))
                    if y.issubset(x):
                        removed_list_mentions.append(
                            self.mentions_ids_in_json_sents[idx+1])

            new_list_mentions = [
                ment for ment in self.mentions_ids_in_json_sents if ment not in removed_list_mentions]

            
            self.mentions_ids_in_json_sents = new_list_mentions

        return(self.mentions_ids_in_json_sents)


    def remove_duplicates(self, oldlist):
        cleanlist = []

        for x in oldlist:
            if x not in cleanlist:
                cleanlist.append(x)
        return(cleanlist)



    def compare_mentions(self, mention_list_1, mention_list_2):
        if (mention_list_1[1] == mention_list_2[1]):
            if (mention_list_1[2][0] < mention_list_2[2][0]):
                return(True)
            else:
                return(False)
        elif(mention_list_1[1] < mention_list_2[1]):
            return(True)
        else:
            return(False)


    def mentions_sort(self):
        n = len(self.mentions_ids_in_json_sents)

        for i in range(n):
            for j in range(0, n-i-1):
                if self.compare_mentions(self.mentions_ids_in_json_sents[j+1], self.mentions_ids_in_json_sents[j]):
                    self.mentions_ids_in_json_sents[j], self.mentions_ids_in_json_sents[j+1] = self.mentions_ids_in_json_sents[j+1], self.mentions_ids_in_json_sents[j]
                    
    def generate_mention_pairs(self, window_size=5, iteration=3):

        self.mentions_ids_in_json_sents = self.remove_duplicates(
                                            self.mentions_ids_in_json_sents)
        self.mentions_sort()
        self.remove_included_mentions(iteration)
        
        list_mention_pairs = []
        n = len(self.mentions_ids_in_json_sents)

        if window_size >= n:
            window_size=n-1

        # Traverse through all array elements
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, i+1):
                if i != j and (i-j) < window_size+1:
                    list_mention_pairs.append(
                        (self.mentions_ids_in_json_sents[j], self.mentions_ids_in_json_sents[i]))
        self.list_mention_pairs = list_mention_pairs

        return(self.list_mention_pairs)


    def find_gender(self,parsed_sentences, mention_list):
        found_gender = "UNK"
        for token_id in mention_list[2]:
            for sent in parsed_sentences:
                if sent["Sent_ID"] == mention_list[1]:
                    for token in sent['Content']:
                        if token["id"] == token_id and token["properties"]["upos"] in ["NOUN", "PRON"]:
                            if "Gender" in token["properties"]:
                                if token["properties"]["Gender"] == "Masc":
                                    found_gender = "M"
                                elif token["properties"]["Gender"] == "Fem":
                                    found_gender = "F"
        return(found_gender)



    def find_number(self,parsed_sentences, mention_list):
        find_number = "UNK"
        for token_id in mention_list[2]:
            for sent in parsed_sentences:
                if sent["Sent_ID"] == mention_list[1]:
                    for token in sent['Content']:
                        if token["id"] == token_id and token["properties"]["upos"] in ["NOUN", "PRON"]:
                            if "Number" in token["properties"]:
                                if token["properties"]["Number"] == "Sing":
                                    find_number = "SG"
                                elif token["properties"]["Number"] == "Plur":
                                    find_number = "PL"
        return(find_number)



    def find_prep(self,parsed_sentences, mention_list):
        find_prep = "NO"
        for token_id in mention_list[2]:
            for sent in parsed_sentences:
                if sent["Sent_ID"] == mention_list[1]:
                    for token in sent['Content']:
                        if token["id"] == token_id and token["properties"]["upos"] in ["NOUN", "PRON"]:
                            if "edges" in token:
                                for edge in token["edges"]:
                                    if edge["label"] == "case":
                                        find_prep = "YES"
        return(find_prep)

    def find_pos(self, parsed_sentences, mention_list):
        find_pos = "UNK"
        for token_id in mention_list[2]:
            for sent in parsed_sentences:
                if sent["Sent_ID"] == mention_list[1]:
                    for token in sent['Content']:
                        if token["id"] == token_id and token["properties"]["upos"] in ["NOUN", "PRON"]:
                            if token["properties"]["upos"] == "NOUN":
                                find_pos = "N"
                            else:
                                find_pos = "PR"
        return(find_pos)


    def find_entity(self, parsed_sentences, mention_list):

        _find_ent = "UNK"
        for token_id in mention_list[2]:
            for sent in parsed_sentences:
                if sent["Sent_ID"] == mention_list[1]:
                    for ent in sent["Sent_Entities_spaCy"]:
                        if token_id in ent[3]:
                            _find_ent = ent[1]

        # Spacy	  ANCOR
        # ==============
        # PER	  PERS
        # LOC	  LOC
        # ORG	  ORG
        # MISC	  AMOUNT,EVENT,FONC,PROD, TIME
        # VOID

        if _find_ent == "PER":
            find_ent = "PERS"
        elif _find_ent == "LOC":
            find_ent = "LOC"
        elif _find_ent == "ORG":
            find_ent = "ORG"
        else:
            find_ent = "UNK"

        return(find_ent)

    def find_prev_next(self, parsed_sentences, mention_list1, mention_list2):

        mention1_size = len(parsed_sentences[mention_list1[1]-1]["Content"])
        mention1_prev_id = mention_list1[2][0]-2
        mention1_next_id = mention_list1[2][-1]
        mention1_sent_id = mention_list1[1]-1

        if mention1_prev_id < 0:
            mention1_prev = "*!*!*"
        else:
            mention1_prev = parsed_sentences[mention1_sent_id]["Content"][mention1_prev_id]["form"]


        if mention1_next_id >= mention1_size:
            mention1_next = "*!*!*"
        else:
            mention1_next = parsed_sentences[mention1_sent_id]["Content"][mention1_next_id]["form"]


        mention2_size = len(parsed_sentences[mention_list2[1]-1]["Content"])
        mention2_prev_id = mention_list2[2][0]-2
        mention2_next_id = mention_list2[2][-1]
        mention2_sent_id = mention_list2[1]-1


        if mention2_prev_id < 0:
            mention2_prev = "*!*!*"
        else:
            mention2_prev = parsed_sentences[mention2_sent_id]["Content"][mention2_prev_id]["form"]


        if mention2_next_id >= mention2_size:
            mention2_next = "*!*!*"
        else:
            mention2_next = parsed_sentences[mention2_sent_id]["Content"][mention2_next_id]["form"]

        tf2yn = {True: "YES", False: "NO"}

        bool_is_next = mention1_next == mention2_next
        bool_is_prev = mention1_prev == mention2_prev


        ID_NEXT = tf2yn[bool_is_next]
        ID_PREV = tf2yn[bool_is_prev]

        if mention1_next == "*!*!*" or mention2_next == "*!*!*":
            ID_NEXT = "NO"

        if mention1_prev == "*!*!*" or mention2_prev == "*!*!*":
            ID_PREV = "NO"

        return(ID_PREV,ID_NEXT)


    def check_id_form(self, mention_list_1, mention_list_2):
        if mention_list_1[0] == mention_list_2[0]:
            return("YES")
        else:
            return("NO")

    def check_sub_form(self, mention_list_1, mention_list_2):
        if ((mention_list_1[0] in mention_list_2[0]) or
                (mention_list_2[0] in mention_list_1[0])):
            return("YES")
        else:
            return("NO")

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def check_incl_rate(self, mention_list_1, mention_list_2):
        words_list_l = mention_list_1[0].lower().split()
        words_list_2 = mention_list_2[0].lower().split()
        len_of_words_min = min(len(words_list_l), len(words_list_2))
        len_intersection = len(self.intersection(words_list_l, words_list_2))
        return(len_intersection/len_of_words_min)

    def check_com_rate(self, mention_list_1, mention_list_2):
        words_list_l = mention_list_1[0].lower().split()
        words_list_2 = mention_list_2[0].lower().split()
        len_of_words_max = max(len(words_list_l), len(words_list_2))
        len_intersection = len(self.intersection(words_list_l, words_list_2))
        return(len_intersection/len_of_words_max)

    def return_ordered_mention(self, mention_list_1, mention_list_2):
        if (mention_list_1[1] == mention_list_2[1]):
            if (mention_list_1[2][0] < mention_list_2[2][0]):
                return([mention_list_1, mention_list_2])
            else:
                return([mention_list_2, mention_list_1])
        elif(mention_list_1[1] < mention_list_2[1]):
            return([mention_list_1, mention_list_2])
        else:
            return([mention_list_2, mention_list_1])


    def get_words_in_sentence(self, json_parsed_sentences_list, sent_tofind_id, start_word_id, end_word_id, end_word="No"):
        words_in_sentence = []
        for sent_id in range(len(json_parsed_sentences_list)):
            if (json_parsed_sentences_list[sent_id-1]["Sent_ID"] == sent_tofind_id):
                end_word_of_sentence = len(
                    json_parsed_sentences_list[sent_id-1]["Content"])
                if (end_word == "Yes"):
                    end_word_id = end_word_of_sentence
                for word_id in range(start_word_id-1, end_word_id):
                    words_in_sentence.append(
                        (json_parsed_sentences_list[sent_id-1]["Content"][word_id]["form"]))
        return(words_in_sentence)


    def get_distance_words(self, json_parsed_sentences_list, mention_list_1, mention_list_2, present_mode="Simple_List"):
        first_mention = self.return_ordered_mention(
            mention_list_1, mention_list_2)[0]
        second_mention = self.return_ordered_mention(
            mention_list_1, mention_list_2)[1]

        words_list = []
        words = []

        if first_mention[1] != second_mention[1]:
            words_list.append(self.get_words_in_sentence(
                json_parsed_sentences_list, first_mention[1], first_mention[2][-1]+1, 1, "Yes"))
            for sent_id in range(first_mention[1]+1, second_mention[1]):
                    words_list.append(self.get_words_in_sentence(
                        json_parsed_sentences_list, sent_id, 1, 1, "Yes"))
            words_list.append(self.get_words_in_sentence(
                json_parsed_sentences_list, second_mention[1], 1, second_mention[2][0]-1))
        else:
            words_list.append(self.get_words_in_sentence(json_parsed_sentences_list,
                                                    first_mention[1], first_mention[2][-1]+1, second_mention[2][0]-1))

        if present_mode == "List_of_List":
            return(words_list)
        else:
            for words_list in words_list:
                words += words_list
            return(words)


    def calc_distance_num_words(self, json_parsed_sentences_list, mention_list_1, mention_list_2):
        words = self.get_distance_words(
            json_parsed_sentences_list, mention_list_1, mention_list_2)
        return(len(words))


    def calc_distance_num_chars(self, json_parsed_sentences_list, mention_list_1, mention_list_2):
        words = self.get_distance_words(
            json_parsed_sentences_list, mention_list_1, mention_list_2)
        acc_word_num = 0
        for word in words:
            acc_word_num += len(word)
        return(acc_word_num)


    def calc_distance_num_mentions(self, list_mentions, mention_list_1, mention_list_2):

        first_mention = self.return_ordered_mention(mention_list_1, mention_list_2)[0]
        second_mention = self.return_ordered_mention(
            mention_list_1, mention_list_2)[1]

        context_left = (first_mention[1], first_mention[2][-1])
        context_right = (second_mention[1], second_mention[2][0])

        relevant_mentions = []

        for mention in list_mentions:


            check_after_left = (mention[1], mention[2][0])
            check_before_right = (mention[1], mention[2][-1])

            if check_after_left > context_left and check_before_right < context_right:
                relevant_mentions.append(mention)

        return(len(relevant_mentions))



    def init_json_mention_pairs(self):

        id = 0
        json_mention_pairs = []

        # m1_TYPE	m2_TYPE	
        # m1_GP	m2_GP	m1_GENDER	m2_GENDER	m1_NUMBER	m2_NUMBER	m1_EN	m2_EN	ID_FORM	ID_SUBFORM
        # INCL_RATE	COM_RATE	ID_DEF	ID_GP	ID_TYPE	ID_EN	ID_GENDER	ID_NUMBER	DISTANCE_MENTION
        # DISTANCE_WORD	DISTANCE_CHAR	EMBEDDED	ID_PREVIOUS	ID_NEXT	IS_CO_REF

        for (left_mention, right_mention) in self.list_mention_pairs:
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

    def generate_json_mention_pairs(self, list_mentions, sentences_json):

        self.init_json_mention_pairs()
        tf2yn = {True: "YES", False: "NO"}

        for coref_pair_id in range(len(self.json_mention_pairs)):

            left_mention = self.json_mention_pairs[coref_pair_id]["coref_left_mention"]
            right_mention = self.json_mention_pairs[coref_pair_id]["coref_right_mention"]


            m1_TYPE=self.find_pos(sentences_json, left_mention)
            m2_TYPE = self.find_pos(sentences_json, right_mention)
            m1_GP = self.find_prep(sentences_json, right_mention)
            m2_GP = self.find_prep(sentences_json, left_mention)
            m1_GENDER = self.find_gender(sentences_json, left_mention)
            m2_GENDER = self.find_gender(sentences_json, right_mention)
            m1_NUMBER = self.find_number(sentences_json, left_mention)
            m2_NUMBER = self.find_number(sentences_json, right_mention)
            m1_EN = self.find_entity(sentences_json, right_mention)
            m2_EN = self.find_entity(sentences_json, left_mention)


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
                sentences_json, left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["DISTANCE_CHAR"] = self.calc_distance_num_chars(
                sentences_json, left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["DISTANCE_MENTION"] = self.calc_distance_num_mentions(
                list_mentions, left_mention, right_mention)
            
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

            EMBEDDED = tf2yn[((m1 in m2) or (m2 in m1))]



            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_GP"] = ID_GP
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_TYPE"] = ID_TYPE
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_EN"] = ID_EN
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_GENDER"] = ID_GENDER
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_NUMBER"] = ID_NUMBER
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["EMBEDDED"] = EMBEDDED

            prev_next_pair=self.find_prev_next(
                sentences_json, left_mention, right_mention)
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_PREVIOUS"] = prev_next_pair[0]
            self.json_mention_pairs[coref_pair_id]["coref_pair_relation_features"]["ID_NEXT"] = prev_next_pair[1]

        return(self.json_mention_pairs)


    def json_mention_pairs2dataframe(self, save_file=True, file_path="./unseen_data.xlsx"):

        df_rows=[]
        for coref_pair in self.json_mention_pairs:
            row=[]
            row.append(coref_pair["coref_pair_id"])
            row.append(coref_pair["coref_left_mention"])
            row.append(coref_pair["coref_right_mention"])
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
