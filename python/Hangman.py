import json
import requests
import random
import string
import secrets
import time
import re
import collections

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None, train_dict=None):
        #self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.incorrect_guesses = []
        self.guessed = set()
        self.tries_remaining = 6

        full_dictionary_location = "words_250000_train.txt"
        #self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary = train_dict
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

        self.current_dictionary = []

        # this stores all the unique letters from the full dictionary in sorted order
        self.letter_list = sorted(set("".join(self.full_dictionary)))

        # create a list called probabilities filled with zeros
        # length of this list is determined by the number of elements in the letter_list variable.
        self.probabilities = [0] * len(self.letter_list)

        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram, self.sixgram, self.sevengram, self.eightgram = self.build_ngram_models(
            self.full_dictionary)

    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word):  # word input example: "_ p p _ e "
        '''
        Given a word with either correctly gussed letters or blanks, this function
        returns the best guess for the next letter based on the n-grams
        '''

        # keep track of incorrect guesses to update the n-grams
        self.incorrect_guesses = list(set(self.guessed_letters) - set(word))

        # only recalibrate if last guess was incorrect and running low on guesses
        if len(self.guessed_letters) > 0 and self.guessed_letters[
            -1] in self.incorrect_guesses and self.tries_remaining <= 3:
            self.recalibrate_n_grams()

        # clear out probabilities from last guess
        self.probabilities = [0] * len(self.letter_list)

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        #clean_word = word[::2]
        clean_word = word

        return self.eightgram_probs(clean_word)

    def build_ngram_models(self, dictionary):
        '''
        build nested dictionary containing occurences for n (1-5) sequences of letters
        unigrams and bigrams have an extra level for length of the word
        for unigram, take only unique letters within each word
        '''
        self.unigram = collections.defaultdict(lambda: collections.defaultdict(int))
        self.bigram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        self.trigram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        self.fourgram = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))
        self.fivegram = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))))
        self.sixgram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))))
        self.sevengram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(int)))))))
        self.eightgram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(int))))))))

        # go through each word in the dictionary
        for word in dictionary:
            # check each letter in the dictionary and update the n-gram
            for i in range(len(word) - 7):
                self.bigram[len(word)][word[i]][word[i + 1]] += 1
                self.trigram[word[i]][word[i + 1]][word[i + 2]] += 1
                self.fourgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1
                self.fivegram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]] += 1
                self.sixgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]] += 1
                self.sevengram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]] += 1
                self.eightgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]][
                    word[i + 7]] += 1
            i = len(word) - 7

            # fill out the rest of the n-grams for words too short
            if len(word) == 2:
                self.bigram[len(word)][word[0]][word[1]] += 1
            elif len(word) == 3:
                self.bigram[len(word)][word[0]][word[1]] += 1
                self.bigram[len(word)][word[1]][word[2]] += 1
                self.trigram[word[0]][word[1]][word[2]] += 1

            # fill out rest of the (1-4)-grams
            elif len(word) == 4:
                self.bigram[len(word)][word[0]][word[1]] += 1
                self.bigram[len(word)][word[1]][word[2]] += 1
                self.bigram[len(word)][word[2]][word[3]] += 1
                self.trigram[word[0]][word[1]][word[2]] += 1
                self.trigram[word[1]][word[2]][word[3]] += 1
                self.fourgram[word[0]][word[1]][word[2]][word[3]] += 1
            # fill out rest of the (1-5)-grams
            elif len(word) == 5:
                self.bigram[len(word)][word[0]][word[1]] += 1
                self.bigram[len(word)][word[1]][word[2]] += 1
                self.bigram[len(word)][word[2]][word[3]] += 1
                self.bigram[len(word)][word[3]][word[4]] += 1
                self.trigram[word[0]][word[1]][word[2]] += 1
                self.trigram[word[1]][word[2]][word[3]] += 1
                self.trigram[word[2]][word[3]][word[4]] += 1
                self.fourgram[word[0]][word[1]][word[2]][word[3]] += 1
                self.fourgram[word[1]][word[2]][word[3]][word[4]] += 1
                self.fivegram[word[0]][word[1]][word[2]][word[3]][word[4]] += 1

            elif len(word) == 6:
                self.bigram[len(word)][word[0]][word[1]] += 1
                self.bigram[len(word)][word[1]][word[2]] += 1
                self.bigram[len(word)][word[2]][word[3]] += 1
                self.bigram[len(word)][word[3]][word[4]] += 1
                self.bigram[len(word)][word[4]][word[5]] += 1
                self.trigram[word[0]][word[1]][word[2]] += 1
                self.trigram[word[1]][word[2]][word[3]] += 1
                self.trigram[word[2]][word[3]][word[4]] += 1
                self.trigram[word[3]][word[4]][word[5]] += 1
                self.fourgram[word[0]][word[1]][word[2]][word[3]] += 1
                self.fourgram[word[1]][word[2]][word[3]][word[4]] += 1
                self.fourgram[word[2]][word[3]][word[4]][word[5]] += 1
                self.fivegram[word[0]][word[1]][word[2]][word[3]][word[4]] += 1
                self.fivegram[word[1]][word[2]][word[3]][word[4]][word[5]] += 1
                self.sixgram[word[0]][word[1]][word[2]][word[3]][word[4]][word[5]] += 1

            elif len(word) >= 7:
                self.bigram[len(word)][word[i]][word[i + 1]] += 1
                self.bigram[len(word)][word[i + 1]][word[i + 2]] += 1
                self.bigram[len(word)][word[i + 2]][word[i + 3]] += 1
                self.bigram[len(word)][word[i + 3]][word[i + 4]] += 1
                self.bigram[len(word)][word[i + 4]][word[i + 5]] += 1
                self.bigram[len(word)][word[i + 5]][word[i + 6]] += 1
                self.trigram[word[i]][word[i + 1]][word[i + 2]] += 1
                self.trigram[word[i + 1]][word[i + 2]][word[i + 3]] += 1
                self.trigram[word[i + 2]][word[i + 3]][word[i + 4]] += 1
                self.trigram[word[i + 3]][word[i + 4]][word[i + 5]] += 1
                self.trigram[word[i + 4]][word[i + 5]][word[i + 6]] += 1
                self.fourgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1
                self.fourgram[word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]] += 1
                self.fourgram[word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]] += 1
                self.fourgram[word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]] += 1
                self.fivegram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]] += 1
                self.fivegram[word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]] += 1
                self.fivegram[word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]] += 1
                self.sixgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]] += 1
                self.sixgram[word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]] += 1
                self.sevengram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]][word[i + 5]][word[i + 6]] += 1

            # fill out self.unigrams
            for letter in set(word):
                self.unigram[len(word)][letter] += 1

        return self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram, self.sixgram, self.sevengram, self.eightgram

    def recalibrate_n_grams(self):
        '''
        re-tabulates the n-grams after eliminating any incorrectly guessed letters
        updates the dictionary to remove words containing incorrectly guessed letters
        '''
        # updates the dictionary to remove words containing incorrectly guessed letters
        new_dict = [word for word in self.full_dictionary if not set(word).intersection(set(self.incorrect_guesses))]
        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram, self.sixgram, self.sevengram, self.eightgram = self.build_ngram_models(
            new_dict)

    def eightgram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses eight-gram to calculate the probability of a certain letter appearing in an eight-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 7):

            # case 1: "letter letter letter letter letter letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                # anchor_letter_8 = word[i + 7]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][anchor_letter_7][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6][anchor_letter_7][
                            letter]
                        letter_count[j] += \
                            self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                                anchor_letter_5][anchor_letter_6][anchor_letter_7][
                                letter]

            # case 2: "letter letter letter letter letter letter blank letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] == '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                # anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][letter][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6][letter][
                            anchor_letter_8]
                        letter_count[j] += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6][letter][
                            anchor_letter_8]

            # case 3: "letter letter letter letter letter blank letter letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] == '_' and word[i + 6] != '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                # anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        letter][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter][anchor_letter_7][
                            anchor_letter_8]
                        letter_count[j] += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter][anchor_letter_7][
                            anchor_letter_8]

            # case 4: "letter letter letter letter blank letter letter letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] == '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                # anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 4 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                        anchor_letter_6][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_6][anchor_letter_7][
                            anchor_letter_8]
                        letter_count[j] += \
                        self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_6][anchor_letter_7][
                            anchor_letter_8]

            # case 5: "letter letter letter blank letter letter letter letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                # anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 5 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_5][
                        anchor_letter_6][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                            self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]
                        letter_count[j] += \
                            self.eightgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]

            # case 6: "letter letter blank letter letter letter letter letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                # anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 6 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.eightgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                            self.eightgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]
                        letter_count[j] += \
                            self.eightgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]

            # case 7: "letter blank letter letter letter letter letter letter"
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] != '_':
                anchor_letter_1 = word[i]
                # anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 7 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.eightgram[anchor_letter_1][letter][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                            self.eightgram[anchor_letter_1][letter][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]
                        letter_count[j] += \
                            self.eightgram[anchor_letter_1][letter][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]

            # case 8: "blank letter letter letter letter letter letter letter"
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_' and word[i + 7] != '_':
                # anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]
                anchor_letter_8 = word[i + 7]

                # calculate occurences of case 8 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.eightgram[letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][anchor_letter_7][
                        anchor_letter_8] > 0 and letter not in self.guessed_letters:
                        total_count += \
                            self.eightgram[letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]
                        letter_count[j] += \
                            self.eightgram[letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                                anchor_letter_6][anchor_letter_7][
                                anchor_letter_8]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between eightgram and sevengram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.7)

        # run the next level down
        return self.sevengram_probs(word)

    def sevengram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a seven-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 6):

            # case 1: "letter letter letter letter letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                # anchor_letter_7 = word[i + 6]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        anchor_letter_6][letter] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6][letter]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6][letter]

            # case 2: "letter letter letter letter letter blank letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] == '_' and word[i + 6] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                # anchor_letter_6 = word[i+5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        letter][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter][anchor_letter_6]

            # case 3: letter letter letter letter blank letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] == '_' and word[i + 5] != '_' and word[i + 6] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                # anchor_letter_5 = word[i+4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                        anchor_letter_5][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_5][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_5][anchor_letter_6]

            # case 4: letter letter letter blank letter letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                # anchor_letter_4 = word[i+3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 4 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                        anchor_letter_5][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]

            # case 5: letter letter blank letter letter letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                # anchor_letter_3 = word[i+2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 5 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sevengram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]
            # case 6: letter blank letter letter letter letter letter
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_':
                anchor_letter_1 = word[i]
                # anchor_letter_2 = word[i+1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 6 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sevengram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]

            # case 7: blank letter letter letter letter letter letter
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_' and word[i + 6] != '_':
                # anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]
                anchor_letter_6 = word[i + 5]
                anchor_letter_7 = word[i + 6]

                # calculate occurences of case 7 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sevengram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5][anchor_letter_6] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sevengram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]
                        letter_count[j] += \
                        self.sevengram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][anchor_letter_6]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between sevengram and sixgram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.6)

            # run the next level down
        return self.sixgram_probs(word)

    def sixgram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a six-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 5):

            # case 1: "letter letter letter letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 4]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if \
                    self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][anchor_letter_5][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter]
                        letter_count[j] += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5][letter]

            # case 2: "letter letter letter letter blank letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] == '_' and word[i + 5] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]
                anchor_letter_5 = word[i + 5]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                        anchor_letter_5] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_5]
                        letter_count[j] += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter][
                            anchor_letter_5]

            # case 3: letter letter letter blank letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_' and word[
                i + 4] != '_' and word[i + 5] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 4]
                anchor_letter_5 = word[i + 5]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                        anchor_letter_5] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                            anchor_letter_5]
                        letter_count[j] += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][anchor_letter_4][
                            anchor_letter_5]

            # case 4: letter letter blank letter letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]
                anchor_letter_5 = word[i + 5]

                # calculate occurences of case 4 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sixgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]
                        letter_count[j] += \
                        self.sixgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]

            # case 5: letter blank letter letter letter letter
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]
                anchor_letter_5 = word[i + 5]

                # calculate occurences of case 5 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sixgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sixgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]
                        letter_count[j] += \
                        self.sixgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]
            # case 6: blank letter letter letter letter letter
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_' and word[i + 5] != '_':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]
                anchor_letter_5 = word[i + 5]

                # calculate occurences of case 6 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.sixgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        anchor_letter_5] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.sixgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]
                        letter_count[j] += \
                        self.sixgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                            anchor_letter_5]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between sevengram and sixgram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.5)

        # run the next level down
        return self.fivegram_probs(word)

    def fivegram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a five-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 4):

            # case 1: "letter letter letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 3]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += \
                        self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter]
                        letter_count[j] += \
                        self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][anchor_letter_4][letter]

            # case 2: "letter letter letter blank letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_' and word[
                i + 4] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter][
                            anchor_letter_4]

            # case 3: letter letter blank letter letter
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3][
                            anchor_letter_4]

            # case 4: letter blank letter letter letter
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of case 4 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]

            # case 5: blank letter letter letter letter
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]
                anchor_letter_4 = word[i + 4]

                # calculate occurences of case 5 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                        anchor_letter_4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]
                        letter_count[j] += self.fivegram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3][
                            anchor_letter_4]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.4)

        # run the next level down
        return self.fourgram_probs(word)

    def fourgram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a four-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 3):

            # case 1: "letter letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 2]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter]
                        letter_count[j] += self.fourgram[anchor_letter_1][anchor_letter_2][anchor_letter_3][letter]

            # case 2: "letter letter blank letter"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter_1][anchor_letter_2][letter][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3]
                        letter_count[j] += self.fourgram[anchor_letter_1][anchor_letter_2][letter][anchor_letter_3]

            # case 3: letter blank letter letter
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter_1][letter][anchor_letter_2][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3]
                        letter_count[j] += self.fourgram[anchor_letter_1][letter][anchor_letter_2][anchor_letter_3]

            # case 4: blank letter letter letter
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]
                anchor_letter_3 = word[i + 3]

                # calculate occurences of case 4 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[letter][anchor_letter_1][anchor_letter_2][
                        anchor_letter_3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3]
                        letter_count[j] += self.fourgram[letter][anchor_letter_1][anchor_letter_2][anchor_letter_3]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.25)

        return self.trigram_probs(word)

    def trigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses tri-gram to calculate the probability of a certain letter appearing in a three-letter sequence for a word of given length
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 2):

            # case 1: "letter letter blank"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 1]

                # calculate occurences of case 1 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[anchor_letter_1][anchor_letter_2][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter_1][anchor_letter_2][letter]
                        letter_count[j] += self.trigram[anchor_letter_1][anchor_letter_2][letter]

            # case 2: "letter blank letter"
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_':
                anchor_letter_1 = word[i]
                anchor_letter_2 = word[i + 2]

                # calculate occurences of case 2 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[anchor_letter_1][letter][
                        anchor_letter_2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter_1][letter][anchor_letter_2]
                        letter_count[j] += self.trigram[anchor_letter_1][letter][anchor_letter_2]

            # case 3: blank letter letter
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_':
                anchor_letter_1 = word[i + 1]
                anchor_letter_2 = word[i + 2]

                # calculate occurences of case 3 and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[letter][anchor_letter_1][
                        anchor_letter_2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[letter][anchor_letter_1][anchor_letter_2]
                        letter_count[j] += self.trigram[letter][anchor_letter_1][anchor_letter_2]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.2)

        # run the next level down
        return self.bigram_probs(word)

    def bigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses bi-gram to calculate the probability of a certain letter appearing in a two-letter sequence for a word of given length
              updates the probabilities set in trigram_probs
        Output: probabilities for each letter to be used in next level
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find either patterns of "letter blank" or "blank letter"
        for i in range(len(word) - 1):
            # case 1: "letter blank"
            if word[i] != '_' and word[i + 1] == '_':
                anchor_letter = word[i]

                # calculate occurences of case 1 and each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.bigram[len(word)][anchor_letter][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][anchor_letter][letter]
                        letter_count[j] += self.bigram[len(word)][anchor_letter][letter]

            # case 2: "blank letter"
            elif word[i] == '_' and word[i + 1] != '_':
                anchor_letter = word[i + 1]

                # calculate occurences of case 2 and each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.bigram[len(word)][letter][anchor_letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][letter][anchor_letter]
                        letter_count[j] += self.bigram[len(word)][letter][anchor_letter]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.1)

        # return letter associated with highest probability
        return self.unigram_probs(word)

    def unigram_probs(self, word):
        '''
        Input: the word in the "clean" format with no spaces and a '_' if letter has not been guessed
        Flow: uses unigram to calculate the probability of a certain letter appearing in a any blank space
              updates the probabilities set in bigram_probs
        Output: letter with the overall highest probability
        '''

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find blank spaces
        for i in range(len(word)):
            # case 1: "letter blank"
            if word[i] == '_':

                # calculate occurences of pattern and each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.unigram[len(word)][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.unigram[len(word)][letter]
                        letter_count[j] += self.unigram[len(word)][letter]

        # calculate the probabilities of each letter appearing
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.05)

        # self.probabilities = probs ## edited code

        # adjust probabilities so they sum to one (not necessary but looks better)
        final_probs = [0] * len(self.letter_list)
        if sum(self.probabilities) > 0:
            for i in range(len(self.probabilities)):
                final_probs[i] = self.probabilities[i] / sum(self.probabilities)

        self.probabilities = final_probs

        # find letter with largest probability
        max_prob = 0
        guess_letter = ''
        for i, letter in enumerate(self.letter_list):
            if self.probabilities[i] > max_prob:
                max_prob = self.probabilities[i]
                guess_letter = letter

        # if no letter chosen from above, pick a random one (extra weight on vowels)
        if guess_letter == '':
            letters = self.letter_list.copy()
            random.shuffle(letters)
            letters_shuffled = ['e', 'a', 'i', 'o', 'u'] + letters
            for letter in letters_shuffled:
                if letter not in self.guessed_letters:
                    return letter

        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def reset(self):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        self.tries_remaining = 6

    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary

        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print(
                    "Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id,
                                                                                                                tries_remains,
                                                                                                                word))
            while tries_remains > 0:
                # get guessed letter from user code
                guess_letter = self.guess(word)

                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))

                try:
                    res = self.request("/guess_letter",
                                       {"request": "guess_letter", "game_id": game_id, "letter": guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e

                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status == "success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status == "failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status == "ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status == "success"

    def my_status(self):
        return self.request("/my_status", {})

    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result


class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)