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

        full_dictionary_location = "words_250000_train.txt"
        #self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary = train_dict
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()

        self.current_dictionary = []

        # this stores all the unique letters from the full dictionary in sorted order
        self.letter_list = sorted(set("".join(self.full_dictionary)))

        # create a list called probabilities filled with zeros
        # length of this list is determined by the number of elements in the letter_set variable.
        self.probabilities = [0] * len(self.letter_list)

        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram = self.build_ngram_models(
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

        ################################################
        ## Replace with your own "guess" function here #
        ################################################

        # clean the word so that we strip away the space characters
        #clean_word = word[::2]
        clean_word = word

        if len(self.guessed_letters) == 0:
            self.guessed_letters = list(set(''.join(word.replace("_", ""))))

        # a list of incorrect guesses to update the ngrams
        self.incorrect_guesses = list(set(self.guessed_letters) - set(word))

        # reconfiguring only if the last guess was incorrect.
        # this ensures that the analysis adapts and updates based on the user's feedback and the game's progress.
        if len(self.guessed_letters) > 0 and self.guessed_letters[-1] in self.incorrect_guesses:
            self.reoptimize_ngrams()

        # reseting the probabilities to zero from the last guess
        self.probabilities = [0] * len(self.letter_list)

        # run through ngram function
        return self.fivegram_probability(clean_word)

    def build_ngram_models(self, dictionary):

        # create a nested dictionary that stores the occurrences of letter sequences ranging from 1 to 5 characters in length.
        # the nested dictionary will have an additional level to account for the length of each word in unigrams and bigrams.
        # for the unigram level, consider only the unique letters within each word.

        unigram = collections.defaultdict(lambda: collections.defaultdict(int))
        bigram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        trigram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        fourgram = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))
        fivegram = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))))

        # iterating through each word in the dictionary
        # count the occurrences of letter sequences in words from the dictionary and update the n-gram models accordingly.
        for word in dictionary:
            # check each letter in the dictionary and update the ngram
            for i in range(len(word) - 4):
                # We exclude the last four letters of the word because it is searching for patterns of
                # four consecutive letters with a blank in the fifth position. Since the last four letters
                # cannot form such a pattern, there is no need to check them, resulting in improved efficiency
                # and focusing on the relevant parts of the word.

                bigram[len(word)][word[i]][word[i + 1]] += 1
                trigram[word[i]][word[i + 1]][word[i + 2]] += 1
                fourgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1
                fivegram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]][word[i + 4]] += 1

            i = len(word) - 4

            # fill rest of the ngrams for words very small words and complete coverage
            if len(word) == 2:
                bigram[len(word)][word[0]][word[1]] += 1
            elif len(word) == 3:
                bigram[len(word)][word[0]][word[1]] += 1
                bigram[len(word)][word[1]][word[2]] += 1
                trigram[word[0]][word[1]][word[2]] += 1
            # fill out rest of the fourgrams
            elif len(word) >= 4:
                bigram[len(word)][word[i]][word[i + 1]] += 1
                bigram[len(word)][word[i + 1]][word[i + 2]] += 1
                bigram[len(word)][word[i + 2]][word[i + 3]] += 1
                trigram[word[i]][word[i + 1]][word[i + 2]] += 1
                trigram[word[i + 1]][word[i + 2]][word[i + 3]] += 1
                fourgram[word[i]][word[i + 1]][word[i + 2]][word[i + 3]] += 1

            # fill out unigrams
            for letter in set(word):
                unigram[len(word)][letter] += 1

        return unigram, bigram, trigram, fourgram, fivegram

    def reoptimize_ngrams(self):

        # regulates the ngrams after removing any incorrectly guessed letters
        # updates the dictionary to eliminate words containing incorrectly guessed letters
        new_dictionary = [word for word in self.full_dictionary if
                          not set(word).intersection(set(self.incorrect_guesses))]
        self.unigram, self.bigram, self.trigram, self.fourgram, self.fivegram = self.build_ngram_models(new_dictionary)

    def fivegram_probability(self, word):

        # given an input word in a clean format with no spaces and placeholders ('_') for unknown letters,
        # the process utilizes five-grams to determine the likelihood of a specific letter appearing in a five-letter sequence for a word.
        # the output provides the probabilities for each letter, which will be utilized in the subsequent stage.

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have four consecutive letters where one of them is blank
        for i in range(len(word) - 4):
            # We exclude the last four letters of the word because it is searching for patterns of
            # four consecutive letters with a blank in the fifth position. Since the last four letters
            # cannot form such a pattern, there is no need to check them, resulting in improved efficiency
            # and focusing on the relevant parts of the word.

            # case 1: "eg word:  xyzw_ "
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] == '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]
                anchor_letter3 = word[i + 2]
                anchor_letter4 = word[i + 3]

                # calculate occurences of "anchor_letter1 anchor_letter2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][anchor_letter4][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][anchor_letter4][letter]
                        letter_count[j] += self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][anchor_letter4][letter]

            # case 2: "eg word: xyz_w "
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_' and word[i + 4] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]
                anchor_letter3 = word[i + 2]
                anchor_letter4 = word[i + 4]

                # calculate occurences of "anchor_letter1 blank anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][letter][anchor_letter4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][letter][anchor_letter4]
                        letter_count[j] += self.fivegram[anchor_letter1][anchor_letter2][anchor_letter3][letter][anchor_letter4]

            # case 3: "eg word: xy_zw "
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]
                anchor_letter3 = word[i + 3]
                anchor_letter4 = word[i + 4]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter1][anchor_letter2][letter][anchor_letter3][
                        anchor_letter4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter1][anchor_letter2][letter][anchor_letter3][
                            anchor_letter4]
                        letter_count[j] += self.fivegram[anchor_letter1][anchor_letter2][letter][anchor_letter3][
                            anchor_letter4]

            # case 4: "eg word: x_yzw"
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 2]
                anchor_letter3 = word[i + 3]
                anchor_letter4 = word[i + 4]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[anchor_letter1][letter][anchor_letter2][anchor_letter3][
                        anchor_letter4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[anchor_letter1][letter][anchor_letter2][anchor_letter3][
                            anchor_letter4]
                        letter_count[j] += self.fivegram[anchor_letter1][letter][anchor_letter2][anchor_letter3][
                            anchor_letter4]

            # case 5: "eg word: _xyzw"
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_' and word[
                i + 4] != '_':
                anchor_letter1 = word[i + 1]
                anchor_letter2 = word[i + 2]
                anchor_letter3 = word[i + 3]
                anchor_letter4 = word[i + 4]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fivegram[letter][anchor_letter1][anchor_letter2][anchor_letter3][
                        anchor_letter4] > 0 and letter not in self.guessed_letters:
                        total_count += self.fivegram[letter][anchor_letter1][anchor_letter2][anchor_letter3][
                            anchor_letter4]
                        letter_count[j] += self.fivegram[letter][anchor_letter1][anchor_letter2][anchor_letter3][
                            anchor_letter4]

        # calculate the probabilities of each letter
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between fivegram and fourgram
        """
       The step of multiplying each probability in probs by 0.40 and adding it to
       the corresponding probability in self.probabilities depicts interpolation.
       It is performed to combine the probabilities obtained from the fivegram level with the
       existing probabilities from the previous levels (trigram and bigram).This interpolation
       helps to balance the influence of higher-level ngrams (trigrams and bigrams) with the
       more specific information provided by the fivegram model. The method assigns a higher weight
       to the probabilities derived from the fivegram model. The factor of 0.40 determines the
       weight assigned to the fivegram probabilities, while the remaining weight (0.60) is assigned
       to the existing probabilities in self.probabilities. Overall, the interpolation step helps in
       combining the information from different ngram models to make more accurate predictions about
       the likelihood of specific letters appearing in the target blank space, considering both local
       and global patterns in the word.
        """

        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.40)

        # go for the next level
        return self.fourgram_probability(word)

    def fourgram_probability(self, word):

        # given a word in a clean format without spaces and placeholders ('_') for unknown letters,
        # the process utilizes four-grams to determine the probabilities of specific letters appearing in a four-letter sequence for a word.
        # the output provides the probabilities for each letter, which will be utilized in the next stage.

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # calculates the probabilities of each letter in a word based on its context using a four-gram model.
        # It considers different cases based on the positions of underscores (_) in the word and updates the letter probabilities accordingly.
        # The probabilities are then interpolated with the existing probabilities from lower-level n-gram models (trigram and bigram)
        # to balance the influence of higher-level n-grams. The function then proceeds to the next level of the n-gram model to further
        # calculate the probabilities.

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 3):

            # case 1: "eg word: abc_"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] == '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]
                anchor_letter3 = word[i + 2]

                # calculate occurences of "anchor_letter1 anchor_letter2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter1][anchor_letter2][anchor_letter3][
                        letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter1][anchor_letter2][anchor_letter3][letter]
                        letter_count[j] += self.fourgram[anchor_letter1][anchor_letter2][anchor_letter3][letter]

            # case 2:  "eg word: ab_c"
            elif word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_' and word[i + 3] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]
                anchor_letter3 = word[i + 3]

                # calculate occurences of "anchor_letter1 blank anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter1][anchor_letter2][letter][
                        anchor_letter3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter1][anchor_letter2][letter][anchor_letter3]
                        letter_count[j] += self.fourgram[anchor_letter1][anchor_letter2][letter][anchor_letter3]

            # case 3: "eg word: a_bc"
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_' and word[i + 3] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 2]
                anchor_letter3 = word[i + 3]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[anchor_letter1][letter][anchor_letter2][
                        anchor_letter3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[anchor_letter1][letter][anchor_letter2][anchor_letter3]
                        letter_count[j] += self.fourgram[anchor_letter1][letter][anchor_letter2][anchor_letter3]

            # case 4:  "eg word: _abc"
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_' and word[i + 3] != '_':
                anchor_letter1 = word[i + 1]
                anchor_letter2 = word[i + 2]
                anchor_letter3 = word[i + 3]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.fourgram[letter][anchor_letter1][anchor_letter2][
                        anchor_letter3] > 0 and letter not in self.guessed_letters:
                        total_count += self.fourgram[letter][anchor_letter1][anchor_letter2][anchor_letter3]
                        letter_count[j] += self.fourgram[letter][anchor_letter1][anchor_letter2][anchor_letter3]

        # calculate the probabilities of each letter
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram

        """
        Multiply each probability in probs by 0.25 and add it to the corresponding probability in self.probabilities.
        This interpolation step combines the probabilities obtained from the trigram model with the existing
        probabilities from the previous levels (trigram and bigram). It balances the influence of higher-level
        ngrams with the more specific information provided by the fourgram model.
        """
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.25)

        # go for the next level
        return self.trigram_probability(word)

    def trigram_probability(self, word):

        # given a word in a clean format without spaces and placeholders ('_') for unknown letters,
        # the process utilizes tri-grams to determine the probabilities of specific letters appearing in a three-letter sequence for a word.
        # the output provides the probabilities for each letter, which will be utilized in the next stage.

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find patterns that have three consecutive letters where one of them is blank
        for i in range(len(word) - 2):

            # case 1: "eg word: ab_"
            if word[i] != '_' and word[i + 1] != '_' and word[i + 2] == '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 1]

                # calculate occurences of "anchor_letter1 anchor_letter2 blank" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[anchor_letter1][anchor_letter2][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter1][anchor_letter2][letter]
                        letter_count[j] += self.trigram[anchor_letter1][anchor_letter2][letter]

            # case 2: "eg word: a_b"
            elif word[i] != '_' and word[i + 1] == '_' and word[i + 2] != '_':
                anchor_letter1 = word[i]
                anchor_letter2 = word[i + 2]

                # calculate occurences of "anchor_letter1 blank anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[anchor_letter1][letter][anchor_letter2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[anchor_letter1][letter][anchor_letter2]
                        letter_count[j] += self.trigram[anchor_letter1][letter][anchor_letter2]

            # case 3: "eg word: _ab"
            elif word[i] == '_' and word[i + 1] != '_' and word[i + 2] != '_':
                anchor_letter1 = word[i + 1]
                anchor_letter2 = word[i + 2]

                # calculate occurences of "blank anchor_letter1 anchor_letter2" and for each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.trigram[letter][anchor_letter1][anchor_letter2] > 0 and letter not in self.guessed_letters:
                        total_count += self.trigram[letter][anchor_letter1][anchor_letter2]
                        letter_count[j] += self.trigram[letter][anchor_letter1][anchor_letter2]

        # calculate the probabilities of each letter
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between trigram and bigram

        """
        Multiply each probability in probs by 0.20 and add it to the corresponding probability in self.probabilities.
        This interpolation step combines the probabilities obtained from the trigram model with the existing
        probabilities from the previous levels. It balances the influence of higher-level
        ngrams with the more specific information provided by the trigram model.
        """
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.20)

        # go for the next level
        return self.bigram_probability(word)

    def bigram_probability(self, word):

        # given a word in a clean format without spaces and placeholders ('_') for unknown letters,
        # the process utilizes bi-grams to determine the probabilities of specific letters appearing in a two-letter sequence for a word.
        # the output provides the probabilities for each letter, which will be used in the next stage.

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find either patterns of "letter blank" or "blank letter"
        for i in range(len(word) - 1):
            # case 1: "eg word: a_"
            if word[i] != '_' and word[i + 1] == '_':
                anchor_letter = word[i]

                # calculate occurences of "anchor_letter blank" and each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.bigram[len(word)][anchor_letter][letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][anchor_letter][letter]
                        letter_count[j] += self.bigram[len(word)][anchor_letter][letter]

            # case 2: "eg word: _a"
            elif word[i] == '_' and word[i + 1] != '_':
                anchor_letter = word[i + 1]

                # calculate occurences of "blank anchor_letter" and each letter not guessed yet
                for j, letter in enumerate(self.letter_list):
                    if self.bigram[len(word)][letter][anchor_letter] > 0 and letter not in self.guessed_letters:
                        total_count += self.bigram[len(word)][letter][anchor_letter]
                        letter_count[j] += self.bigram[len(word)][letter][anchor_letter]

        # calculate the probabilities of each letter
        if total_count > 0:
            for i in range(len(self.letter_list)):
                probs[i] = letter_count[i] / total_count

        # interpolate probabilities between bigram and unigram
        for i, p in enumerate(self.probabilities):
            self.probabilities[i] = p + probs[i] * (0.10)

        # return letter associated with highest probability
        return self.unigram_probability(word)

    def unigram_probability(self, word):

        # given a word in a clean format without spaces and placeholders ('_') for unknown letters,
        # the process utilizes unigrams to calculate the probabilities of specific letters appearing in any blank space.
        # The output provides the letter with the highest overall probability.

        # vector of probabilities for each letter
        probs = [0] * len(self.letter_list)

        total_count = 0
        letter_count = [0] * len(self.letter_list)

        # traverse the word and find blank spaces
        for i in range(len(word)):
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

        # adjust probabilities so they sum to one
        final_probs = [0] * len(self.letter_list)
        if sum(self.probabilities) > 0:
            for i in range(len(self.probabilities)):
                final_probs[i] = self.probabilities[i] / sum(self.probabilities)

        self.probabilities = final_probs

        # finding letter with highest probability
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