import random

from Hangman import HangmanAPI

guess_count = 5
removal_percent = 0.4

if __name__ == "__main__":

    filepath = 'words_250000_train.txt'
    text_file = open(filepath, "r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()

    chosen_words_list = []
    for i in range(guess_count):
        chosen_word = random.choice(full_dictionary)
        chosen_words_list.append(chosen_word)
        full_dictionary.remove(chosen_word)

    #hangman = HangmanAPI(None, None, None, full_dictionary)

    test_list = []
    for word in chosen_words_list:
        remove_count = int(len(word) * removal_percent)
        original_word = word
        for j in range(remove_count):
            index = random.choice(range(len(word)))
            word = word[:index] + '_' + word[index+1:]
        test_list.append((original_word, word))


    for test_case_tuple in test_list:
        hangman = HangmanAPI(None, None, None, chosen_words_list)
        correct_word = test_case_tuple[0]
        starting_word = test_case_tuple[1]

        output_char = hangman.guess(starting_word)

        if output_char in correct_word:

