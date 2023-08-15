import random

from Hangman import HangmanAPI
import numpy as np

iterations = 5
games_per_iteration = 50
removal_percent = 1.0

def remove_char_from_word(word, char_set):
    for char in char_set:
        word = word.replace(char, "_")
    return word

if __name__ == "__main__":

    filepath = 'words_250000_train.txt'
    text_file = open(filepath, "r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()

    chosen_words_list = []
    for j in range(iterations):
        chosen_words_for_iter = []
        for i in range(games_per_iteration):

            # Don't pick the word if >4 unique letters
            chosen_word = random.choice(full_dictionary)
            while len(set(list(chosen_word))) > 4:
                chosen_word = random.choice(full_dictionary)

            chosen_words_for_iter.append(chosen_word)
            full_dictionary.remove(chosen_word)
        chosen_words_list.append(chosen_words_for_iter)

    test_list = []
    for word_list in chosen_words_list:
        test_list_for_iter = []
        for word in word_list:
            chars_in_word = list(set(list(word)))
            remove_count = int(len(chars_in_word) * removal_percent)
            original_word = word
            removed_char_set= set()
            for j in range(remove_count):
                char_to_remove = random.choice(chars_in_word)
                word = word.replace(char_to_remove, "_")
                chars_in_word.remove(char_to_remove)
                removed_char_set.add(char_to_remove)
            test_list_for_iter.append((original_word, word, removed_char_set))
        test_list.append(test_list_for_iter)

    hangman = HangmanAPI(train_dict=full_dictionary)
    success_rates = []
    for itr in range(iterations):
        tests_for_this_itr = test_list[itr]
        success_count = 0
        for test in tests_for_this_itr:
            print(f"Starting iteration:{itr} test : {test}")
            hangman.reset()
            correct_word = test[0]
            current_word = test[1]
            remaining_chars = test[2]

            success = False
            for iattempt in range(6):
                guess_char = hangman.guess(current_word)
                hangman.guessed_letters.append(guess_char)
                # Correct guess
                if guess_char in remaining_chars:
                    remaining_chars.remove(guess_char)
                    current_word = remove_char_from_word(correct_word, remaining_chars)
                    if len(remaining_chars) == 0:
                        success = True
                        break


            if success:
                success_count += 1
                print(f"**Successful**, correct_word:{correct_word}, final_word:{current_word}")
            else:
                print(f">> Failed, correct_word:{correct_word}, final_word:{current_word}")

        success_rate = success_count / len(tests_for_this_itr) * 100.0
        success_rates.append(success_rate)
        print(f"Success rate iter {itr}:{success_rate}")

    # Calculate mean success rate
    mean_success_rate = np.mean(success_rates)
    print(f"Mean Success Rate : {mean_success_rate}")








