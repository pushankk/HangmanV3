import random

from Hangman import HangmanAPI

games_count = 50
removal_percent = 0.6

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
    for i in range(games_count):
        chosen_word = random.choice(full_dictionary)
        chosen_words_list.append(chosen_word)
        full_dictionary.remove(chosen_word)

    test_list = []
    for word in chosen_words_list:
        chars_in_word = list(set(list(word)))
        remove_count = int(len(chars_in_word) * removal_percent)
        original_word = word
        removed_char_set= set()
        for j in range(remove_count):
            char_to_remove = random.choice(chars_in_word)
            word = word.replace(char_to_remove, "_")
            chars_in_word.remove(char_to_remove)
            removed_char_set.add(char_to_remove)
        test_list.append((original_word, word, removed_char_set))

    hangman = HangmanAPI(train_dict=full_dictionary)
    success_count = 0
    for test in test_list:
        print(f"Starting test : {test}")
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

    success_rate = success_count / len(test_list) * 100.0;
    print(f"Success rate:{success_rate}")








