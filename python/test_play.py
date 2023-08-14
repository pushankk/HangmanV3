from Hangman import HangmanAPI


if __name__ == "__main__":
    hangman = HangmanAPI()

    start_word = "a b a _ _ o n _ _ n t"
    hangman.guess(start_word)