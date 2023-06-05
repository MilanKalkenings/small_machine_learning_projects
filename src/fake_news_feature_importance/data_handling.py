import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import regex as re
import string


class DataHandler:
    def __init__(self, max_instances: int, root: str):
        x, y = self.load_x_y(root=root, max_instances=max_instances)
        x_clean = self.clean(x=x)
        x_final = self.engineer_features(x=x_clean)
        x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size=0.2)

        # fit scaler on train, transform on train and test (no data snooping)
        scaler = StandardScaler()
        scaler.fit(X=x_train)
        x_train.loc[:, :] = scaler.transform(x_train)
        x_test.loc[:, :] = scaler.transform(x_test)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def fraction_secure(numerator: float, denominator: float):
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def clean(x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(["date", "subject"], axis=1)

    @staticmethod
    def x_y_split(
            data: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.Series]:
        x = data.drop("is_fake", axis=1)
        y = data["is_fake"]
        return x, y

    @staticmethod
    def words_from_text(text: str) -> List[str]:
        text_lower = text.lower()
        return ''.join(char for char in text_lower if char not in set(string.punctuation)).split()

    @staticmethod
    def count_char(text: str, char: str) -> int:
        return re.subn(fr"\{char}", '', text)[1]

    def get_data_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.x_train, self.y_train

    def get_data_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.x_test, self.y_test

    def engineer_features(self, x: pd.DataFrame) -> pd.DataFrame:
        # univariate transformations:
        cols_done_uni = []
        cols = ["text", "title"]
        for col in cols:
            lexical_diversity = x[col].apply(self.lexical_diversity)
            lexical_diversity.name = col + "_lexical_diversity"
            cols_done_uni.append(lexical_diversity)

            num_words = x[col].apply(self.num_words)
            num_words.name = col + "_num_words"
            cols_done_uni.append(num_words)

            num_exclamation = x[col].apply(self.count_char, args=["!"])
            num_exclamation.name = col + "_num_exclamation"
            cols_done_uni.append(num_exclamation)

            num_question = x[col].apply(self.count_char, args=["?"])
            num_question.name = col + "_num_question"
            cols_done_uni.append(num_question)
        data_uni = pd.concat(cols_done_uni, axis=1)

        # multivariate transformations:
        relative_title_length = data_uni["title_num_words"] / (data_uni["text_num_words"] + data_uni["title_num_words"])
        relative_title_length.name = "relative_title_length"
        return pd.concat([data_uni, relative_title_length], axis=1)

    def lexical_diversity(self, text: str) -> float:
        words = self.words_from_text(text=text)
        return self.fraction_secure(numerator=len(set(words)), denominator=len(words))

    def num_words(self, text: str) -> int:
        words = self.words_from_text(text=text)
        return len(words)

    def load_x_y(
            self,
            root: str,
            max_instances: int) \
            -> Tuple[pd.DataFrame, pd.Series]:
        fake_news = pd.read_csv(root + "/fake.csv")
        real_news = pd.read_csv(root + "/true.csv")
        fake_news["is_fake"] = 1
        real_news["is_fake"] = 0
        data = pd.concat([fake_news, real_news]).sample(frac=1).reset_index(drop=True).head(max_instances)
        x, y = self.x_y_split(data=data)
        return x, y




